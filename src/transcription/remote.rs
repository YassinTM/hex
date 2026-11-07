//! Remote transcription (YassinTM fork patch).
//!
//! When `HEX_REMOTE_TRANSCRIBE_URL` is set (for example
//! `http://gamingpc:8787/transcribe`), a finished offline clip is sent to that
//! server and its reply is used as the transcript. The request body is the
//! 16 kHz mono clip as little-endian signed 16-bit PCM; the reply is the plain
//! UTF-8 transcript. Any failure (unset, unreachable, timeout, non-2xx) returns
//! `None` and the caller transcribes locally as usual, so dictation never
//! depends on the server being up.
//!
//! The only hook into upstream code is the `try_transcribe` call in
//! `engine.rs`'s offline `finish` closure; the test at the bottom fails if
//! that call disappears.

use std::time::{Duration, Instant};

pub(crate) const URL_ENV: &str = "HEX_REMOTE_TRANSCRIBE_URL";
const SAMPLE_RATE: f64 = 16_000.0;

/// The configured server URL, if any.
pub(crate) fn url() -> Option<String> {
    std::env::var(URL_ENV).ok().map(|value| value.trim().to_owned()).filter(|value| !value.is_empty())
}

/// Transcribes `clip` (16 kHz mono f32) on the configured server, or returns
/// `None` so the caller falls back to the local model.
pub(crate) fn try_transcribe(clip: &[f32]) -> Option<String> {
    let url = url()?;
    let started = Instant::now();
    match transcribe_at(&url, clip, timeout_for(clip.len())) {
        Ok(text) => {
            tracing::info!(
                audio_ms = (clip.len() as f64 / SAMPLE_RATE * 1000.0) as u64,
                remote_ms = started.elapsed().as_millis() as u64,
                "transcribed on the remote server"
            );
            Some(text)
        }
        Err(error) => {
            tracing::warn!(%error, elapsed_ms = started.elapsed().as_millis() as u64,
                "remote transcription failed; transcribing locally");
            None
        }
    }
}

/// Generous enough for a slow upload, short enough that a dead server costs
/// little before the local fallback: 4 s plus a tenth of the audio length.
fn timeout_for(samples: usize) -> Duration {
    Duration::from_secs_f64(4.0 + samples as f64 / SAMPLE_RATE * 0.1)
}

fn encode(clip: &[f32]) -> Vec<u8> {
    let mut body = Vec::with_capacity(clip.len() * 2);
    for &sample in clip {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        body.extend_from_slice(&value.to_le_bytes());
    }
    body
}

pub(crate) fn transcribe_at(url: &str, clip: &[f32], timeout: Duration) -> Result<String, String> {
    let agent = crate::net::agent(url, |config| {
        config
            .timeout_connect(Some(Duration::from_millis(1500)))
            .timeout_global(Some(timeout))
            .http_status_as_error(true)
    });
    let body = encode(clip);
    let mut response = agent
        .post(url)
        .header("content-type", "application/octet-stream")
        .header("x-hex-sample-rate", "16000")
        .send(&body[..])
        .map_err(|error| error.to_string())?;
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|error| error.to_string())?;
    Ok(text.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    /// One-shot fake server: checks the body length and answers `reply`.
    fn serve_once(reply: &'static str, status: u16) -> (String, std::thread::JoinHandle<usize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/transcribe", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0u8; length];
            reader.read_exact(&mut body).unwrap();
            let mut stream = reader.into_inner();
            write!(
                stream,
                "HTTP/1.1 {status} X\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                reply.len()
            )
            .unwrap();
            stream.flush().unwrap();
            length
        });
        (url, handle)
    }

    #[test]
    fn sends_pcm16_and_returns_the_reply() {
        let (url, server) = serve_once("  hello from the gaming pc \n", 200);
        let clip = vec![0.25f32; 16_000];
        let text = transcribe_at(&url, &clip, Duration::from_secs(5)).unwrap();
        assert_eq!(text, "hello from the gaming pc");
        assert_eq!(server.join().unwrap(), clip.len() * 2);
    }

    #[test]
    fn server_errors_are_failures_so_the_caller_falls_back() {
        let (url, server) = serve_once("boom", 500);
        assert!(transcribe_at(&url, &[0.0; 160], Duration::from_secs(5)).is_err());
        server.join().unwrap();
    }

    #[test]
    fn unreachable_server_fails_fast() {
        let started = Instant::now();
        assert!(transcribe_at("http://127.0.0.1:9/transcribe", &[0.0; 160], Duration::from_secs(5)).is_err());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn encode_clamps_and_scales() {
        let body = encode(&[1.5, -1.5, 0.0]);
        assert_eq!(body, [0xff, 0x7f, 0x01, 0x80, 0x00, 0x00]);
    }

    /// Guards the fork's only upstream hook: the offline finish path must ask
    /// the remote server before the local model.
    #[test]
    fn engine_offline_finish_calls_the_remote_hook() {
        let engine = include_str!("engine.rs");
        let hook = engine
            .find("remote::try_transcribe(")
            .expect("engine.rs no longer calls remote::try_transcribe; re-add the hook in the offline finish closure");
        let local = engine[hook..]
            .find(".transcribe(")
            .expect("the local model call should follow the remote hook");
        assert!(local > 0);
    }
}
