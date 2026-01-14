use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

pub enum InferenceEvent {
    Output(String),
    Ready,
    Error(String),
    Exit,
    SoulLevel(u64),
}

pub struct InferenceSession {
    pub active_process: Option<Child>,
    pub input_tx: Option<Sender<String>>,
    pub event_rx: Receiver<InferenceEvent>,
    pub is_dreaming: bool,
}

impl InferenceSession {
    pub fn new() -> Self {
        let (_, rx) = channel();
        Self {
            active_process: None,
            input_tx: None,
            event_rx: rx,
            is_dreaming: false,
        }
    }

    pub fn is_active(&self) -> bool {
        self.active_process.is_some()
    }

    pub fn spawn(&mut self, model_path: &str, temp: f64, max_tokens: usize) -> anyhow::Result<()> {
        let exe = std::env::current_exe()?;
        let mut command = Command::new(exe);
        command
            .arg("inference")
            .arg("--model")
            .arg(model_path)
            .arg("--temp")
            .arg(temp.to_string())
            .arg("--max-tokens")
            .arg(max_tokens.to_string())
            .stdout(Stdio::piped())
            .stdin(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command.spawn()?;

        // Channels
        let (event_tx, event_rx) = channel();
        self.event_rx = event_rx;
        let (in_tx, in_rx) = channel::<String>();
        self.input_tx = Some(in_tx);

        // Stdin Thread
        let mut stdin = child.stdin.take().unwrap();
        thread::spawn(move || {
            while let Ok(msg) = in_rx.recv() {
                if writeln!(stdin, "{}", msg).is_err() {
                    break;
                }
                if stdin.flush().is_err() {
                    break;
                }
            }
        });

        // Stdout Thread (Streaming)
        let mut stdout = child.stdout.take().unwrap();
        let ev_tx_out = event_tx.clone();
        thread::spawn(move || {
            let mut buffer = [0u8; 1024];
            let re_ansi = regex::Regex::new(r"\x1B\[([0-9]{1,2}(;[0-9]{1,2})*)?m").unwrap();
            let re_soul = regex::Regex::new(r"Soul Level: (\d+)").unwrap();

            loop {
                match stdout.read(&mut buffer) {
                    Ok(0) => {
                        let _ = ev_tx_out.send(InferenceEvent::Exit);
                        break;
                    }
                    Ok(n) => {
                        let chunk = &buffer[0..n];
                        // Using lossy conversion to avoid panics on partial UTF-8 bytes at boundary
                        // Ideally we'd buffer, but for now this is robust enough for logs
                        let s = String::from_utf8_lossy(chunk);

                        // 1. Strip ANSI codes
                        let s_no_ansi = re_ansi.replace_all(&s, "");

                        // 2. Parse Soul Level
                        if let Some(caps) = re_soul.captures(&s_no_ansi) {
                            if let Some(m) = caps.get(1) {
                                if let Ok(lvl) = m.as_str().parse::<u64>() {
                                    let _ = ev_tx_out.send(InferenceEvent::SoulLevel(lvl));
                                }
                            }
                        }

                        // 3. Filter Garbage
                        let s_clean: String = s_no_ansi
                            .chars()
                            .filter(|c| {
                                if *c == '\n' || *c == '\r' || *c == '\t' {
                                    return true;
                                }
                                if c.is_control() {
                                    return false;
                                }
                                if *c == '\u{FFFD}' {
                                    return false;
                                }
                                true
                            })
                            .collect();

                        if !s_clean.is_empty() {
                            let _ = ev_tx_out.send(InferenceEvent::Output(s_clean));
                        }
                    }
                    Err(e) => {
                        let _ = ev_tx_out.send(InferenceEvent::Error(e.to_string()));
                        break;
                    }
                }
            }
        });

        // Stderr Thread (Control Signals & Error Catching)
        let stderr = child.stderr.take().unwrap();
        let ev_tx_err = event_tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for l in reader.lines() {
                if let Ok(line) = l {
                    let trimmed = line.trim();
                    if trimmed == "<<READY>>" {
                        let _ = ev_tx_err.send(InferenceEvent::Ready);
                    } else if !trimmed.is_empty() {
                        // Distinguish real errors from informational STDERR (tracing, debug logs)
                        let lower = trimmed.to_lowercase();
                        let is_real_error = lower.contains("error")
                            || lower.contains("panic")
                            || lower.contains("failed")
                            || lower.contains("fatal")
                            || lower.contains("abort");

                        // Skip certain noisy but harmless messages
                        let is_info_noise = lower.contains("portable mode")
                            || lower.contains("cwd set to")
                            || trimmed.starts_with("📍");

                        if is_real_error && !is_info_noise {
                            let _ = ev_tx_err.send(InferenceEvent::Error(line));
                        } else {
                            // Just show as normal output (will appear without scary "Error:" prefix)
                            let _ = ev_tx_err.send(InferenceEvent::Output(format!("{}\n", line)));
                        }
                    }
                } else {
                    break;
                }
            }
        });

        self.active_process = Some(child);
        Ok(())
    }

    pub fn send_message(&self, text: &str) {
        if let Some(tx) = &self.input_tx {
            let _ = tx.send(text.to_string());
        }
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.active_process.take() {
            let _ = child.kill();
        }
        self.input_tx = None;
    }
}

impl Default for InferenceSession {
    fn default() -> Self {
        Self::new()
    }
}
