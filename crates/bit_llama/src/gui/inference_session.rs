use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

pub enum InferenceEvent {
    Output(String),
    Ready,
    Error(String),
    Exit,
}

pub struct InferenceSession {
    pub active_process: Option<Child>,
    pub input_tx: Option<Sender<String>>,
    pub event_rx: Receiver<InferenceEvent>,
}

impl InferenceSession {
    pub fn new() -> Self {
        let (_, rx) = channel();
        Self {
            active_process: None,
            input_tx: None,
            event_rx: rx,
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

        // Windows: Hide console is managed by main.rs attribute?
        // If release build, it has no console. Pipes work.
        // If debug build, console might appear or not depending on config.
        // We want hidden. But current_exe inherits?
        // If parent is GUI (hidden), child is typically hidden if piped.

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
                if let Err(_) = writeln!(stdin, "{}", msg) {
                    break;
                }
            }
        });

        // Stdout Thread (Streaming)
        let mut stdout = child.stdout.take().unwrap();
        let ev_tx_out = event_tx.clone();
        thread::spawn(move || {
            let mut buffer = [0u8; 1024];
            loop {
                match stdout.read(&mut buffer) {
                    Ok(0) => {
                        let _ = ev_tx_out.send(InferenceEvent::Exit);
                        break;
                    }
                    Ok(n) => {
                        let chunk = &buffer[0..n];
                        // Converting chunk to string.
                        // Note: If a multibyte char is split, this is an issue.
                        // Ideally we buffer incomplete bytes.
                        // For MVP, we rely on token-based flushing from sender.
                        let s = String::from_utf8_lossy(chunk).to_string();
                        let _ = ev_tx_out.send(InferenceEvent::Output(s));
                    }
                    Err(e) => {
                        let _ = ev_tx_out.send(InferenceEvent::Error(e.to_string()));
                        break;
                    }
                }
            }
        });

        // Stderr Thread (Control Signals)
        let stderr = child.stderr.take().unwrap();
        let ev_tx_err = event_tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(l) = line {
                    if l.trim() == "<<READY>>" {
                        let _ = ev_tx_err.send(InferenceEvent::Ready);
                    } else if !l.trim().is_empty() {
                        // Optional: Forward other logs as errors or ignored
                    }
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
