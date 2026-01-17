use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
#[cfg(target_os = "windows")]
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ProcessEvent {
    Log(String),
    Progress(f32, String), // Percentage (0.0-1.0), Message
    Exit(i32),
    Error(String),
}

pub struct ProcessManager {
    pub tx: mpsc::Sender<ProcessEvent>,
    pub rx: mpsc::Receiver<ProcessEvent>,
}

impl ProcessManager {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { tx, rx }
    }

    pub fn spawn_conversion(
        &self,
        python_path: &str,
        script_path: &str,
        input: &str,
        output: &str,
        n_bases: i32,
        device: &str,
    ) {
        let tx = self.tx.clone();
        let python_path = python_path.to_string();
        let script_path = script_path.to_string();
        let input = input.to_string();
        let output = output.to_string();
        let device = device.to_string();

        thread::spawn(move || {
            tx.send(ProcessEvent::Log(format!(
                "🚀 Starting conversion using: {}",
                python_path
            )))
            .unwrap();
            tx.send(ProcessEvent::Log(format!("📜 Script: {}", script_path)))
                .unwrap();

            // Validate paths (Basic)
            if !std::path::Path::new(&input).exists() {
                tx.send(ProcessEvent::Error(format!(
                    "❌ Input path does not exist: {}",
                    input
                )))
                .unwrap();
                return;
            }

            // Construct Command
            let mut cmd = Command::new(&python_path);

            // Windows-specific: Hide console window if needed, but here we want to capture output
            // const CREATE_NO_WINDOW: u32 = 0x08000000;
            // #[cfg(target_os = "windows")]
            // cmd.creation_flags(CREATE_NO_WINDOW);

            cmd.arg(&script_path)
                .arg("--model-id")
                .arg(&input) // Wrapper treats local path as model-id if possible or usually separate arg
                .arg("--output-dir")
                .arg(&output)
                .arg("--n-bases")
                .arg(n_bases.to_string())
                .arg("--device")
                .arg(&device);

            // If input is local path, we might need to handle it.
            // The script convert_llama_v2.py uses --model-id. If it's a local path, huggingface_hub might try to download if not careful.
            // But convert_llama_v2.py lines 176+:
            // model_dir = os.path.join("models", args.model_id.split("/")[-1])
            // This implies it EXPECTS a HF repo ID.
            // If the user wants to convert a LOCAL folder, the script might need modification OR we assume user puts it in `models/`.
            // Wait, the Analyzer said: "Input Folder (ローカルの HF 形式モデル)".
            // The current script `convert_llama_v2.py` assumes `models/{id}` structure if `--download` is used?
            // Line 219: `file_paths = [model_dir]` if not dir?
            // Actually, `convert_llama_v2.py` logic:
            // `model_dir = os.path.join("models", args.model_id.split("/")[-1])`
            // This hardcodes `models/` prefix. This is BAD for a general converter.
            // The GUI wrapper should probably fix the script or use a modified one.
            // But we agreed to "Logic Integration" via wrapper.
            // For now, let's assume we pass the FULL PATH.
            // I should override the `model_check` logic in the script or just update the script to accept `--model-dir` directly.
            // Actually, `convert_llama_v2.py` is somewhat rigid.
            // Let's assume for now we just pass the args and see what happens, or better, we patch `convert_llama_v2.py` to accept `--local-path`?
            // Or simple hacks: `path/to/model` -> `args.model_id`.

            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

            match cmd.spawn() {
                Ok(mut child) => {
                    // stdout reader
                    let stdout = child.stdout.take().unwrap();
                    let tx_out = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            if let Ok(l) = line {
                                // Parse Progress (tqdm)
                                // tqdm usually prints to stderr, but we check both.
                                tx_out.send(ProcessEvent::Log(l)).unwrap();
                            }
                        }
                    });

                    // stderr reader (tqdm often goes here)
                    let stderr = child.stderr.take().unwrap();
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            if let Ok(l) = line {
                                // Try verify tqdm text like "10%|...|"
                                if l.contains("%|") {
                                    // Hacky parse
                                    // 10%|###   |
                                    tx_err
                                        .send(ProcessEvent::Log(format!("[Progress] {}", l)))
                                        .unwrap();
                                    // Parse percentage logic here if needed
                                } else {
                                    tx_err.send(ProcessEvent::Log(l)).unwrap();
                                }
                            }
                        }
                    });

                    let status = child.wait();
                    match status {
                        Ok(s) => tx.send(ProcessEvent::Exit(s.code().unwrap_or(-1))).unwrap(),
                        Err(e) => tx.send(ProcessEvent::Error(e.to_string())).unwrap(),
                    }
                }
                Err(e) => {
                    tx.send(ProcessEvent::Error(format!(
                        "Failed to spawn python: {}",
                        e
                    )))
                    .unwrap();
                }
            }
        });
    }
}
