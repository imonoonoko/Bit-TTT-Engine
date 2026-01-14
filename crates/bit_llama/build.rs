fn main() {
    if cfg!(target_os = "windows") {
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        // Optional: Set other properties like version info if needed
        // res.set("FileDescription", "Bit-Llama Studio");

        if let Err(e) = res.compile() {
            eprintln!("Error compiling windows resources: {}", e);
        }
    }
}
