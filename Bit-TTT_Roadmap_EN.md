Bit-TTT Engineering Roadmap

This document visualizes the engineering process of the Bit-TTT project and the technical dependencies of each task.

🗺️ Roadmap Flowchart

graph TD
    %% Class Defaults
    classDef done fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef urgent fill:#f8d7da,stroke:#dc3545,stroke-width:4px,color:#721c24;
    classDef next fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
    classDef future fill:#e2e3e5,stroke:#adb5bd,stroke-width:1px,color:#383d41;

    %% Node Definitions
    Start((Phase A<br/>Done)):::done

    subgraph Phase_B0 [🛑 Phase B-0: Structural Fix]
        B0[Fix Command Invocation<br/>cargo run → current_exe]:::urgent
        noteB0[Required for<br/>distribution]
        B0 -.- noteB0
    end

    subgraph Phase_B [🚧 Phase B: Integration]
        B1[Log Streaming<br/>state.rs → GUI Console]:::next
        B2[Config Sync<br/>Config → CLI Args]:::next
        B3[Error Handling<br/>Process Monitoring/Stopping]:::next
    end

    subgraph Phase_C [📊 Phase C: Visualization]
        C1[Learning Curve<br/>Loss Parsing & Plot]:::future
        C2[VRAM Monitoring<br/>Real usage check]:::future
    end

    subgraph Phase_D [⚡ Phase D: Completion]
        D1[Inference Playground<br/>Chat UI]:::future
        D2[Release Build<br/>Distribution Package]:::future
    end

    %% Dependencies
    Start ==> B0
    B0 ==> B1
    B0 --> B2
    
    B1 ==> C1
    B1 --> B3
    
    C1 --> D1
    B3 --> D1
    
    B2 --> D2
    C2 --> D2

    %% Style Application
    linkStyle 0,1,3 stroke-width:4px,stroke:#dc3545,fill:none;


📝 Phase Explanation

🛑 Phase B-0: Structural Fix

Correct the dependency on the development environment (`cargo run`) for app execution.

Fix Command Invocation: Modify external process launch logic in `ui.rs` to use `current_exe()` and invoke own binary as a subcommand. This ensures operation in user environments without Python or Rust toolchains.

🚧 Phase B: Integration

Deepen the integration between Rust (GUI) and Python (Learning Core) to ensure application stability.

Log Streaming: Display standard output from background learning processes in the GUI console in real-time.

Config Sync: Ensure parameters set in the GUI (Learning Rate, Steps, etc.) are reliably passed as CLI arguments.

Error Handling: Appropriately display error messages without freezing the GUI during OOM (Out of Memory) or exceptions.

📊 Phase C: Visualization

Enhance visual feedback beyond text logs.

Learning Curve: Parse Loss values from logs and draw a real-time line graph.

VRAM Monitoring: Monitor memory usage and display indicators to prevent crashes.

⚡ Phase D: Completion

Finalize as an all-in-one LLM development studio.

Inference Playground: Add a tab to load the model currently being trained and test chat interactions.

Release Build: Create the final release build and a distributable package.
