# Dependency Map: GUI Enhancement

```mermaid
graph TD
    User([User]) -->|Interacts| UI[GUI Layer (ui.rs)]
    UI -->|Updates| App[BitStudioApp (mod.rs)]
    
    subgraph State Management
        App -->|Owns| PState[ProjectState (state.rs)]
        App -->|Owns| Config[ProjectConfig (config.rs)]
    end

    subgraph Process Control
        PState -->|Spawns| Child[Child Process (Command)]
        Child -->|StdOut/Err| Logs[Log Buffer]
        PState -->|Reads| Logs
        PState -->|Polls| Child
    end

    subgraph File System
        App -->|Scans| Projects[projects/ Dir]
        PState -->|Writes| JSON[project.json]
        Child -->|Reads| Data[data/ Dir]
        Child -->|Writes| Models[models/ Dir]
    end

    %% Risks
    style Child fill:#f9f,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    note[Risk: Process Hang freezing UI] -.-> Child
    note2[Risk: Large Log Buffer memory] -.-> Logs
```

## Risk Assessment
1.  **Process Blocking**: `try_wait()` is non-blocking, but if we read large chunks of logs synchronously in the UI loop, it might stutter.
    *   *Mitigation*: Use `std::sync::mpsc` or `crossbeam` to stream logs from a background thread to UI thread.
2.  **State Desync**: If the user edits `project.json` externally, the GUI might not reflect it until reload.
3.  **Config Compatibility**: Adding new fields to `ProjectConfig` necessitates backward compatibility for existing projects.
