# Dependency Map: Phase C

## 1. Logging Architecture
```mermaid
graph TD
    Sources[Modules (Train/Gui/etc)] -->|tracing::info!| Subscriber[Tracing Subscriber]
    
    Subscriber -->|Layer 1| Stdout[Console Output]
    Subscriber -->|Layer 2| File[Log File (optional)]
    Subscriber -->|Layer 3| Channel[MPSC Channel]
    
    Channel -->|Receiver| GUI[GUI Log Window]
```

## 2. Data Loading Unification
```mermaid
graph TD
    Train[train.rs] --> SharedLoader[src/loader.rs]
    Eval[evaluate.rs] --> SharedLoader
    
    SharedLoader -->|Read| Dataset[.u32 / .txt]
```

## 3. Configuration Flow
```mermaid
graph TD
    CLI[Args] --> ConfigObj[BitLlamaConfig]
    GUI[UI State] --> ConfigObj
    
    ConfigObj -->|Serialize| JSON[config.json]
    ConfigObj -->|Deserialize| Engine[Core Engine]
```
