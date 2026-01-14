# Dependency Map: Phase C (VRAM Monitor)

## 1. Dependency Visualization (Mermaid)

```mermaid
graph TD
    A[Bit-Llama GUI] --> B(GUI Loop);
    B --> C{Feature: cuda?};
    C -- Yes --> D[Monitor Module];
    C -- No --> E[Dummy Monitor / None];
    D --> F[NVML Wrapper];
    F --> G[NVIDIA Driver / GPU];
    
    subgraph "State Management"
        S[SharedState] --> H[VRAM Metrics];
        D -. Update .-> H;
    end
    
    subgraph "UI Components"
        I[Dashboard] -. Read .-> H;
        J[TrainingGraph] -. Read .-> H;
    end
```

## 2. Risk Assessment (Dependencies)

| Dependency | Impact | Risk | Mitigation |
| :--- | :--- | :--- | :--- |
| **`nvml-wrapper`** | Connects to NVIDIA Management Library | High (Linkage Error) | Wrap in `cfg(feature = "cuda")`. Graceful degradation if DLL missing. |
| **Running on CPU** | No GPU, No NVML | Low (Panic?) | Ensure runtime check `is_available()` before init. |
| **Build Time** | Linking external libs | Medium | Windows Path issues. Instructions in README. |
| **State Lock** | `SharedState` mutex contention | Low | Updates occur on separate thread or infrequent check (1Hz). |

## 3. Critical Path
1.  Add `nvml-wrapper` to `Cargo.toml`.
2.  Implement `Monitor` struct with `cfg` guards.
3.  Wire up `SharedState`.
4.  UI Display.
