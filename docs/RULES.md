# 🚨 Project Rules (Highest Priority)

## 💻 MSVC Environment Wrapper
**Trigger**: Execution of build commands (`cargo build`, `nvcc`, `cl`, etc.)
**Status**: **MANDATORY**

Due to the environment lacking `cl.exe` in the PATH, the Agent **MUST** wrap all compilation commands with the Visual Studio environment script.

### 🛠 Command Wrapper Pattern
Use `cmd /c` with `call` to execute the setup script before the actual command.

```batch
cmd /c "call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && <Original Command>"
```

### Examples
- **Cargo**:
  `cmd /c "call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo build"`
- **NVCC**:
  `cmd /c "call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && nvcc ..."`
