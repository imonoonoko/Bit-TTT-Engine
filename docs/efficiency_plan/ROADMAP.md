# Roadmap: Context Efficiency Optimization

## Phase 1: Context Foundation (Immediate)
Define the core "Why, What, How" structure.

*   [ ] **Create `docs/context/` Directory**: Central storage for agent context.
*   [ ] **Create `docs/context/mission.md` (WHY)**: Define project goals (Bit-TTT, Efficient LLM Training).
*   [ ] **Create `docs/context/map.md` (WHAT)**: High-level architecture map pointing to key components (`crates/`, `docs/`, `tools/`).
*   [ ] **Create `docs/context/protocols.md` (HOW)**: Establish standard operating procedures (Analysis -> Plan -> Execute).

## Phase 2: Root Identity Refactoring
Establish a lightweight entry point.

*   [ ] **Create `ACTION_GUIDE.md` (Root)**:
    *   Max 60 lines.
    *   Imports/Links to `mission.md`, `map.md`, `protocols.md`.
    *   Acts as the primary "Map" for the agent.

## Phase 3: Rule Refinement
Transition from monolithic instructions to progressive disclosure.

*   [ ] **Refactor `v5.md`**:
    *   Split detailed workflows into `docs/context/workflows/`.
    *   Keep `v5.md` (or rename to `.agent/rules/coding_core.md`) as a lightweight constraint file.
*   [ ] **Cleanup**: Remove redundant legacy context files.

## Phase 4: Verification
Ensure the agent utilizes the new context.

*   [ ] **Verification Task**: Run a sample "Lite" task using only the new context structure to verify efficiency.
