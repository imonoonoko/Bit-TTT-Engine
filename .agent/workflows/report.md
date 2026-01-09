---
description: Generate a timestamped session summary report in docs/ (Japanese)
---

# Session Reporting Workflow

This workflow generates a summary of the current session's achievements and saves it as a timestamped markdown file in the `docs/` directory.

## Step 1: Determine Date and Time
Get the current date and time in the format `YYYY-MM-DD_HH-mm`.
(The agent knows the current time from metadata).

## Step 2: Compile Summary Content
Review the following sources to compile a comprehensive summary:
1.  `task.md` (Completed tasks)
2.  `discussion_log.md` (Key decisions and user approvals)
3.  Recent file modifications ( Implementation details)

Construct the report with the following sections in **Japanese**:
-   **Header**: `# Session Summary: [YYYY-MM-DD HH:mm]`
-   **Objectives (目的)**: What was the main goal of this session?
-   **Key Achievements (成果)**: Bullet points of completed features, refactors, or fixes.
-   **Artifacts Created/Modified (作成・変更ファイル)**: List of key files.
-   **Decisions & Insights (決定事項・学び)**: Important conceptual decisions or learnings.
-   **Next Steps (次への一歩)**: What is left to do?

## Step 3: Create Report File
Create a new file at `docs/Session_Summary/SESSION_SUMMARY_[YYYY-MM-DD_HH-mm].md` with the compiled content.

## Step 4: Notify User
Inform the user that the report has been created and provide the path.
