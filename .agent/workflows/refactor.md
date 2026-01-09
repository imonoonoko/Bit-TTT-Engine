---
description: 安全かつ構造的なリファクタリングプロセス
---

# /refactor - Refactoring Workflow

既存の機能を壊さずに、構造やコード品質を改善するプロセス。

## Step 1: 現状把握 (Analysis)
まず、対象のコードと依存関係を深く理解する。
```bash
view_file_outline <target_file>
grep_search <target_class/function> src/
```

## Step 2: 計画策定 (Planning)
1. **Safety Check**: 
   - 変更の影響範囲を特定
   - 既存テストの有無を確認

2. **Refactoring Plan**:
   - `docs/REFACTOR_PLAN.md`（大規模な場合）または
   - `docs/QUICK_PLAN.md`（小規模な場合）を作成
   - 変更前後の構造を明記

## Step 3: 安全な実装 (Safe Implementation)
1. **Incremental Changes**:
   - 一度にすべてを変えない
   - 小さな関数/メソッド単位で変更

2. **Verification Loop**:
   - 変更 → `python -m py_compile <file>`
   - 変更 → `python run_tests.py`

## Step 4: 結果確認 (Verification)
1. **Regression Test**: 
   - 既存機能が壊れていないか確認
   - `python src/tools/pre_demon.py` で危険パターンチェック

2. **Cleanup**:
   - 不要になった古い関数/変数を削除
   - `docs/CODE_ATLAS.md` を更新

---
// turbo
## Auto-Check
```bash
python src/tools/pre_demon.py
python run_tests.py
```
