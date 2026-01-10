# 🛠️ Bit-TTT Tools

このフォルダには、Bit-TTT プロジェクトの各種操作を行うための簡易スクリプトが含まれています。
PowerShell から実行してください。

## 🚀 主要ツール

### 1. データセット作成: `BitLlama-Preprocess.ps1`
テキストファイルを学習用のバイナリ形式 (`.u32`) に変換します。
```powershell
.\BitLlama-Preprocess.ps1 --input corpus.txt --output data.u32 --tokenizer tokenizer.json
```

### 2. 学習 (Training): `BitLlama-Train.ps1`
モデルの学習を実行します。
```powershell
.\BitLlama-Train.ps1 --data data.u32 --steps 1000 --min-lr 0.0001
```

### 3. 推論・チャット (Chat): `BitLlama-Chat.ps1`
学習済みモデルを使って対話を行います。
```powershell
.\BitLlama-Chat.ps1 --model bit_llama.bitt
.\BitLlama-Chat.ps1 --model .  # カレントディレクトリのモデルを使用
```

### 4. モデル変換 (Export): `BitLlama-Export.ps1`
学習結果 (`.safetensors`, `config.json`) を配布用の単一ファイル (`.bitt`) に変換します。
```powershell
.\BitLlama-Export.ps1 --output distributed_model.bitt
```

### 5. GUI 起動: `BitLlama-GUI.ps1`
学習やチャットを GUI で行うランチャーを起動します。
```powershell
.\BitLlama-GUI.ps1
```

### 6. Wiki40b-ja 準備: `BitLlama-PrepareWiki.ps1`
Wiki40b 日本語データセットをダウンロードし、学習用データを作成します。
```powershell
.\BitLlama-PrepareWiki.ps1 --limit 1000 # テスト実行
```

## ⚠️ 注意事項
- 初回実行時はコンパイルが行われるため、起動に時間がかかります。
- 引数がわからない場合は、各スクリプトに `--help` を付けて実行することで詳細を確認できます（例: `.\BitLlama-Train.ps1 --help`）。
