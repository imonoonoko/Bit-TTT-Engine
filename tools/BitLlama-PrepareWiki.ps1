Write-Host "=== Bit-Llama Wiki40b-ja Preparation Tool ===" -ForegroundColor Cyan
Write-Host "Usage: .\BitLlama-PrepareWiki.ps1 --output_dir <dir> --limit <N>"
Write-Host "Example: .\BitLlama-PrepareWiki.ps1 --limit 1000 (Dry Run)"

# Check dependencies
# python -c "import datasets" 2>$null
# if ($LASTEXITCODE -ne 0) {
#    Write-Log "Installing required python packages: datasets tokenizers"
#    pip install datasets tokenizers
# }

python tools/prepare_wiki40b.py $args
