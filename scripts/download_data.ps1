# Download GTZAN from Kaggle into data/raw. Skips if already present.
# Credentials, if Kaggle asks for them: %USERPROFILE%\.kaggle\kaggle.json
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path data/raw/Data/genres_original) { Write-Output "GTZAN already in data/raw, skipping"; exit 0 }
& .venv\Scripts\kaggle.exe datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p data/raw --unzip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
