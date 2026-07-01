$ErrorActionPreference = "Stop"

$pythonCmd = $null
foreach ($candidate in @("py", "python")) {
    try {
        $versionOutput = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0 -and ($versionOutput -join " ") -match "Python 3") {
            $pythonCmd = $candidate
            break
        }
    } catch {
    }
}

if (-not $pythonCmd) {
    Write-Host "Python 3.10+ was not found. Install Python from https://www.python.org/downloads/ and run this script again."
    exit 1
}

if (-not (Test-Path ".venv")) {
    & $pythonCmd -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -U pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" main.py
