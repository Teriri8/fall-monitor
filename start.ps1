param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"

function Resolve-AppPython {
    $candidates = @(
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")
    )
    foreach ($p in $candidates) {
        if ($p -and (Test-Path $p)) {
            & $p -c "import cv2, ultralytics, fastapi" 2>$null
            if ($LASTEXITCODE -eq 0) { return $p }
        }
    }
    foreach ($cmd in @("py", "python")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            if ($cmd -eq "py") {
                $pyPath = & py -3.11 -c "import sys, cv2, ultralytics, fastapi; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $pyPath) { return $pyPath.Trim() }
                $pyPath = & py -3 -c "import sys, cv2, ultralytics, fastapi; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $pyPath) { return $pyPath.Trim() }
            } else {
                & python -c "import cv2, ultralytics, fastapi" 2>$null
                if ($LASTEXITCODE -eq 0) { return $found.Source }
            }
        }
    }
    return $null
}

$PythonExe = Resolve-AppPython
if (-not $PythonExe) {
    Write-Error "No Python with opencv-python, ultralytics, and fastapi found. Run: pip install -r requirements.txt"
    exit 1
}

Push-Location $PSScriptRoot
try {
    & $PythonExe -m pc.run_all @ForwardArgs
}
finally {
    Pop-Location
}
