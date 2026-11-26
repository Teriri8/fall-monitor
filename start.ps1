param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"
$PythonExe = "C:\Users\蔡华升\AppData\Local\Programs\Python\Python311\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python 3.11 not found at $PythonExe"
    exit 1
}

Push-Location $PSScriptRoot
try {
    & $PythonExe -m pc.run_all @ForwardArgs
}
finally {
    Pop-Location
}
