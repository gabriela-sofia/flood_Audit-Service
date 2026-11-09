param(
    [string]$PythonBase = "python",
    [switch]$AproveitarPacotesInstalados,
    [switch]$InstalarAuditoria
)
$ErrorActionPreference = "Stop"
$Ambiente = Join-Path $PSScriptRoot ".venv"
$Interpretador = Join-Path $Ambiente "Scripts\python.exe"
if ($AproveitarPacotesInstalados) {
    & $PythonBase -m venv --without-pip --system-site-packages $Ambiente
    if ($LASTEXITCODE -ne 0) { throw "Não foi possível preparar o ambiente de conferência." }
} elseif (!(Test-Path -LiteralPath $Interpretador)) {
        & $PythonBase -m venv $Ambiente
        if ($LASTEXITCODE -ne 0) { throw "Não foi possível criar o ambiente." }
}
if ($InstalarAuditoria) {
    & $Interpretador -m pip install -r (Join-Path $PSScriptRoot "requirements-auditoria.txt")
    if ($LASTEXITCODE -ne 0) { throw "Não foi possível instalar as dependências." }
}
& $Interpretador -c "import sys; print(sys.executable); print(sys.version)"
