param([string]$Output)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Output) { $Output = Join-Path $Root 'WushuBridge-Trainer.exe' }
$Compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $Compiler)) { throw '.NET Framework 4.x C# compiler not found.' }
$Source = Join-Path $PSScriptRoot 'launcher\Launcher.cs'
$Manifest = Join-Path $PSScriptRoot 'launcher\app.manifest'
& $Compiler /nologo /target:winexe /platform:anycpu /optimize+ /codepage:65001 "/out:$Output" "/win32manifest:$Manifest" /r:System.Windows.Forms.dll /r:System.Drawing.dll /r:System.Core.dll /r:System.Xml.Linq.dll /r:System.Web.Extensions.dll $Source
if ($LASTEXITCODE -ne 0) { throw "Launcher build failed: $LASTEXITCODE" }
Get-Item -LiteralPath $Output | Select-Object FullName, Length
