param(
    [string]$PartDirectory = $PSScriptRoot
)
$ErrorActionPreference = 'Stop'

$root = [System.IO.Path]::GetFullPath($PartDirectory)
$manifestPath = Join-Path $root 'RELEASE_PARTS.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing RELEASE_PARTS.json in $root"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $manifest.archive_name -or -not $manifest.archive_sha256 -or @($manifest.parts).Count -ne 3) {
    throw 'Invalid release parts manifest.'
}
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $root $manifest.archive_name))
$separator = [System.IO.Path]::DirectorySeparatorChar
$prefix = $root.TrimEnd($separator) + $separator
if (-not $outputPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Archive output must stay inside the part directory.'
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Output already exists; refusing to overwrite: $outputPath"
}

foreach ($entry in $manifest.parts) {
    $path = [System.IO.Path]::GetFullPath((Join-Path $root $entry.name))
    if (-not $path.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Part path escapes the part directory: $($entry.name)"
    }
    $file = Get-Item -LiteralPath $path -ErrorAction Stop
    if ($file.Length -ne [long]$entry.bytes) { throw "Part size mismatch: $($entry.name)" }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne [string]$entry.sha256) { throw "Part SHA-256 mismatch: $($entry.name)" }
}

$out = [System.IO.File]::Create($outputPath)
try {
    foreach ($entry in $manifest.parts) {
        $path = Join-Path $root $entry.name
        $inputStream = [System.IO.File]::OpenRead($path)
        try { $inputStream.CopyTo($out) }
        finally { $inputStream.Dispose() }
    }
}
finally { $out.Dispose() }

$joinedHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($joinedHash -ne [string]$manifest.archive_sha256) {
    throw "Joined ZIP hash mismatch: $joinedHash"
}
Write-Host "Ready: $outputPath"
Write-Host "SHA-256: $joinedHash"
