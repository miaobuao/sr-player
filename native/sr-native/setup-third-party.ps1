# setup-third-party.ps1 — rebuild everything sr-native links against or loads.
#
# Run this once after cloning, and again whenever pins.json changes:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File native\sr-native\setup-third-party.ps1
#
# (`powershell`, not `pwsh`: the reference machine has Windows PowerShell 5.1 and
# no PowerShell 7, so the documented invocation has to be the one that exists.)
#
# It produces three things, none of which is committed:
#
#     third_party/ncnn/            ncnn, from the pinned release, verified by hash
#     third_party/vulkan/          vulkan-1.lib, generated from the system loader
#     models/                      the pinned weights, converted for our loader
#
# Every hash comes from pins.json. Nothing here floats.

[CmdletBinding()]
param(
    # Not defaulted from $PSScriptRoot: a param() default is evaluated in the
    # caller's scope, where $PSScriptRoot is empty.
    [string]$RepoRoot,
    [switch]$SkipNcnn,
    [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }

# ---------------------------------------------------------------------------
# Environment notes, each of which cost real time to discover on the reference
# machine. They are written down here so the next person does not repeat them.
#
#  * PowerShell 5.1 negotiates TLS 1.0 by default and a modern host will close
#    the connection. Set TLS 1.2 explicitly.
#  * `github.com` itself may be unreachable while `api.github.com`,
#    `codeload.github.com` and `objects.githubusercontent.com` are fine. Getting
#    a release asset therefore means asking the API for the asset and following
#    the redirect it returns, not guessing the /releases/download/ URL.
#  * `curl.exe` on Windows uses schannel and failed the TLS handshake here even
#    with --ssl-no-revoke, through a proxy or without one. Do not use it.
#  * `Invoke-WebRequest` silently truncates large downloads. Anything above a few
#    megabytes goes through HttpClient with ResponseHeadersRead.
#  * If the machine is behind a proxy, git picks it up from HTTP_PROXY but
#    .NET does not necessarily; pass -Proxy if the downloads fail.
# ---------------------------------------------------------------------------

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.Net.Http

$UserAgent = @{ 'User-Agent' = 'sr-player-setup' }
$Octet = @{ 'User-Agent' = 'sr-player-setup'; 'Accept' = 'application/octet-stream' }

function Get-Pins {
    $path = Join-Path $PSScriptRoot 'pins.json'
    if (-not (Test-Path $path)) { throw "pins.json is missing next to this script" }
    return Get-Content $path -Raw | ConvertFrom-Json
}

# pins.json records an ordinary repository URL, because that is what a reader wants
# to see. The API endpoints need `owner/repo`.
function Get-RepoSlug($url) {
    if ($url -match 'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$') { return "$($Matches[1])/$($Matches[2])" }
    throw "cannot read owner/repo out of '$url'"
}

function Assert-Hash($path, $expected, $what) {
    if (-not $expected -or $expected -eq 'PENDING') {
        Write-Warning "  no recorded sha256 for $what; skipping the check"
        return
    }
    $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected.ToLower()) {
        throw "$what failed its hash check`n  expected $expected`n  actual   $actual"
    }
    Write-Host "  sha256 ok  $what"
}

# Streams to disk. Do not replace this with Invoke-WebRequest: it truncates.
function Save-Url($url, $out, $accept) {
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.AllowAutoRedirect = $true
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromMinutes(60)
    $client.DefaultRequestHeaders.Add('User-Agent', 'sr-player-setup')
    if ($accept) { $client.DefaultRequestHeaders.Add('Accept', $accept) }
    try {
        $response = $client.GetAsync($url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        $response.EnsureSuccessStatusCode() | Out-Null
        $stream = [IO.File]::Create($out)
        try { $response.Content.CopyToAsync($stream).Wait() } finally { $stream.Close() }
    } finally {
        $client.Dispose()
    }
}

# A release asset, fetched the only way that works when github.com is blocked.
function Save-ReleaseAsset($repo, $tag, $name, $out) {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/tags/$tag" `
        -Headers $UserAgent -TimeoutSec 60
    $asset = $release.assets | Where-Object { $_.name -eq $name } | Select-Object -First 1
    if (-not $asset) { throw "release $tag of $repo has no asset named $name" }
    Write-Host ("  {0}  ({1:N1} MB)" -f $asset.name, ($asset.size / 1MB))
    Save-Url $asset.url $out 'application/octet-stream'
}

# A single file out of a repository, through the blob API — the same workaround.
function Save-RepoFile($repo, $path, $out) {
    $meta = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/contents/$path" `
        -Headers $UserAgent -TimeoutSec 120
    $blob = Invoke-RestMethod -Uri $meta.git_url -Headers $UserAgent -TimeoutSec 600
    [IO.File]::WriteAllBytes($out, [Convert]::FromBase64String($blob.content))
}

# Runs a command inside the MSVC environment. dumpbin and lib are not on PATH.
function Invoke-Msvc($batchBody) {
    $vsRoot = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
        -products * -latest -property installationPath
    if (-not $vsRoot) { throw "Visual Studio not found; vswhere returned nothing" }
    $vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat is missing at $vcvars" }

    $bat = Join-Path $env:TEMP ("sr-native-msvc-{0}.bat" -f ([guid]::NewGuid().ToString('N')))
    @(
        '@echo off',
        "call `"$vcvars`" >nul 2>&1",
        'if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )',
        $batchBody
    ) | Set-Content -Path $bat -Encoding ASCII
    try {
        & cmd.exe /c $bat
        if ($LASTEXITCODE -ne 0) { throw "MSVC step failed with exit code $LASTEXITCODE" }
    } finally {
        Remove-Item $bat -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------

$pins = Get-Pins
$thirdParty = Join-Path $PSScriptRoot 'third_party'
$scratch = Join-Path $RepoRoot '.sr-setup'
New-Item -ItemType Directory -Force -Path $thirdParty, $scratch | Out-Null

# --- ncnn ------------------------------------------------------------------
if (-not $SkipNcnn) {
    Write-Host "`n== ncnn $($pins.ncnn.tag) =="
    $zip = Join-Path $scratch $pins.ncnn.artifact
    if (-not (Test-Path $zip)) {
        Save-ReleaseAsset (Get-RepoSlug $pins.ncnn.repository) $pins.ncnn.tag $pins.ncnn.artifact $zip
    }
    Assert-Hash $zip $pins.ncnn.artifact_sha256 $pins.ncnn.artifact

    $extract = Join-Path $scratch 'ncnn-extract'
    Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -Path $zip -DestinationPath $extract -Force
    $inner = Get-ChildItem $extract -Directory | Select-Object -First 1
    if (-not $inner) { throw "the ncnn archive did not contain a directory" }

    $target = Join-Path $thirdParty 'ncnn'
    Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path (Join-Path $target 'lib') | Out-Null
    Copy-Item (Join-Path $inner.FullName 'x64\include') (Join-Path $target 'include') -Recurse -Force
    Copy-Item (Join-Path $inner.FullName 'x64\lib\*.lib') (Join-Path $target 'lib') -Force
    Copy-Item (Join-Path $inner.FullName 'x64\lib\cmake') (Join-Path $target 'lib\cmake') -Recurse -Force
    # ncnnoptimize is used below to convert the model weights; keep it with them.
    Copy-Item (Join-Path $inner.FullName 'x64\bin\ncnnoptimize.exe') (Join-Path $target 'lib') -Force
    Write-Host "  staged third_party/ncnn"

    # --- vulkan-1.lib ------------------------------------------------------
    Write-Host "`n== vulkan-1.lib =="
    $vulkanDir = Join-Path $thirdParty 'vulkan'
    New-Item -ItemType Directory -Force -Path $vulkanDir | Out-Null
    $def = Join-Path $scratch 'vulkan-1.def'
    $exportsText = Join-Path $scratch 'vulkan-exports.txt'

    Invoke-Msvc @"
dumpbin /nologo /exports "%SystemRoot%\System32\vulkan-1.dll" > "$exportsText" 2>&1
if errorlevel 1 ( echo DUMPBIN_FAILED & exit /b 2 )
"@

    $names = Get-Content $exportsText | ForEach-Object {
        if ($_ -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$') { $Matches[1] }
    } | Sort-Object -Unique
    if ($names.Count -lt 100) { throw "only $($names.Count) exports parsed out of vulkan-1.dll" }

    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine('LIBRARY vulkan-1.dll')
    [void]$sb.AppendLine('EXPORTS')
    foreach ($n in $names) { [void]$sb.AppendLine("    $n") }
    [IO.File]::WriteAllText($def, $sb.ToString())

    Invoke-Msvc @"
lib /nologo /def:"$def" /machine:x64 /out:"$vulkanDir\vulkan-1.lib"
if errorlevel 1 ( echo LIB_FAILED & exit /b 1 )
"@
    Write-Host "  generated vulkan-1.lib from $($names.Count) exports"
}

# --- models ----------------------------------------------------------------
if (-not $SkipModels) {
    $models = Join-Path $RepoRoot 'models'

    # RIFE 4.25: published as fp16, converted to fp32 with ncnn's own reader.
    Write-Host "`n== RIFE 4.25 =="
    $rife = $pins.models.'rife-4.25'
    $rifeDir = Join-Path $models 'rife-4.25'
    New-Item -ItemType Directory -Force -Path $rifeDir | Out-Null
    $publishedParam = Join-Path $scratch 'rife-4.25.published.param'
    $publishedBin = Join-Path $scratch 'rife-4.25.published.bin'

    Save-RepoFile (Get-RepoSlug $rife.repository) "$($rife.repo_path)/flownet.param" $publishedParam
    Save-RepoFile (Get-RepoSlug $rife.repository) "$($rife.repo_path)/flownet.bin" $publishedBin
    Assert-Hash $publishedBin $rife.published.'flownet.bin'.sha256 'rife-4.25 published flownet.bin'
    Copy-Item $publishedBin (Join-Path $rifeDir 'flownet.bin.fp16') -Force

    $ncnnoptimize = Join-Path $thirdParty 'ncnn\lib\ncnnoptimize.exe'
    if (-not (Test-Path $ncnnoptimize)) {
        throw "ncnnoptimize.exe is missing; run without -SkipNcnn so ncnn is staged first"
    }
    # ncnnoptimize announces the custom layer on stderr, and PowerShell 5.1 turns a
    # native command's stderr into a terminating error while ErrorActionPreference is
    # Stop. Scope the preference down for this one call.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $ncnnoptimize $publishedParam $publishedBin (Join-Path $rifeDir 'flownet.param') `
            (Join-Path $rifeDir 'flownet.bin') 0 2>&1 | Out-Null
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    # ncnnoptimize prints progress to stderr and exits 0 even with a custom layer it
    # cannot run shape inference for, so the exit code is not the check — the size is.
    $converted = Join-Path $rifeDir 'flownet.bin'
    if (-not (Test-Path $converted) -or (Get-Item $converted).Length -lt 20MB) {
        throw "ncnnoptimize did not produce a plausible fp32 flownet.bin"
    }
    Assert-Hash $converted $rife.installed.'flownet.bin'.sha256 'rife-4.25 installed flownet.bin'
    Write-Host "  staged models/rife-4.25"

    # Real-ESRGAN x4plus, out of its release bundle.
    Write-Host "`n== Real-ESRGAN x4plus =="
    $esrgan = $pins.models.'realesrgan-x4plus'
    $bundle = Join-Path $scratch $esrgan.artifact
    if (-not (Test-Path $bundle)) {
        Save-ReleaseAsset (Get-RepoSlug $esrgan.repository) $esrgan.release $esrgan.artifact $bundle
    }
    $esrganExtract = Join-Path $scratch 'esrgan-extract'
    Remove-Item $esrganExtract -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -Path $bundle -DestinationPath $esrganExtract -Force
    $esrganDir = Join-Path $models 'realesrgan-x4plus'
    New-Item -ItemType Directory -Force -Path $esrganDir | Out-Null
    $src = Get-ChildItem $esrganExtract -Recurse -Filter 'realesrgan-x4plus.param' | Select-Object -First 1
    if (-not $src) { throw "the Real-ESRGAN bundle has no realesrgan-x4plus.param" }
    Copy-Item $src.FullName (Join-Path $esrganDir 'model.param') -Force
    Copy-Item ($src.FullName -replace '\.param$', '.bin') (Join-Path $esrganDir 'model.bin') -Force
    Assert-Hash (Join-Path $esrganDir 'model.bin') $esrgan.installed.'model.bin'.sha256 'realesrgan model.bin'
    Write-Host "  staged models/realesrgan-x4plus"
}

Write-Host "`nDone. Build with:"
Write-Host "  cmake -S native/sr-native -B native/sr-native/build -G Ninja -DCMAKE_BUILD_TYPE=Release"
Write-Host "  cmake --build native/sr-native/build"
Write-Host "(from a shell where vcvars64.bat has been called)"
