# Builds the end-to-end test fixture.
#
# The fixture is deliberately DVD-shaped and deliberately awkward:
#   * 720x480 MPEG-2 at 29.97 with hard cuts (exercises scene detection)
#   * a 5.1 AC-3 track whose dialogue sits in the centre channel, plus a second
#     stereo track (the enhanced track must not disturb the originals)
#   * a subtitle track, chapters and a font attachment, because "remux without
#     losing anything" is the part nobody notices until it breaks
#
# Usage:  powershell -File testdata\make_fixture.ps1

$ErrorActionPreference = 'Stop'
$out = Join-Path (Split-Path -Parent $PSScriptRoot) 'testdata'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if (-not $ffmpeg) { throw 'ffmpeg not found on PATH' }

$base = Join-Path $out 'sample-base.mkv'
$video = Join-Path $out 'sample-dvd.mkv'
$srt = Join-Path $out 'sample.srt'
$meta = Join-Path $out 'chapters.txt'
$font = Join-Path $out 'sample-font.ttf'

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

# --- sidecar files ---------------------------------------------------------
Write-Utf8NoBom $srt @"
1
00:00:00,500 --> 00:00:02,000
This is a hard cut test.

2
00:00:03,000 --> 00:00:05,500
Second shot, second subtitle line.
"@

Write-Utf8NoBom $meta @"
;FFMETADATA1
title=SR Test Fixture

[CHAPTER]
TIMEBASE=1/1000
START=0
END=4000
title=Opening

[CHAPTER]
TIMEBASE=1/1000
START=4000
END=8000
title=Second Shot
"@

$systemFont = Join-Path $env:WINDIR 'Fonts\arial.ttf'
if (Test-Path $systemFont) {
    Copy-Item $systemFont $font -Force
} else {
    [byte[]]$bytes = 0..2047 | ForEach-Object { $_ % 251 }
    [System.IO.File]::WriteAllBytes($font, $bytes)
}

# --- step 1: picture + two audio tracks + subtitle -------------------------
# All inputs first, then the filter graph, then output options.
$filter = '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];[3:a]volume=0.25,pan=5.1|FL=c0|FR=c0|FC=c0|LFE=c0|BL=c0|BR=c0[dx];[4:a]volume=0.9[fx];[dx][fx]amix=inputs=2:duration=longest:normalize=0[a51];[5:a]volume=0.5[a2]'

& $ffmpeg -y -hide_banner -loglevel error `
    -f lavfi -i "testsrc2=size=720x480:rate=30000/1001:duration=3" `
    -f lavfi -i "smptebars=size=720x480:rate=30000/1001:duration=2.5" `
    -f lavfi -i "testsrc=size=720x480:rate=30000/1001:duration=2.5" `
    -f lavfi -i "sine=frequency=1200:sample_rate=48000:duration=8" `
    -f lavfi -i "sine=frequency=70:sample_rate=48000:duration=8" `
    -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=8" `
    -i $srt `
    -filter_complex $filter `
    -map "[v]" -map "[a51]" -map "[a2]" -map 6:s `
    -c:v mpeg2video -q:v 4 -pix_fmt yuv420p `
    -c:a:0 ac3 -b:a:0 384k -ac:a:0 6 `
    -c:a:1 ac3 -b:a:1 192k -ac:a:1 2 `
    -c:s srt `
    -metadata:s:a:0 language=eng -metadata:s:a:0 title="Main 5.1" `
    -metadata:s:a:1 language=jpn -metadata:s:a:1 title="Stereo dub" `
    -metadata:s:s:0 language=chi `
    -map_metadata -1 `
    -f matroska $base

# --- step 2: chapters + a font attachment ---------------------------------
& $ffmpeg -y -hide_banner -loglevel error `
    -i $base -i $meta -attach $font `
    -map 0 -map_metadata 1 -map_chapters 1 `
    -c copy -c:t copy `
    -metadata:s:t:0 mimetype=application/x-truetype-font `
    -metadata:s:t:0 filename=sample-font.ttf `
    -f matroska $video

Remove-Item -Force $base

Write-Host 'fixture written:'
Get-ChildItem $out | Select-Object Name, Length | Format-Table
