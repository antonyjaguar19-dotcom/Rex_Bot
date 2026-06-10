# start_dashboard_ngrok.ps1
# -------------------------------------------------------------------
# Exposes the Rex VFX dashboard (http://127.0.0.1:7860) on a FIXED free
# public HTTPS URL using ngrok's free static domain. Same URL every run.
#
# ONE-TIME SETUP (browser, ~2 min):
#   1. Sign up free:           https://dashboard.ngrok.com/signup
#   2. Copy your authtoken:    https://dashboard.ngrok.com/get-started/your-authtoken
#   3. Claim your free domain: https://dashboard.ngrok.com/domains  (e.g. rexbot-jeffy.ngrok-free.app)
#   4. Put both in 05_Config\secrets.env:
#        NGROK_AUTHTOKEN=xxxxxxxxxxxxxxxxxxxx
#        NGROK_DOMAIN=rexbot-jeffy.ngrok-free.app
#        CLAW_DASHBOARD_URL=https://rexbot-jeffy.ngrok-free.app
#   5. Set a dashboard password too (public URL):
#        DASHBOARD_PASSWORD=pick-a-strong-pass
#   6. Restart the bot, then run this script.
#
# Run AFTER the bot is up. Keep this window open (Ctrl+C closes the tunnel).
# -------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$tools   = "E:\Rexjaw_VFX\00_Tools"
$ngrok   = Join-Path $tools "ngrok.exe"
$secrets = "E:\Rexjaw_VFX\05_Config\secrets.env"

function Get-EnvKey([string]$key) {
    if (-not (Test-Path $secrets)) { return $null }
    $line = Select-String -Path $secrets -Pattern ("^\s*{0}\s*=\s*(.+)$" -f [regex]::Escape($key)) |
            Select-Object -First 1
    if ($line) { return $line.Matches[0].Groups[1].Value.Trim().Trim('"') }
    return $null
}

# 1. Get ngrok.exe (download + unzip once into 00_Tools).
if (-not (Test-Path $ngrok)) {
    Write-Host "ngrok.exe not found - downloading to $tools ..."
    $zip = Join-Path $env:TEMP "ngrok-v3-stable-windows-amd64.zip"
    Invoke-WebRequest -Uri "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tools -Force
    Remove-Item $zip -ErrorAction SilentlyContinue
    Write-Host "Downloaded ngrok.exe"
}

# 2. Read config from secrets.env.
$token  = Get-EnvKey "NGROK_AUTHTOKEN"
$domain = Get-EnvKey "NGROK_DOMAIN"

if (-not $token) {
    Write-Error "NGROK_AUTHTOKEN missing in secrets.env. See setup steps at top of this script."
    exit 1
}
if (-not $domain) {
    Write-Error "NGROK_DOMAIN missing in secrets.env (your claimed *.ngrok-free.app). See setup steps."
    exit 1
}

# 3. Password gate check (public URL).
if (-not (Get-EnvKey "DASHBOARD_PASSWORD")) {
    Write-Warning "DASHBOARD_PASSWORD is NOT set - public URL would have NO login gate."
    Write-Warning "Add  DASHBOARD_PASSWORD=...  to secrets.env and RESTART the bot first."
    $ans = Read-Host "Continue WITHOUT a password gate anyway? (type YES to proceed)"
    if ($ans -ne "YES") { Write-Host "Aborted."; exit 1 }
}

# 4. Register authtoken (idempotent) + run on the fixed domain.
& $ngrok config add-authtoken $token | Out-Null
Write-Host ""
Write-Host "Starting ngrok -> https://$domain  ->  http://127.0.0.1:7860"
Write-Host "Open that URL on your phone (same every run). Ctrl+C closes the tunnel."
Write-Host ""
& $ngrok http ("--url=" + $domain) 7860
