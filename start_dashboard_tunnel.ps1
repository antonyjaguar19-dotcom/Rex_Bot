# start_dashboard_tunnel.ps1
# -------------------------------------------------------------------
# Exposes the Rex VFX dashboard (http://127.0.0.1:7860) on a FREE public
# HTTPS URL using a Cloudflare Quick Tunnel. No account, no port-forwarding.
#
# Run this AFTER the bot is up (claw_bot.py launches the dashboard).
# The public URL (https://something.trycloudflare.com) prints below — open
# it on your phone over mobile data.
#
# SECURITY: the public URL is reachable by anyone. The dashboard login gate
# is your protection — set DASHBOARD_PASSWORD in 05_Config/secrets.env first.
# -------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$tools  = "E:\Rexjaw_VFX\00_Tools"
$cf     = Join-Path $tools "cloudflared.exe"
$secrets = "E:\Rexjaw_VFX\05_Config\secrets.env"

# 1. Get cloudflared.exe (download once into 00_Tools, stays contained).
if (-not (Test-Path $cf)) {
    Write-Host "cloudflared.exe not found - downloading to $cf ..."
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $url -OutFile $cf
    Write-Host "Downloaded cloudflared.exe"
}

# 2. Refuse-to-expose-naked check: warn loudly if no dashboard password.
$hasPass = (Test-Path $secrets) -and `
           (Select-String -Path $secrets -Pattern '^\s*DASHBOARD_PASSWORD\s*=\s*\S' -Quiet)
if (-not $hasPass) {
    Write-Warning "DASHBOARD_PASSWORD is NOT set in secrets.env."
    Write-Warning "The public URL would have NO login gate (anyone could drive your GPU)."
    Write-Warning "Add this line to 05_Config\secrets.env, then RESTART the bot:"
    Write-Warning "    DASHBOARD_PASSWORD=pick-a-strong-pass"
    $ans = Read-Host "Continue WITHOUT a password gate anyway? (type YES to proceed)"
    if ($ans -ne "YES") { Write-Host "Aborted."; exit 1 }
}

# 3. Open the tunnel.
Write-Host ""
Write-Host "Starting Cloudflare tunnel -> http://127.0.0.1:7860"
Write-Host "Look for the line:  https://<random>.trycloudflare.com  -- that is your phone URL."
Write-Host "(Keep this window open. Ctrl+C closes the tunnel.)"
Write-Host ""
& $cf tunnel --url http://127.0.0.1:7860
