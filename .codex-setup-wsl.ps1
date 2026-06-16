$ErrorActionPreference = "Continue"
Write-Host "== Enabling WSL and Virtual Machine Platform =="
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
Write-Host "== Installing/updating WSL =="
wsl --install -d Ubuntu
wsl --set-default-version 2
wsl --update
Write-Host "== Done. If you see a restart-required message, please reboot Windows, then come back to Codex and say: 继续 =="
Read-Host "Press Enter to close"
