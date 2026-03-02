# AI Operator Development - One-Click Installer
# PowerShell script to install the skill for Claude Code

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Operator Development - Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Determine skills directory
$skillsDir = Join-Path $env:APPDATA "Claude\skills"

# Create directory if not exists
if (-not (Test-Path $skillsDir)) {
    Write-Host "Creating skills directory: $skillsDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $skillsDir -Force | Out-Null
}

# Installation target
$targetDir = Join-Path $skillsDir "ai-operator-development"

Write-Host "Skills directory: $skillsDir" -ForegroundColor Green
Write-Host "Target: $targetDir" -ForegroundColor Green
Write-Host ""

# Check if already installed
if (Test-Path $targetDir) {
    Write-Host "Skill already installed. Updating..." -ForegroundColor Yellow
    Remove-Item -Path $targetDir -Recurse -Force
}

# Clone from GitHub
Write-Host "Installing from GitHub..." -ForegroundColor Cyan
Write-Host ""

try {
    git clone https://github.com/vogtsw/operator_pre.git $targetDir

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Installation Successful!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Installed to: $targetDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage in Claude Code:" -ForegroundColor Cyan
    Write-Host "  /ai-operator-development model.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Or:" -ForegroundColor Cyan
    Write-Host "  Use the ai-operator-development skill to analyze my model" -ForegroundColor White
    Write-Host ""

    # Verify installation
    $skillFile = Join-Path $targetDir "SKILL.md"
    if (Test-Path $skillFile) {
        Write-Host "Verification: SKILL.md found" -ForegroundColor Green
    }

} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Installation Failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure Git is installed: git --version" -ForegroundColor White
    Write-Host "2. Check internet connection" -ForegroundColor White
    Write-Host "3. Try running PowerShell as Administrator" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
