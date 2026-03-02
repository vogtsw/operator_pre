@echo off
REM AI Operator Development - One-Click Installer for Windows
REM Double-click this file to install the skill

setlocal EnableDelayedExpansion

echo ========================================
echo AI Operator Development - Installer
echo ========================================
echo.

REM Get skills directory
set "SKILLS_DIR=%APPDATA%\Claude\skills"
set "TARGET_DIR=%SKILLS_DIR%\ai-operator-development"

echo Skills directory: %SKILLS_DIR%
echo Target: %TARGET_DIR%
echo.

REM Create directory if not exists
if not exist "%SKILLS_DIR%" (
    echo Creating skills directory...
    mkdir "%SKILLS_DIR%"
)

REM Check if already installed
if exist "%TARGET_DIR%" (
    echo Skill already installed. Updating...
    rmdir /s /q "%TARGET_DIR%"
)

REM Clone from GitHub
echo Installing from GitHub...
echo.

git clone https://github.com/vogtsw/operator_pre.git "%TARGET_DIR%"

if errorlevel 1 (
    echo.
    echo ========================================
    echo Installation Failed!
    echo ========================================
    echo.
    echo Troubleshooting:
    echo 1. Make sure Git is installed
    echo 2. Check internet connection
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Successful!
echo ========================================
echo.
echo Installed to: %TARGET_DIR%
echo.
echo Usage in Claude Code:
echo   /ai-operator-development model.py
echo.
echo Or:
echo   Use the ai-operator-development skill to analyze my model
echo.

pause
endlocal
