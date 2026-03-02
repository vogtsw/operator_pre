#!/bin/bash
# AI Operator Development - Installer for Linux/Mac
# Run: bash install.sh or ./install.sh

set -e

echo "========================================"
echo "AI Operator Development - Installer"
echo "========================================"
echo ""

# Determine skills directory
SKILLS_DIR="$HOME/.claude/skills"
TARGET_DIR="$SKILLS_DIR/ai-operator-development"
REPO_URL="https://github.com/vogtsw/operator_pre.git"

echo "Skills directory: $SKILLS_DIR"
echo "Target: $TARGET_DIR"
echo ""

# Create directory if not exists
if [ ! -d "$SKILLS_DIR" ]; then
    echo "Creating skills directory..."
    mkdir -p "$SKILLS_DIR"
fi

# Remove existing installation
if [ -d "$TARGET_DIR" ]; then
    echo "Removing existing installation..."
    rm -rf "$TARGET_DIR"
fi

# Clone from GitHub
echo "Installing from GitHub..."
echo ""

git clone "$REPO_URL" "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Installation Successful!"
    echo "========================================"
    echo ""
    echo "Installed to: $TARGET_DIR"
    echo ""
    echo "Usage in Claude Code:"
    echo "  /ai-operator-development model.py"
    echo ""
    echo "Or:"
    echo "  Use the ai-operator-development skill to analyze my model"
    echo ""

    # Verify
    if [ -f "$TARGET_DIR/SKILL.md" ]; then
        echo "Verification: SKILL.md found ✓"
    fi
else
    echo ""
    echo "========================================"
    echo "Installation Failed!"
    echo "========================================"
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure Git is installed: git --version"
    echo "2. Check internet connection"
    exit 1
fi
