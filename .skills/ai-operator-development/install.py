#!/usr/bin/env python3
"""
AI Operator Development - Python Installer
Run: python install.py
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50 + "\n")

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed: {e}")
        return False

def main():
    print_header("AI Operator Development - Installer")

    # Determine skills directory based on platform
    system = platform.system()
    if system == "Windows":
        skills_dir = os.path.join(os.getenv('APPDATA', ''), 'Claude', 'skills')
    else:  # Linux/Mac
        skills_dir = os.path.join(os.path.expanduser('~'), '.claude', 'skills')

    target_dir = os.path.join(skills_dir, 'ai-operator-development')
    repo_url = "https://github.com/vogtsw/operator_pre.git"

    print(f"Platform: {system}")
    print(f"Skills directory: {skills_dir}")
    print(f"Target: {target_dir}")
    print(f"Repository: {repo_url}\n")

    # Create skills directory
    if not os.path.exists(skills_dir):
        print(f"Creating skills directory...")
        os.makedirs(skills_dir, exist_ok=True)

    # Remove existing installation
    if os.path.exists(target_dir):
        print("Removing existing installation...")
        run_command(f'rm -rf "{target_dir}"' if system != "Windows" else f'rmdir /s /q "{target_dir}"',
                   "Clean up")

    # Clone repository
    print("\nInstalling from GitHub...\n")
    success = run_command(f'git clone "{repo_url}" "{target_dir}"', "Clone repository")

    if success:
        print_header("Installation Successful!")
        print(f"Installed to: {target_dir}\n")
        print("Usage in Claude Code:")
        print("  /ai-operator-development model.py")
        print("\nOr:")
        print('  Use the ai-operator-development skill to analyze my model')
        print()

        # Verify
        skill_file = os.path.join(target_dir, 'SKILL.md')
        if os.path.exists(skill_file):
            print("Verification: SKILL.md found ✓")
        return 0
    else:
        print_header("Installation Failed!")
        print("\nTroubleshooting:")
        print("1. Make sure Git is installed: git --version")
        print("2. Check internet connection")
        print("3. Try running with appropriate permissions\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
