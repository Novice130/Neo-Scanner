#!/usr/bin/env bash
set -e

echo "=== Building Neo Scanner for macOS ==="

# Activate virtual environment
source .venv/bin/activate

# 1. Run PyInstaller to build Neo Scanner.app
echo "--- Running PyInstaller ---"
pyinstaller neo_scanner.spec --noconfirm

# 2. Package into .dmg using hdiutil
echo "--- Creating .dmg Disk Image ---"
rm -rf dist/dmg_staging dist/Neo_Scanner-macOS-v1.0.0.dmg
mkdir -p dist/dmg_staging
cp -R "dist/Neo Scanner.app" dist/dmg_staging/
ln -s /Applications dist/dmg_staging/Applications

hdiutil create -volname "Neo Scanner" \
  -srcfolder dist/dmg_staging \
  -ov -format UDZO \
  dist/Neo_Scanner-macOS-v1.0.0.dmg

rm -rf dist/dmg_staging

echo "=== Successfully built dist/Neo_Scanner-macOS-v1.0.0.dmg ==="
