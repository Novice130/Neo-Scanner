#!/usr/bin/env bash
set -e

echo "=== Building Neo Scanner for macOS ==="

# Activate virtual environment
source .venv/bin/activate

# 1. Run PyInstaller to build Neo Scanner.app
echo "--- Running PyInstaller ---"
pyinstaller neo_scanner.spec --noconfirm

# Ensure Frameworks/Python exists as a hard file
if [ -f "dist/Neo Scanner.app/Contents/Frameworks/Python.framework/Versions/3.11/Python" ]; then
  rm -f "dist/Neo Scanner.app/Contents/Frameworks/Python"
  cp "dist/Neo Scanner.app/Contents/Frameworks/Python.framework/Versions/3.11/Python" "dist/Neo Scanner.app/Contents/Frameworks/Python"
fi

# Fix OpenSSL libcrypto conflict with opencv-python on macOS
if [ -f "/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib" ]; then
  cp -f /opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib "dist/Neo Scanner.app/Contents/Frameworks/libcrypto.3.dylib"
  cp -f /opt/homebrew/opt/openssl@3/lib/libssl.3.dylib "dist/Neo Scanner.app/Contents/Frameworks/libssl.3.dylib"
fi

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
