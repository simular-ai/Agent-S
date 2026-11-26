#!/bin/sh
# Builds Agent S.app into ./build, signed with the virtualization entitlement, with the DeepSeek
# Harness runtime from scripts/vendor-deepseek.sh: Node in Contents/Helpers, its packages in
# Contents/Resources/DeepSeek, every Mach-O signed with the app's identity.
# Signed with the local "Agent S Development" certificate when it exists (scripts/make-signing-cert.sh),
# so the Keychain's "Always Allow" lasts across builds; otherwise ad hoc (macOS asks again after
# every build). For distribution, sign with a Developer ID and notarize instead (see README).
set -e
cd "$(dirname "$0")/.."
CONFIG=${1:-release}
if [ "$#" -gt 0 ]; then shift; fi
# Keep the linked SDK version so macOS uses the native appearance of the selected SDK.
# SwiftPM's Swift Build engine can otherwise record the minimum OS as the SDK.
SDK_VERSION=$(xcrun --sdk macosx --show-sdk-version)
MINIMUM_OS=$(plutil -extract LSMinimumSystemVersion raw -o - Resources/Info.plist)
set -- "$@" -Xlinker -platform_version -Xlinker macos -Xlinker "$MINIMUM_OS" -Xlinker "$SDK_VERSION"
swift build -c "$CONFIG" "$@"
BIN=$(swift build -c "$CONFIG" "$@" --show-bin-path)/AgentS
APP="build/Agent S.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/AgentS"
cp Resources/Info.plist "$APP/Contents/Info.plist"
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
cp Resources/ThirdPartyNotices.txt "$APP/Contents/Resources/ThirdPartyNotices.txt"
cp ../../LICENSE "$APP/Contents/Resources/LICENSE"
./scripts/vendor-deepseek.sh
RUNTIME=build/deepseek-runtime
mkdir -p "$APP/Contents/Helpers" "$APP/Contents/Resources/DeepSeek"
cp "$RUNTIME/bin/node" "$APP/Contents/Helpers/node"
cp -R "$RUNTIME/node_modules" "$RUNTIME/package.json" "$RUNTIME/NODE-LICENSE" "$APP/Contents/Resources/DeepSeek/"
IDENTITY=${SIGN_IDENTITY:-}
if [ -z "$IDENTITY" ] && security find-certificate -c "Agent S Development" >/dev/null 2>&1; then
  IDENTITY="Agent S Development"
fi
set --
if [ "${SIGN_TIMESTAMP:-}" = 1 ]; then set -- --timestamp; fi
# Nested code first. The native modules Node loads must share the app's Team ID.
while IFS= read -r file; do
  codesign --force --sign "${IDENTITY:--}" --options runtime "$@" "$APP/Contents/Resources/DeepSeek/node_modules/$file"
done < "$RUNTIME/mach-o.txt"
NODE_ENTITLEMENTS=Resources/Node.entitlements
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" --entitlements "$NODE_ENTITLEMENTS" "$APP/Contents/Helpers/node"
if codesign --display --verbose=2 "$APP/Contents/Helpers/node" 2>&1 | grep -q '^TeamIdentifier=not set'; then
  # Without a Team ID (ad hoc or a local certificate), library validation would reject every
  # native module, so local builds let Node load modules signed by the same local identity.
  NODE_ENTITLEMENTS=build/node-local.entitlements
  plutil -insert com.apple.security.cs.disable-library-validation -bool YES -o "$NODE_ENTITLEMENTS" Resources/Node.entitlements
  codesign --force --sign "${IDENTITY:--}" --options runtime "$@" --entitlements "$NODE_ENTITLEMENTS" "$APP/Contents/Helpers/node"
fi
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" \
  --entitlements Resources/AgentS.entitlements "$APP"
codesign --verify --verbose=1 "$APP"
echo "Built $APP"
