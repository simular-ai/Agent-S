#!/bin/sh
# Builds Agent S.app into ./build, signed with the virtualization entitlement.
# Signed with the local "Agent S Development" certificate when it exists (scripts/make-signing-cert.sh),
# so the Keychain's "Always Allow" lasts across builds; otherwise ad hoc (macOS asks again after
# every build). For distribution, sign with a Developer ID and notarize instead (see README).
set -e
cd "$(dirname "$0")/.."
CONFIG=${1:-release}
if [ "$#" -gt 0 ]; then shift; fi
swift build -c "$CONFIG" "$@"
BIN=$(swift build -c "$CONFIG" "$@" --show-bin-path)/AgentS
APP="build/Agent S.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/AgentS"
cp Resources/Info.plist "$APP/Contents/Info.plist"
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
cp Resources/ThirdPartyNotices.txt "$APP/Contents/Resources/ThirdPartyNotices.txt"
IDENTITY=${SIGN_IDENTITY:-}
if [ -z "$IDENTITY" ] && security find-certificate -c "Agent S Development" >/dev/null 2>&1; then
  IDENTITY="Agent S Development"
fi
codesign --force --sign "${IDENTITY:--}" --options runtime \
  --entitlements Resources/AgentS.entitlements "$APP"
codesign --verify --verbose=1 "$APP"
echo "Built $APP"
