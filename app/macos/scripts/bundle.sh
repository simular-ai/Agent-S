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

APP_INFO=Resources/Info.plist
if [ -n "${SPARKLE_PUBLIC_ED_KEY:-}" ]; then
  # Configure updates before building. An ordinary local build has no update feed.
  mkdir -p build
  APP_INFO=build/Info-Sparkle.plist
  python3 -I - Resources/Info.plist "$APP_INFO" "$SPARKLE_PUBLIC_ED_KEY" \
    "${SPARKLE_FEED_URL:-https://agent-s.ai/appcast.xml}" <<'PYTHON'
import base64
import plistlib
import sys
from urllib.parse import urlsplit

source, destination, public_key, feed_url = sys.argv[1:]
try:
    key_bytes = base64.b64decode(public_key, validate=True)
except ValueError:
    key_bytes = b""
if len(key_bytes) != 32 or base64.b64encode(key_bytes).decode("ascii") != public_key:
    sys.exit("SPARKLE_PUBLIC_ED_KEY must be a base64-encoded 32-byte Ed25519 public key.")

try:
    feed = urlsplit(feed_url)
    valid_feed = (
        feed.scheme == "https" and feed.hostname and feed.port != 0
        and feed.username is None and feed.password is None and not feed.fragment
        and not any(character.isspace() or ord(character) < 32 for character in feed_url)
        and "\\" not in feed_url
    )
except ValueError:
    valid_feed = False
if not valid_feed:
    sys.exit("SPARKLE_FEED_URL must be an HTTPS URL without credentials or a fragment.")

with open(source, "rb") as file:
    info = plistlib.load(file)
info.update(
    SUFeedURL=feed_url,
    SUPublicEDKey=public_key,
    SUEnableAutomaticChecks=True,
    SUAutomaticallyUpdate=False,
    SUAllowsAutomaticUpdates=False,
    SUVerifyUpdateBeforeExtraction=True,
    SURequireSignedFeed=True,
)
with open(destination, "wb") as file:
    plistlib.dump(info, file, sort_keys=False)
PYTHON
fi

# Keep the linked SDK version so macOS uses the native appearance of the selected SDK.
# SwiftPM's Swift Build engine can otherwise record the minimum OS as the SDK.
SDK_VERSION=$(xcrun --sdk macosx --show-sdk-version)
MINIMUM_OS=$(plutil -extract LSMinimumSystemVersion raw -o - Resources/Info.plist)
set -- "$@" -Xlinker -platform_version -Xlinker macos -Xlinker "$MINIMUM_OS" -Xlinker "$SDK_VERSION"
swift build -c "$CONFIG" "$@"
BIN_DIR=$(swift build -c "$CONFIG" "$@" --show-bin-path)
APP="build/Agent S.app"
SPARKLE_FRAMEWORK="$APP/Contents/Frameworks/Sparkle.framework"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Frameworks"
cp "$BIN_DIR/AgentS" "$APP/Contents/MacOS/AgentS"
cp "$APP_INFO" "$APP/Contents/Info.plist"
cp Resources/AppIcon.icns Resources/ThirdPartyNotices.txt Resources/Sparkle-LICENSE "$APP/Contents/Resources/"
# Versioned frameworks use symlinks to expose their executable and resources.
ditto "$BIN_DIR/Sparkle.framework" "$SPARKLE_FRAMEWORK"
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
# Sign Sparkle's helper executables and bundles first, then the framework that seals them.
SPARKLE_CONTENTS="$SPARKLE_FRAMEWORK/Versions/Current"
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" "$SPARKLE_CONTENTS/XPCServices/Installer.xpc"
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" --preserve-metadata=entitlements \
  "$SPARKLE_CONTENTS/XPCServices/Downloader.xpc"
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" "$SPARKLE_CONTENTS/Autoupdate"
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" "$SPARKLE_CONTENTS/Updater.app"
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" "$SPARKLE_FRAMEWORK"

# The native modules Node loads must share the app's Team ID.
while IFS= read -r file; do
  codesign --force --sign "${IDENTITY:--}" --options runtime "$@" "$APP/Contents/Resources/DeepSeek/node_modules/$file"
done < "$RUNTIME/mach-o.txt"
NODE_ENTITLEMENTS=Resources/Node.entitlements
APP_ENTITLEMENTS=Resources/AgentS.entitlements
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" --entitlements "$NODE_ENTITLEMENTS" "$APP/Contents/Helpers/node"
if codesign --display --verbose=2 "$APP/Contents/Helpers/node" 2>&1 | grep -q '^TeamIdentifier=not set'; then
  # Local signatures lack a Team ID. Allow Agent S to load Sparkle and Node to load its modules.
  NODE_ENTITLEMENTS=build/node-local.entitlements
  APP_ENTITLEMENTS=build/agent-s-local.entitlements
  LOCAL_LIBRARIES_KEY='com\.apple\.security\.cs\.disable-library-validation'
  plutil -insert "$LOCAL_LIBRARIES_KEY" -bool YES -o "$NODE_ENTITLEMENTS" Resources/Node.entitlements
  plutil -insert "$LOCAL_LIBRARIES_KEY" -bool YES -o "$APP_ENTITLEMENTS" Resources/AgentS.entitlements
  codesign --force --sign "${IDENTITY:--}" --options runtime "$@" --entitlements "$NODE_ENTITLEMENTS" "$APP/Contents/Helpers/node"
fi
codesign --force --sign "${IDENTITY:--}" --options runtime "$@" \
  --entitlements "$APP_ENTITLEMENTS" "$APP"
codesign --verify --deep --strict --verbose=1 "$APP"
echo "Built $APP"
