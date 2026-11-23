#!/bin/sh
# Build, sign, and notarize the macOS release disk image. GitHub publishing is separate.
set -eu
cd "$(dirname "$0")/.."

: "${SIGN_IDENTITY:?Set SIGN_IDENTITY to the Developer ID Application signing identity.}"
: "${NOTARY_KEYCHAIN_PROFILE:?Set NOTARY_KEYCHAIN_PROFILE to a notarization profile in Keychain.}"

version=$(plutil -extract CFBundleShortVersionString raw -o - Resources/Info.plist)
name="Agent-S-macOS-${version}-arm64.dmg"
if [ -e "build/$name" ] || [ -e "build/$name.sha256" ]; then
  echo "Release image already exists at build/$name." >&2
  exit 1
fi

SIGN_TIMESTAMP=1 ./scripts/bundle.sh release "$@"
app="build/Agent S.app"
if [ "$(lipo -archs "$app/Contents/MacOS/AgentS")" != arm64 ]; then
  echo "Build the release with an arm64 Xcode toolchain." >&2
  exit 1
fi
signature=$(codesign --display --verbose=4 "$app" 2>&1)
if ! printf '%s\n' "$signature" | grep -q '^Authority=Developer ID Application:' || \
   ! printf '%s\n' "$signature" | grep -q '^Timestamp='; then
  echo "The release app needs a Developer ID Application signature and secure timestamp." >&2
  exit 1
fi
codesign --verify --deep --strict "$app"

work=$(mktemp -d "$PWD/build/release.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir "$work/contents"
ditto "$app" "$work/contents/Agent S.app"
ln -s /Applications "$work/contents/Applications"
image="$work/$name"
hdiutil create -quiet -format UDZO -volname "Agent S" -srcfolder "$work/contents" "$image"
codesign --sign "$SIGN_IDENTITY" --timestamp "$image"

if ! xcrun notarytool submit "$image" --keychain-profile "$NOTARY_KEYCHAIN_PROFILE" \
     --wait --output-format json > "$work/notary.json"; then
  cat "$work/notary.json"
  exit 1
fi
cat "$work/notary.json"
if [ "$(plutil -extract status raw -o - "$work/notary.json")" != Accepted ]; then
  echo "Notarization failed. Use the submission ID above with xcrun notarytool log." >&2
  exit 1
fi
xcrun stapler staple "$image"
xcrun stapler validate "$image"
codesign --verify "$image"

(cd "$work" && shasum -a 256 "$name" > "$name.sha256")
mv "$image" "$work/$name.sha256" build/
printf 'Built build/%s and build/%s.sha256\n' "$name" "$name"
