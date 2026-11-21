#!/bin/sh
# Creates "Agent S Development", a self-signed code-signing certificate in your login keychain, so
# every build of Agent S has the same signature and the Keychain's "Always Allow" keeps working
# after a rebuild. Local development only: it is trusted by nobody but this Mac's Keychain ACLs.
set -e
NAME="Agent S Development"
if security find-certificate -c "$NAME" >/dev/null 2>&1; then
  echo "\"$NAME\" already exists."; exit 0
fi
DIR=$(mktemp -d); trap 'rm -rf "$DIR"' EXIT
cat > "$DIR/cert.cnf" <<CNF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $NAME
[ext]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
CNF
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$DIR/cert.cnf" \
  -keyout "$DIR/key.pem" -out "$DIR/cert.pem" 2>/dev/null
PASS=$(openssl rand -hex 16)
openssl pkcs12 -export -legacy -inkey "$DIR/key.pem" -in "$DIR/cert.pem" -name "$NAME" \
  -passout pass:"$PASS" -out "$DIR/cert.p12" 2>/dev/null \
  || openssl pkcs12 -export -inkey "$DIR/key.pem" -in "$DIR/cert.pem" -name "$NAME" -passout pass:"$PASS" -out "$DIR/cert.p12"
security import "$DIR/cert.p12" -k ~/Library/Keychains/login.keychain-db -P "$PASS" -T /usr/bin/codesign
echo "Created \"$NAME\". Rebuild with scripts/bundle.sh; macOS asks once more, then Always Allow sticks."
