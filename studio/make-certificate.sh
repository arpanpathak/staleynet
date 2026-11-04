#!/usr/bin/env bash
# Creates a self-signed certificate so the studio can be served over HTTPS,
# which browsers require before they grant microphone access to a page that
# is not on localhost. Valid for this machine's host name and LAN addresses.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p certs

names="DNS:localhost,DNS:$(hostname),IP:127.0.0.1"
for address in $(hostname -I); do names="$names,IP:$address"; done

openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
  -keyout certs/studio.key -out certs/studio.crt \
  -subj "/CN=StaleyNET Studio" -addext "subjectAltName=$names" 2>/dev/null
echo "wrote certs/studio.crt and certs/studio.key for $names"
