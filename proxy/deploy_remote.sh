#!/usr/bin/env bash
set -euo pipefail
PUB="/home/u320837031/domains/draft.bertholomus.com/public_html"
BAK="/home/u320837031/backups/draft-bertholomus-pre-drs"
mkdir -p "$BAK"
if [ -f "$PUB/default.php" ]; then
  mv "$PUB/default.php" "$BAK/default.php"
fi
tar -xzf /home/u320837031/drs-proxy.tar.gz -C "$PUB"
chmod 644 "$PUB/index.php" "$PUB/.htaccess"
chmod 600 /home/u320837031/draft-proxy-secret.txt
rm -f /home/u320837031/drs-proxy.tar.gz /home/u320837031/drs_deploy_remote.sh
ls -la "$PUB"
sha256sum "$PUB/index.php" "$PUB/.htaccess"
