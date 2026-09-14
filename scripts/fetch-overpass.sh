#!/bin/bash
# Retry Overpass across mirrors until we get JSON back.
QL="$(dirname "$0")/overpass.ql"
OUT="$(dirname "$0")/../data/raw-overpass.json"
EPS=(
  https://overpass-api.de/api/interpreter
  https://overpass.private.coffee/api/interpreter
  https://overpass.kumi.systems/api/interpreter
  https://overpass.osm.ch/api/interpreter
  https://overpass.osm.jp/api/interpreter
)
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  for ep in "${EPS[@]}"; do
    echo "attempt $attempt -> $ep"
    curl -s -m 180 --data-urlencode "data@$QL" "$ep" -o "$OUT.tmp"
    if jq -e '.elements|length' "$OUT.tmp" >/dev/null 2>&1; then
      mv "$OUT.tmp" "$OUT"
      echo "OK: $(jq '.elements|length' "$OUT") elements from $ep"
      exit 0
    fi
    sleep 5
  done
  sleep 15
done
echo "FAILED"
exit 1
