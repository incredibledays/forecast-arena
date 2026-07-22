#!/usr/bin/env bash
# Rebuild the frontend static assets:
#   1. Compile Tailwind (scans templates → static/css/tailwind.css)
#   2. Pre-gzip every asset our /static/<f> override knows about
#      (see app.py `_STATIC_GZ_TYPES`). The Werkzeug dev server can't
#      compress on-the-fly for send_from_directory, so we ship pre-
#      compressed .gz variants next to each file.
#
# Run this after editing HTML/CSS classes, or once after cloning.
set -euo pipefail
cd "$(dirname "$0")"

# 1. Tailwind
npx tailwindcss -i tailwind.input.css -o static/css/tailwind.css --minify

# 2. Pre-gzip CSS + JS in static/. `-k` keeps the original next to the .gz,
# `-9` maxes compression (these are static assets shipped once and cached).
find static -type f \( -name '*.css' -o -name '*.js' -o -name '*.svg' -o -name '*.json' \) -print0 \
  | xargs -0 -n1 gzip -kf9

echo
echo "Done. Sizes:"
find static -type f \( -name '*.css' -o -name '*.js' \) -not -name '*.gz' -exec bash -c '
  orig=$(stat -c%s "$1")
  gz=$(stat -c%s "$1.gz" 2>/dev/null || echo 0)
  printf "  %-42s %7d → %7d bytes (%d%%)\n" "$1" "$orig" "$gz" $(( gz * 100 / (orig + 1) ))
' _ {} \;
