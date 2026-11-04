#!/usr/bin/env bash
# Renders linkedin.svg into linkedin.gif: one seamless 5.4-second loop.
#
# Each frame is a screenshot of the SVG in headless Chromium with its
# animations paused at that frame's time; ffmpeg then builds one palette for
# the whole loop and encodes the GIF with it.
#
# Settings (environment variables):
#   CHROME  a headless Chromium binary   (default: Playwright's headless shell)
#   FPS     frames per second            (default: 15)
#   SCALE   output width in pixels       (default: 1080)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
chrome="${CHROME:-$(ls ~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell | tail -1)}"
fps="${FPS:-15}"
scale="${SCALE:-1080}"

# One loop of the cover's animations, in seconds.
loop_seconds=5.4

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/frames"

{
  echo '<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:#070404}svg{display:block}</style></head><body>'
  cat "$here/linkedin.svg"
  echo "<script>const root = document.querySelector('svg'); root.pauseAnimations(); root.setCurrentTime(parseFloat(location.hash.slice(1)));</script></body></html>"
} > "$work/page.html"

frames=$(python3 -c "print(round($loop_seconds * $fps))")
for ((frame = 0; frame < frames; frame++)); do
  time=$(python3 -c "print($frame / $fps)")
  "$chrome" --no-sandbox --hide-scrollbars --window-size=1080,1350 \
    --screenshot="$work/frames/$(printf %04d "$frame").png" "file://$work/page.html#$time" > /dev/null 2>&1
done

ffmpeg -loglevel error -y -framerate "$fps" -i "$work/frames/%04d.png" \
  -vf "scale=$scale:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a" \
  -loop 0 "$here/linkedin.gif"

echo "wrote $here/linkedin.gif ($frames frames at $fps fps)"
