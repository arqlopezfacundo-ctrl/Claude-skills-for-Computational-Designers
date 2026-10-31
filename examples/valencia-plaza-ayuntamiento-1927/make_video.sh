#!/usr/bin/env bash
# Full pipeline: Cycles render -> painterly post -> H.264 MP4 loop.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
RES=${RES:-1600x900}
SAMPLES=${SAMPLES:-6}
JOBS=${JOBS:-2}

$PY render.py --frames 0:480 --res "$RES" --samples "$SAMPLES" --out render/exr --skip-existing
$PY post.py --src render/exr --dst render/frames --jobs "$JOBS"
ffmpeg -y -framerate 24 -i render/frames/f%04d.png \
  -vf "scale=1920:1080:flags=lanczos,format=yuv420p" \
  -c:v libx264 -preset slow -crf 17 -tune film -movflags +faststart \
  valencia_plaza_1927_loop.mp4
