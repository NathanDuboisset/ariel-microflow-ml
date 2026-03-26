#!/usr/bin/env bash
set -euo pipefail

# Export MLIR from the repo-root TFLite model using IREE.
# Requires: `iree-import-tflite` available in PATH.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN="${1:-"$ROOT/models/lenet_int8.tflite"}"
OUT="${2:-"$ROOT/models/lenet_int8.mlir"}"

if [[ ! -f "$IN" ]]; then
  echo "Input .tflite not found: $IN" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

iree-import-tflite "$IN" -o "$OUT"
echo "Wrote: $OUT"

