#!/usr/bin/env bash
set -euo pipefail

# Run from the project root on AutoDL.
# Usage:
#   bash scripts/download_autodl_ccpd.sh [/root/autodl-tmp/ccpd_data] [max_images]

DATA_ROOT="${1:-/root/autodl-tmp/ccpd_data}"
MAX_IMAGES="${2:-0}"
ARCHIVE="$DATA_ROOT/downloads/CCPD2019.tar.xz"
RAW_DIR="$DATA_ROOT/raw"
OUTPUT_ROOT="datasets/ccpd_prepared"
URL="https://zenodo.org/records/15647076/files/CCPD2019.tar.xz?download=1"

mkdir -p "$DATA_ROOT/downloads" "$RAW_DIR" datasets

echo "Downloading CCPD2019 to $ARCHIVE"
curl -L --fail --continue-at - --output "$ARCHIVE" "$URL"

echo "Extracting to $RAW_DIR"
tar -xf "$ARCHIVE" -C "$RAW_DIR"

echo "Converting CCPD to project datasets"
python scripts/prepare_ccpd_dataset.py \
  --raw-dir "$RAW_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --max-images "$MAX_IMAGES" \
  --copy-mode hardlink

echo "Prepared datasets:"
echo "  YOLO: $OUTPUT_ROOT/plate_yolo_dataset/data.yaml"
echo "  Recognition: $OUTPUT_ROOT/plate_recognition_dataset"
