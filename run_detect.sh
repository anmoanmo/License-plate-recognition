#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python detect.py --source test_images/3.jpg
