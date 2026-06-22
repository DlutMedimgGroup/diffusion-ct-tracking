#!/usr/bin/env bash
set -e

echo "========================================"
echo " Diffusion-CT-Tracking Demo Downloader"
echo "========================================"

# -------------------------------------------------
# Resolve project root (script-safe)
# -------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[Info] Project root: $PROJECT_ROOT"

# -------------------------------------------------
# Create folders
# -------------------------------------------------
echo "[Step] Creating folders..."

mkdir -p "$PROJECT_ROOT/trained_models/demo_coarse"
mkdir -p "$PROJECT_ROOT/trained_models/demo_local"
mkdir -p "$PROJECT_ROOT/demo_data"

# -------------------------------------------------
# Base URL (split to avoid 4open rewriting)
# -------------------------------------------------
BASE_PART1="https://github.com/zmr-anonymous"
BASE_PART2="/diffusion-ct-tracking/releases/download/v1.0"
BASE_URL="${BASE_PART1}${BASE_PART2}"

echo "[Info] Download base: $BASE_URL"

# -------------------------------------------------
# Download files
# -------------------------------------------------
echo "[Step] Downloading checkpoints and demo data..."

cd "$PROJECT_ROOT"

wget -c --show-progress "${BASE_URL}/stage1_coarse.pth"
wget -c --show-progress "${BASE_URL}/stage2_refine.pth"
wget -c --show-progress "${BASE_URL}/demo_data.zip"

# -------------------------------------------------
# Move checkpoints
# -------------------------------------------------
echo "[Step] Organizing files..."

mv -f stage1_coarse.pth trained_models/demo_coarse/
mv -f stage2_refine.pth trained_models/demo_local/

# -------------------------------------------------
# Unzip demo data
# -------------------------------------------------
echo "[Step] Extracting demo data..."

unzip -o demo_data.zip -d demo_data
rm -f demo_data.zip

echo "========================================"
echo " ✅ Download complete!"
echo "========================================"
echo
echo "Next steps:"
echo "  python run_inference.py --config configs/demo_coarse.toml"
echo