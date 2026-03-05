#!/usr/bin/env bash
# Download GloVe 6B 50d word embeddings for NSCK V18 word HV seeding.
#
# Usage:
#   bash nsck/scripts/download_embeddings.sh
#
# The embeddings file (~170 MB) will be placed at:
#   nsck/data/embeddings/glove.6B.50d.txt
#
# This file is listed in .gitignore and must NOT be committed to the repo.
# After downloading, NSCK will automatically use GloVe-based HV seeding
# for known words and fall back to hash seeding for unknown words.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EMBED_DIR="$REPO_ROOT/data/embeddings"
ZIP_FILE="$EMBED_DIR/glove.6B.zip"
TXT_FILE="$EMBED_DIR/glove.6B.50d.txt"

if [ -f "$TXT_FILE" ]; then
    echo "GloVe embeddings already present at $TXT_FILE"
    WORD_COUNT=$(wc -l < "$TXT_FILE")
    echo "  -> $WORD_COUNT word vectors loaded"
    exit 0
fi

mkdir -p "$EMBED_DIR"

echo "Downloading GloVe 6B 50d embeddings (~170 MB)..."
curl -L --progress-bar \
    "https://nlp.stanford.edu/data/glove.6B.zip" \
    -o "$ZIP_FILE"

echo "Extracting glove.6B.50d.txt..."
unzip -o "$ZIP_FILE" "glove.6B.50d.txt" -d "$EMBED_DIR"
rm -f "$ZIP_FILE"

echo ""
echo "Done. Embeddings saved to: $TXT_FILE"
WORD_COUNT=$(wc -l < "$TXT_FILE")
echo "  -> $WORD_COUNT word vectors"
echo ""
echo "NSCK V18 will now use GloVe-based HV seeding for known words."
echo "Unknown words continue to use the hash fallback (backward compatible)."
