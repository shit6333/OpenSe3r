#!/usr/bin/env bash

set -euo pipefail

IMAGE="/mnt/HDD4/ricky/data/scannet_processed_test/scene0000_00/color/000050.png"
SAVE_DIR="./proxyclip_test"

mkdir -p "$SAVE_DIR"

for i in $(seq 0 36); do
    echo "Running decoder_layer=$i ..."
    
    python amb3r/proxyclip_test.py \
        --image "$IMAGE" \
        --decoder_layer "$i" \
        --save_dir "$SAVE_DIR"

    SRC="${SAVE_DIR}/summary_compare.png"
    DST="${SAVE_DIR}/summary_compare_dec${i}.png"

    if [ -f "$SRC" ]; then
        mv "$SRC" "$DST"
        echo "Saved: $DST"
    else
        echo "Warning: $SRC not found after decoder_layer=$i"
    fi
done

echo "Done."