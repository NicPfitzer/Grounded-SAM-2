#!/bin/bash
set -euo pipefail

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

usage() {
    echo "Usage: $0 [variant]"
    echo
    echo "Available variants:"
    for key in "${!SAM2_VARIANTS[@]}"; do
        printf "  %-15s %s\n" "$key" "${SAM2_VARIANTS[$key]}"
    done | sort
    exit 1
}

# Define the SAM 2.1 checkpoints lookup table
declare -A SAM2_VARIANTS=(
    [tiny]="sam2.1_hiera_tiny.pt"
    [small]="sam2.1_hiera_small.pt"
    [base_plus]="sam2.1_hiera_base_plus.pt"
    [large]="sam2.1_hiera_large.pt"
)

variant="${1:-base_plus}"

if [[ "$variant" == "--help" || "$variant" == "-h" || "$variant" == "--list" ]]; then
    usage
fi

if [[ -z "${SAM2_VARIANTS[$variant]:-}" ]]; then
    echo "Unknown variant '$variant'."
    usage
fi

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the base URL for the SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
filename="${SAM2_VARIANTS[$variant]}"
url="${SAM2p1_BASE_URL}/${filename}"

echo "Downloading ${filename} checkpoint..."
$CMD "$url" || { echo "Failed to download checkpoint from $url"; exit 1; }

echo "Checkpoint ${filename} downloaded successfully."
