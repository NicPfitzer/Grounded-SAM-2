#!/bin/bash
set -euo pipefail

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Define available Grounding DINO checkpoints
declare -A GDINO_VARIANTS=(
    [swint_ogc]="v0.1.0-alpha/groundingdino_swint_ogc.pth"
    [swinb_cogcoor]="v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
)

usage() {
    echo "Usage: $0 [variant]"
    echo
    echo "Available variants:"
    for key in "${!GDINO_VARIANTS[@]}"; do
        printf "  %-15s %s\n" "$key" "${GDINO_VARIANTS[$key]}"
    done | sort
    exit 1
}

variant="${1:-swint_ogc}"

if [[ "$variant" == "--help" || "$variant" == "-h" || "$variant" == "--list" ]]; then
    usage
fi

if [[ -z "${GDINO_VARIANTS[$variant]:-}" ]]; then
    echo "Unknown variant '$variant'."
    usage
fi

# Define the URL for the base checkpoint
BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/"
filename_path="${GDINO_VARIANTS[$variant]}"
checkpoint_url="${BASE_URL}${filename_path}"

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Download only the base checkpoint using wget
filename="${filename_path##*/}"
echo "Downloading ${filename} checkpoint..."
$CMD "$checkpoint_url" || { echo "Failed to download checkpoint from $checkpoint_url"; exit 1; }

echo "Checkpoint ${filename} downloaded successfully."
