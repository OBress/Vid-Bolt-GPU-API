#!/bin/bash

# This script clones the required model repositories into the Vid-Bolt-GPU-API directory.
# Run this script from the root of the Vid-Bolt-GPU-API project after cloning.

# Exit on error
set -e

echo "Starting setup of external repositories..."

# Define repositories
REPOS=(
    "https://github.com/Lightricks/LTX-2.git"
    "https://github.com/Tongyi-MAI/Z-Image.git"
    "https://github.com/ModelTC/LightX2V.git"
    "https://github.com/jamichss/Stream-DiffVSR.git"
)

for repo in "${REPOS[@]}"; do
    # Get the folder name from the URL (e.g., LTX-2)
    folder=$(basename "$repo" .git)
    
    echo "------------------------------------------"
    if [ -d "$folder" ]; then
        # Check if it's already a git repo
        if [ -d "$folder/.git" ]; then
            echo "$folder already exists and is a git repository. Skipping clone."
            continue
        fi
        
        echo "Found existing directory '$folder' but it's not a git repository (likely an empty gitlink)."
        echo "Cleaning up '$folder' to allow for a fresh clone..."
        rm -rf "$folder"
    fi
    
    echo "Cloning $folder from $repo..."
    git clone "$repo"
done

echo "------------------------------------------"
echo "All external repositories have been set up successfully."
