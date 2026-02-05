#!/bin/bash

# This script clones the required model repositories into the repos/ directory.
# Run this script from the root of the Vid-Bolt-GPU-API project after cloning.

# Exit on error
set -e

echo "Starting setup of external repositories..."

# Create repos directory if it doesn't exist
mkdir -p repos
cd repos

# Define repositories
REPOS=(
    "https://github.com/Lightricks/LTX-2.git"
    "https://github.com/Tongyi-MAI/Z-Image.git"
    "https://github.com/ModelTC/LightX2V.git"
    "https://github.com/ace-step/ACE-Step.git"
    "https://github.com/facebookresearch/audiocraft.git"
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
        
        echo "Found existing directory '$folder' but it's not a git repository (likely vendored)."
        echo "Skipping '$folder'..."
        continue
    fi
    
    echo "Cloning $folder from $repo..."
    git clone "$repo"
    
    # Remove .git directory to vendor the repo
    echo "Removing .git directory from $folder to vendor..."
    rm -rf "$folder/.git"
done

cd ..
echo "------------------------------------------"
echo "All external repositories have been set up in repos/ directory."
