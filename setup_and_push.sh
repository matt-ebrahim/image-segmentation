#!/bin/bash

# Configuration
REPO_NAME="medical-image-segmentation"
DESCRIPTION="U-Net medical image segmentation experiment using Hippocampus dataset"

echo "Attempting to setup git and push to remote..."

# Check if we are in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not a git repository."
    exit 1
fi

# Rename branch to main
echo "Setting branch to main..."
git branch -M main

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo "GitHub CLI detected. Attempting to create remote repository..."
    
    # Check if authenticated
    if gh auth status &> /dev/null; then
        gh repo create "$REPO_NAME" --public --description "$description" --source=. --remote=origin --push
        echo "Repository created and pushed successfully via GitHub CLI!"
        exit 0
    else
        echo "GitHub CLI found, but you are not logged in."
        echo "Please run 'gh auth login' and try again, or follow the manual steps below."
    fi
fi

# Fallback: Manual setup
echo "Please create a new repository on GitHub named '$REPO_NAME'."
echo "Once created, paste the remote URL below (e.g., https://github.com/username/$REPO_NAME.git):"
read REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "Error: URL not provided."
    exit 1
fi

echo "Adding remote origin..."
git remote add origin "$REPO_URL"

echo "Pushing to remote..."
git push -u origin main

echo "Done! Your code is now on GitHub."
