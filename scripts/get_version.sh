#!/bin/bash

# This script computes the next version number based on the last git tag and whether there have been changes since that tag.
# It is designed to be run by GitHub Actions as part of a release workflow, 
# but can also be run locally for testing.
# It uses git commands to find the last tag, check for changes, and compute the new version number based on the current date and an optional suffix.
# but doesn't commit anything. 


# Exit immediately if a command exits with a non-zero status
set -e

# Find the last tag
last_tag=$(git tag --list --sort=-creatordate | grep -E '^v?[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}(\.[0-9]+)?$' | head -n1 || true)

# Determine the base commit to check against
if [ -z "$last_tag" ]; then
    echo "No previous tags found. Using first commit as base." >&2
    base=$(git rev-list --max-parents=0 HEAD)
else
    echo "Last tag found: $last_tag" >&2
    base=$last_tag
fi

# Check for changes, ignoring version.txt
changed=$(git diff --name-only "$base" HEAD | grep -v '^version.txt$' || true)

if [ -z "$changed" ]; then
    echo "No changes found since last release. Exiting." >&2
    # Output an empty string and exit successfully
    echo ""
    exit 0
fi

# Compute the new version
today=$(date -u +%Y.%-m.%-d)

if [[ "$last_tag" =~ ^v?($today)(\.([0-9]+))?$ ]]; then
    suffix=${BASH_REMATCH[3]:-0}
    new_version="$today.$((suffix+1))"
else
    new_version="$today"
fi

echo "Changes detected. Computing new version..." >&2

# Output ONLY the new version string to stdout so GitHub Actions can capture it
echo "$new_version"