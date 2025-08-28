#!/bin/bash
set -e

# Load the .env file
set -o allexport
source .env

# Check if PYPI_ART_TOKEN is set
if [[ -z "${PYPI_ART_TOKEN}" ]]; then
    echo "Error: PYPI_ART_TOKEN is not set."
    exit 1
fi

# Delete the dist directory
rm -rf dist

# Build the package
uv run hatch build


# If the token is set, proceed with publishing
uv publish --username=__token__ --password=$PYPI_ART_TOKEN
