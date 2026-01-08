#!/bin/bash
set -e

# Ensure script is executed from its directory
cd "$(dirname "$0")"

# Target platform: linux/amd64 (x86_64)
PLATFORM="linux/amd64"
IMAGE_NAME="arx5-sdk-x86"

echo "Building Docker image for $PLATFORM..."

# Build using the specific Dockerfile.x86
# docker build -f Dockerfile.x86 -t $IMAGE_NAME:latest . --network host
# Note: running from unix/docker directory, context is parent directory (root of repo)
docker build \
 --build-arg http_proxy=http://172.17.0.1:7890 \
 --build-arg https_proxy=http://172.17.0.1:7890 \
  -f Dockerfile.x86 -t $IMAGE_NAME:latest ..

echo "Build complete. Image: $IMAGE_NAME:latest"
