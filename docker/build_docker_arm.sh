#!/bin/bash
set -e

# Ensure script is executed from its directory
cd "$(dirname "$0")"

# Target platform: linux/arm64 (for Jetson, Raspberry Pi, etc.)
PLATFORM="linux/arm64"
IMAGE_NAME="arx5-sdk-arm64"

echo "Building Docker image for $PLATFORM..."

# Use Docker Buildx for multi-arch build
# You might need to run 'docker run --privileged --rm tonistiigi/binfmt --install all' if this is your first time cross-compiling
# Note: running from unix/docker directory, context is parent directory (root of repo)
docker buildx build --platform $PLATFORM -t $IMAGE_NAME:latest --load -f Dockerfile.arm .. 

echo "Build complete. Image: $IMAGE_NAME:latest"
echo "To save this image to a tar file for transfer:"
echo "  docker save -o ${IMAGE_NAME}.tar ${IMAGE_NAME}:latest"
