#!/bin/bash
set -e

# Ensure script is executed from its directory
cd "$(dirname "$0")"

# Target platform: linux/arm64 (for Jetson, Raspberry Pi, etc.)
PLATFORM="linux/arm64"
IMAGE_NAME="arx5-sdk-arm64"

echo "Building Docker image for $PLATFORM..."

# Ensure a buildx builder is available and bootstrapped
BUILDER_NAME="multiarch-builder"
if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
	echo "Creating buildx builder: $BUILDER_NAME"
	docker buildx create --name "$BUILDER_NAME" --use
fi
echo "Bootstrapping buildx builder..."
docker buildx inspect --bootstrap >/dev/null 2>&1 || true
echo "Finished setting up buildx builder."

# Use Docker Buildx for multi-arch build
# Note: running from docker directory, context is parent directory (root of repo)
docker buildx build --network=host --platform $PLATFORM -t $IMAGE_NAME:latest --load -f Dockerfile.arm .. 

echo "Build complete. Image: $IMAGE_NAME:latest"
echo "To save this image to a tar file for transfer:"
echo "  docker save -o ${IMAGE_NAME}.tar ${IMAGE_NAME}:latest"
