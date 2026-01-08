# Use Ubuntu 22.04 (compatible with ROS Humble era)
FROM ubuntu:22.04

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# git, cmake, build-essential: for building SOEM and project
# curl, wget: utilities
# libusb-1.0-0-dev, libudev-dev: for joycon/hidapi
# can-utils: for CAN debugging
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libusb-1.0-0-dev \
    libudev-dev \
    can-utils \
    kmod \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (Mamba)
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda \
    && rm Miniforge3-$(uname)-$(uname -m).sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Set working directory
WORKDIR /app

# Copy environment file first to leverage cache
COPY conda_environments/py310_environment.yaml /app/conda_environments/

# Prepare environment file for ARM64:
# 1. Remove the problematic 'conda-forge::soem' line which fails on aarch64
# 2. Create the environment
RUN sed -i '/conda-forge::soem/d' /app/conda_environments/py310_environment.yaml && \
    mamba env create -f /app/conda_environments/py310_environment.yaml && \
    mamba clean --all -f -y

# Set default shell to use the conda environment
SHELL ["conda", "run", "-n", "arx-py310", "/bin/bash", "-c"]

# Manually build and install SOEM (Simple Open EtherCAT Master) for ARM64
# Instructions from the environment yaml file
RUN git clone https://github.com/OpenEtherCATsociety/SOEM.git /tmp/SOEM && \
    cd /tmp/SOEM && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/SOEM

# Copy the rest of the project
COPY . /app

# Build the Python C++ Extension
# 1. Create build directory
# 2. Run CMake and Make
# 3. Move the generated .so file to the python root so scripts can import it
RUN cd python && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    # Move the compiled shared object to python/ directory for easy import
    # The output path is controlled by CMake, look specifically in the build dir or absolute path
    find . -name "arx5_interface.cpython-*.so" -exec cp {} ../ \;

# Install the joystick package
RUN pip install pygame

# Setup environment variables needed at runtime
# Ensure libraries in lib/aarch64 are found (if running on aarch64)
ENV LD_LIBRARY_PATH=/app/lib/aarch64:$LD_LIBRARY_PATH
ENV PYTHONPATH=/app/python:$PYTHONPATH

# Default command: Open a bash shell in the environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "arx-py310"]
CMD ["/bin/bash"]
