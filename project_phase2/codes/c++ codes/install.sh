#!/bin/bash

set -e

echo "Creating build directory..."
mkdir -p build
cd build

echo "Running CMake..."
cmake ..

echo "Building project..."
make -j$(nproc)

echo "Build complete. Executable is in build/SatellitePropagator"