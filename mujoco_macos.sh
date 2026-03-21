#!/bin/bash

uv venv --clear
source .venv/bin/activate

# Install the package
uv pip install "reachy-mini[mujoco]"

# Create the lib directory and symlink
mkdir -p .venv/lib
ln -s \
    ~/.local/share/uv/python/cpython-3.12.11-macos-x86_64-none/lib/libpython3.12.dylib \
    .venv/lib/libpython3.12.dylib
