#!/usr/bin/env bash
# this script is automatically run from CMakeLists.txt

BUILD_ROOT=$PWD
TORCH_PREFIX=$PWD/torch

echo "[Pre-build]  dependency installer script running... (jetson-video)"
echo "[Pre-build]  build root directory:       $BUILD_ROOT"


# break on errors
#set -e


# install packages
sudo apt-get update
sudo apt-get install -y libqt4-dev qt4-dev-tools libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev
sudo apt-get update


echo "[Pre-build]  Finished jetson-video CMakePreBuild script"
