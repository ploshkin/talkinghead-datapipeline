#! /usr/bin/bash

JPEGHDF5_ROOT=$1
ANACONDA_PKGS=$2/pkgs

HDF5_PKG=$(ls $ANACONDA_PKGS | grep hdf5 | head -n 1)
HDF5_LIBRARIES=$ANACONDA_PKGS/$HDF5_PKG

JPEG_PKG=$(ls $ANACONDA_PKGS | grep jpeg | head -n 1)
JPEG_LIBRARIES=$ANACONDA_PKGS/$JPEG_PKG

echo $HDF5_LIBRARIES
echo $JPEG_LIBRARIES

PATH=$PATH:$HDF5_LIBRARIES:$JPEG_LIBRARIES

CURDIR=$(pwd)

cd $JPEGHDF5_ROOT
mkdir build
pushd build
rm -rf *
cmake ..
make
popd

# sudo is needed
HDF5_PLUGIN_DIR=/usr/local/hdf5/lib/plugin/
mkdir -p $HDF5_PLUGIN_DIR
cp build/libjpeg_h5plugin.so $HDF5_PLUGIN_DIR

cd $CURDIR
