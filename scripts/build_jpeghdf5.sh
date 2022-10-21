#! /usr/bin/bash

JPEGHDF5_ROOT=$1

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
