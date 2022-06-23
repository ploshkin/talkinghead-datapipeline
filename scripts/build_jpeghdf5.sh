#! /usr/bin/bash

ANACONDA_PKGS=$1/pkgs
JPEGHDF5_ROOT=$2

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

cd $CURDIR

echo "Now move \
'${JPEGHDF5_ROOT}/build/libjpeg_h5plugin.so' to \
'/usr/local/hdf5/lib/plugin'"
