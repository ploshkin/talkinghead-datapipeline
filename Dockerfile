FROM nvcr.io/nvidia/cuda:11.4.0-base-ubuntu20.04

# Install latest libjpeg, libhdf5, CMake and Python3.9 
RUN apt-get update -y \
    && apt-get install -y \
        liblzma-dev \
        libjpeg-dev \
        libhdf5-dev \
        cmake protobuf-compiler \
        software-properties-common \
        ffmpeg \
        libsm6 \
        libxext6 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y python3.9 python3-pip>=3.9 python3.9-venv

# Create venv
ENV VIRTUAL_ENV=/opt/dpl-venv
RUN python3.9 -m pip install --upgrade pip \
    && python3.9 -m venv $VIRTUAL_ENV \
    && $VIRTUAL_ENV/bin/python3.9 -m pip install --upgrade pip
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /dpl

COPY . /dpl/

# Install dependencies
RUN bash scripts/install_deps.sh

# Install jpegHDF5 plugin
RUN bash scripts/build_jpeghdf5.sh thirdparty/jpegHDF5
