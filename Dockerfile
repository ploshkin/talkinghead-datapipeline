FROM nvcr.io/nvidia/cuda:11.4.0-base-ubuntu20.04

RUN apt-get update \
    && apt install software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && install python3.9 \
    && apt install python3-pip

WORKDIR /dpl

RUN python3.9 -m venv --system-site-packages /dpl/dpl-venv \
    && source /dpl/dpl-venv/bin/activate

COPY . /dpl/

RUN bash /dpl/scripts/install_deps.sh
