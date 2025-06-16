FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ARG libdir
ARG nproc=8

# Fix GPG keys for NVIDIA CUDA package repositories
RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys  \
      http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys  \
      https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Experimentally determined dependencies
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        cmake \
        libglm-dev \
        libglu1-mesa-dev \
        libegl1-mesa-dev \
        libglib2.0-0 \
        libopenmpi2 \
        libsm6 \
        libxrender1 \
        libxext6 \
        build-essential \
        g++ \
        git \
        gosu \
        sudo \
        # for getting Miniconda
        wget

# install conda
RUN mkdir /src/ && \
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /src/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /src/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    rm -r /src

# Add cifs-utils for Alexnet mount
# Mount alexnet synthetic-datagen output as writeable
#    Use noperm to give docker user unlimited permissions
RUN apt update && apt install -y cifs-utils && \
    mkdir -p /alexnet/synthetic-datagen/outputs && \
    echo "//alexnet/VCVPublic/ /alexnet/ cifs ro,nounix,user=,password=,noperm 0 0" >> /etc/fstab && \
    echo "//alexnet/VCVPublic/users/aleksey/motion-planning /alexnet/users/aleksey/motion-planning/outputs cifs rw,nounix,user=,password=,noperm 0 0" >> /etc/fstab

COPY $libdir/requirements.tar /src/motion-planning/requirements.tar
RUN bash -c "cd /src/motion-planning && \
        tar -xf requirements.tar && \
        source /etc/profile.d/conda.sh && \
        conda create -n mp python=3.10 && \
        conda activate mp && \
        scripts/setup.sh -f && \
        rm -fr requirements.tar /root/.cache"

COPY $libdir/code.tar /src/motion-planning/code.tar

RUN cd /src/motion-planning && \
        tar xf code.tar && \
        rm code.tar

RUN chmod -R a+rw /src/motion-planning

COPY entrypoint.sh /src/entrypoint.sh
RUN chmod +x /src/entrypoint.sh

WORKDIR /src/motion-planning
ENTRYPOINT ["/src/entrypoint.sh"]
