FROM nvidia/cuda:10.1-cudnn7-devel-centos7 as base

RUN yum install -y \
      git \
      wget \
      zsh \
    && yum clean all

# https://michaellee1.github.io/CellSegSite/cellseg_tutorial.html
# download src
WORKDIR /root/
RUN git clone https://github.com/andrewrech/CellSeg \
      && cd /root/CellSeg/src/modelFiles \
      && wget "https://get.rech.io/final_weights.h5"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       && chmod 700 Miniconda3-latest-Linux-x86_64.sh \
       && ./Miniconda3-latest-Linux-x86_64.sh -b \
       && source /root/miniconda3/etc/profile.d/conda.sh

RUN export PATH="/root/miniconda3/bin:$PATH" \
      && source /root/miniconda3/etc/profile.d/conda.sh \
      && conda create -y --name cellsegsegmenter python=3.6 \
      && conda activate cellsegsegmenter \
      && conda config --add channels conda-forge \
      && conda install -y cytoolz==0.10.0

WORKDIR /root/CellSeg
COPY requirements.txt /root/CellSeg
RUN export PATH="/root/miniconda3/bin:$PATH" \
      && source /root/miniconda3/etc/profile.d/conda.sh \
      && conda activate cellsegsegmenter \
      && pip install -r requirements.txt \
      && pip install jupyter


