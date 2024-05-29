FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN conda create -n videomae python=3.8 -y && \
    conda install -n videomae pip -y && \
    conda clean -a -y

SHELL ["conda", "run", "-n", "videomae", "/bin/bash", "-c"]

RUN pip install transformers[torch] \
    accelerate==0.27.2 \
    datasets \
    opencv-python \
    imageio[ffmpeg]

COPY . /workspace
WORKDIR /workspace

CMD ["bash"]
