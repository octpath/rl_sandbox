FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt install -y wget git vim
RUN apt install -y libgl1-mesa-dev libglib2.0-0

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh
RUN bash Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -b -p /root/miniconda3
ENV PATH /root/miniconda3/bin:$PATH

RUN conda install -y gh --channel conda-forge

RUN pip install -U pip
RUN pip install numpy polars pandas scikit-learn jupyterlab loguru omegaconf opencv-python matplotlib

RUN pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install tensorflow==2.15.0.post1
RUN pip install tf-keras
RUN pip install -U keras-cv
RUN pip install -U keras-nlp
RUN pip install -U keras

RUN conda init
