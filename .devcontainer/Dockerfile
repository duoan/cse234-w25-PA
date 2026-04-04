FROM mcr.microsoft.com/devcontainers/miniconda:0-3.11

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and related packages
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia && \
    conda clean -afy

# Install common development packages
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipython \
    numpy \
    scipy \
    pandas \
    matplotlib \
    scikit-learn \
    black \
    pylint \
    pytest

WORKDIR /workspaces
