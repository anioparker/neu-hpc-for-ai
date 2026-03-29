FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    ninja-build \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy dependency metadata first for better caching
COPY requirements.txt pyproject.toml setup.py /workspace/

# Install PyTorch first
# If cu124 wheels are unavailable for your chosen torch version, switch to cu121.
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies if present
RUN if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt; fi

# Copy the rest of the project
COPY . /workspace

# Install your package
RUN python3 -m pip install -e .

ENTRYPOINT ["/bin/bash"]