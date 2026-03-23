FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install nerfacc from python wheel
RUN pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html

# Source code, datasets and outputs are bind-mounted at runtime via docker-compose
VOLUME ["/workspace/src"]
VOLUME [ "/workspace/datasets" ]
VOLUME [ "/workspace/output" ]

WORKDIR /workspace/src

CMD ["bash"]