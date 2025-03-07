# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set environment variables for better performance
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    make \
    gcc \
    g++ \
    pkg-config \
    freetype* \
    libpng-dev \
    # Added for better parallelism
    parallel \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy the local repository contents
COPY . /app/

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Set torch cache dir for esm weights
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    chmod -R 777 /root/.cache

# Create cache directories for intermediate results
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Default command to run predictions
ENTRYPOINT ["python", "entrypoint.py"]