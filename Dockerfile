# Use an official PyTorch DEVELOPMENT image with CUDA 11.8 support
# Check https://hub.docker.com/r/pytorch/pytorch/tags for available tags
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

# Set the working directory
WORKDIR /app

# Install system dependencies (add build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
COPY .env .env 
COPY train_lora_local.py train_lora_local.py
# You might want to copy other necessary files/folders here
# e.g., COPY ./scripts /app/scripts

# Install Python packages
# Pin numpy<2 but remove protobuf pin
# Pin transformers to a version compatible with trl==0.7.10
# Pin accelerate to a version compatible with transformers==4.36.2
# Add flash-attn for performance
RUN pip install --no-cache-dir \
    wandb \
    python-dotenv \
    datasets \
    accelerate==0.20.3 \
    peft==0.10.0 \
    trl==0.7.10 \
    transformers==4.36.2 \
    "numpy<2" \
    # protobuf==3.20.3 # Removed pin
    sacrebleu \
    unbabel-comet \
    mauve-text \
    flash-attn --no-build-isolation

# Make port 80 available to the world outside this container (if needed for future web interfaces)
# EXPOSE 80 

# Define the command to run your application
CMD ["python", "train_lora_local.py"] 