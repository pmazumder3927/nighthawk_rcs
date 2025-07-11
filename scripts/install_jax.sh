#!/bin/bash
# Install JAX with GPU support

echo "Installing JAX with GPU support..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing JAX with GPU support"
    pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "CUDA not detected, installing JAX for CPU only"
    pip install jax jaxlib
fi

# Install other dependencies
pip install -r requirements.txt

echo "JAX installation complete!"
echo "Run 'python -c \"import jax; print(jax.devices())\"' to verify installation" 