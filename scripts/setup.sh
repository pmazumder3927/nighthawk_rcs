#!/bin/bash

# Setup script for RCS Topology Optimization package

echo "=========================================="
echo "RCS Topology Optimization Setup"
echo "Version 0.2.0 - 3D with JAX GPU Support"
echo "=========================================="

# Check Python version
echo -n "Checking Python version... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Python $PYTHON_VERSION found"
    
    # Check if version is 3.8+ (required for some dependencies)
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3,8) else 1)'; then
        echo "✓ Python version is compatible"
    else
        echo "⚠ Warning: Python 3.8+ is required for all features"
        exit 1
    fi
else
    echo "❌ Python 3 not found!"
    echo "Please install Python 3.8 or later"
    exit 1
fi

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
pip install --upgrade pip wheel setuptools

# Install core dependencies
echo -e "\nInstalling core dependencies..."
pip install numpy scipy matplotlib tqdm pyyaml h5py

# Install optimization libraries
echo -e "\nInstalling optimization libraries..."
pip install scikit-optimize pymoo nlopt

# Install geometry and mesh libraries
echo -e "\nInstalling geometry libraries..."
pip install trimesh meshio
pip install pygmsh  # May require gmsh system package

# Install visualization libraries
echo -e "\nInstalling visualization libraries..."
pip install plotly pyvista vedo ipywidgets

# GPU acceleration (optional)
echo -e "\n=========================================="
echo "GPU ACCELERATION SETUP"
echo "=========================================="

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name --format=csv,noheader
    
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    echo "CUDA Version: $CUDA_VERSION"
    
    echo -e "\nInstalling JAX with GPU support..."
    if [ "$CUDA_VERSION" = "12" ]; then
        pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    elif [ "$CUDA_VERSION" = "11" ]; then
        pip install jax[cuda11] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "⚠ Unsupported CUDA version. Install JAX manually:"
        echo "  pip install jax[cuda${CUDA_VERSION}] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    fi
    
    # Install Numba with CUDA support for additional kernels
    pip install numba
else
    echo "⚠ No NVIDIA GPU detected. Installing JAX for CPU only."
    pip install jax jaxlib
    echo "GPU acceleration will not be available - code will use CPU automatically."
fi

# Install remaining dependencies
echo -e "\nInstalling remaining dependencies..."
pip install pytest black flake8 imageio

# Handle potentially problematic dependencies
echo -e "\n⚠ Note about optional dependencies:"
echo "1. Mayavi: Complex to install, not required for basic functionality"
echo "2. VTK: May require system packages on some platforms"
echo "3. PyGmsh: Requires gmsh to be installed separately"

# Test imports
echo -e "\n=========================================="
echo "Testing installation..."
echo "=========================================="

python3 -c "
import numpy as np
import scipy
import matplotlib
import trimesh
import plotly
print('✓ Core dependencies OK')

try:
    import jax
    import jax.numpy as jnp
    print('✓ JAX imported successfully')
    if len(jax.devices('gpu')) > 0:
        print('✓ GPU acceleration available')
        print(f'  Available devices: {jax.devices()}')
    else:
        print('⚠ GPU acceleration not available (will use CPU)')
        print(f'  Available devices: {jax.devices()}')
except Exception as e:
    print(f'⚠ JAX not available: {e}')
    print('Will use CPU-only computation')
    
try:
    import pyvista
    print('✓ 3D visualization OK')
except:
    print('⚠ PyVista not available')
"

echo -e "\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python test_installation.py"
echo ""
echo "To run the 3D demo with JAX GPU acceleration:"
echo "  python examples/rcs_3d_optimization_demo.py"
echo ""
echo "Note: 3D optimization may take significant time depending on settings."
echo "With GPU acceleration, expect ~2-5 minutes for the demo with default settings."
echo "" 