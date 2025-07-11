# Nighthawk RCS Topology Optimization Sandbox

A Python-based simulation environment for exploring radar cross-section (RCS) topology optimization techniques inspired by the F-117 Nighthawk stealth aircraft development.

## Overview

This project demonstrates the fundamental principles of RCS reduction through geometric optimization, using simplified physical optics approximations and modern optimization algorithms.

## Features

- **RCS Calculation**: Physical optics-based RCS computation for arbitrary geometries
- **3D Optimization**: Advanced 3D topology optimization with GPU acceleration
- **JAX Integration**: GPU-accelerated computations with automatic differentiation
- **Topology Optimization**: Gradient-based and evolutionary optimization algorithms
- **Visualization**: Real-time 3D geometry and RCS pattern visualization
- **F-117 Inspired**: Initial geometries and constraints based on stealth aircraft design principles

## Installation

### Option 1: Automatic Setup (Recommended)

```bash
# Clone the repository
cd nighthawk_rcs

# Run automated setup with JAX GPU support
./setup.sh
```

### Option 2: Manual Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install JAX with GPU support
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test JAX integration
python test_jax_integration.py

# Test overall installation
python test_installation.py
```

## Quick Start

### 3D Optimization with JAX (Recommended)

```python
from src.rcs_calc_3d import RCS3DCalculator
from src.optimization_3d import TopologyOptimizer3D
from src.geometry_3d import Geometry3D
import trimesh

# Create initial geometry
mesh = trimesh.creation.box(extents=[2, 2, 1])
geometry = Geometry3D(mesh)

# Initialize RCS calculator with GPU acceleration
rcs_calc = RCS3DCalculator(frequency=10e9, use_gpu=True)

# Initialize optimizer
optimizer = TopologyOptimizer3D(rcs_calc)

# Run optimization
optimized_geometry = optimizer.gradient_descent_3d(
    geometry, n_iterations=50, method='adam'
)

# Visualize results
optimizer.visualize_results()
```

### 2D Optimization (Legacy)

```python
from src.rcs_calc import RCSCalculator
from src.optimization import TopologyOptimizer
from src.geometry import create_f117_inspired_2d_profile

# Create initial geometry
geometry = create_f117_inspired_2d_profile()

# Initialize optimizer
optimizer = TopologyOptimizer(geometry)

# Run optimization
optimized_geometry = optimizer.optimize(iterations=100)
```

### Run Examples

```bash
# 3D optimization demo
python examples/rcs_3d_optimization_demo.py

# JAX integration demo
jupyter notebook jax_demo.ipynb
```

## Project Structure

```
nighthawk_rcs/
├── src/                   # Core source code
│   ├── rcs_calc.py       # RCS calculation module
│   ├── geometry.py       # Geometry representation
│   ├── optimization.py   # Optimization algorithms
│   └── visualization.py  # Visualization tools
├── examples/             # Example scripts and notebooks
├── tests/               # Unit tests
├── docs/                # Additional documentation
└── visualizations/      # Output visualizations
```

## Physics Background

The RCS calculation uses a simplified physical optics (PO) approximation, which is suitable for electrically large objects and provides reasonable accuracy for preliminary design studies.

## License

MIT License - See LICENSE file for details
