# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nighthawk RCS is a Python-based simulation environment for radar cross-section (RCS) topology optimization. It demonstrates the fundamental principles of stealth technology through geometric optimization, using physical optics approximations and modern optimization algorithms with GPU acceleration via JAX.

## Key Architecture

### Module Structure
- `src/geometry_3d.py` - 3D mesh handling with trimesh, RBF-based smooth deformations
- `src/rcs_calc_3d.py` - Physical Optics RCS calculations with JAX GPU acceleration
- `src/optimization_3d.py` - Topology optimization algorithms (gradient descent, differential evolution)
- `src/visualization_3d.py` - 3D visualization with Plotly/PyVista/Vedo backends

### Data Flow
1. Geometry creation/loading → Geometry3D object
2. RCS calculation setup → mesh invariant precomputation, JAX compilation
3. Optimization loop → deformation, RCS evaluation, gradient/population updates
4. Visualization → geometry plots, RCS patterns, evolution animations

## Development Commands

### Setup Environment
```bash
# First time setup (includes JAX GPU support detection)
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate environment
source venv/bin/activate
```

### Testing
```bash
# Test installation and GPU availability
python tests/test_installation.py

# Run optimization tests
python tests/test_optimizations.py
```

### Running Examples
```bash
# Main 3D optimization demo (GPU recommended)
python examples/rcs_3d_optimization_demo.py

# Interactive notebooks
jupyter notebook examples/jax_demo.ipynb
jupyter notebook examples/interactive_rcs_exploration.ipynb
```

### Code Quality
```bash
# Format code
black src/ examples/ tests/

# Lint code
flake8 src/ examples/ tests/
```

## Key Technical Details

### JAX GPU Integration
- Pre-compiled kernels using `jax.jit` for performance
- Batch operations with `vmap` for multiple angle evaluation
- Graceful CPU fallback when GPU unavailable
- Check GPU availability: `len(jax.devices('gpu')) > 0`

### Optimization Parameters (config.yaml)
- Frequency: 10 GHz (X-band)
- Learning rate: 0.2
- Max displacement: 2.0 meters
- Population size (DE): 20
- Mutation rate: 0.15

### Performance Benchmarks
- Single RCS: ~0.01s (GPU) vs ~0.1s (CPU)
- Full optimization (50 iterations): 5-10 minutes
- Batch speedup: 474x for multiple angles
- Memory: <2GB GPU RAM for 10k face meshes

## Important Patterns

### Mesh Deformation
Uses Radial Basis Function (RBF) interpolation for smooth, topology-preserving deformations:
```python
geometry.apply_deformation(control_points, displacements, rbf_kernel='thin_plate')
```

### RCS Calculation
Physical Optics implementation with proper illumination checking and surface current calculation:
```python
rcs = rcs_calc.calculate_rcs(mesh, theta_deg, phi_deg, polarization='VV')
```

### Optimization
Multiple algorithms available, all following common interface:
```python
optimized = optimizer.gradient_descent_3d(geometry, n_iterations=50, method='adam')
optimized = optimizer.jax_differential_evolution_3d(geometry, n_generations=100)
```

## Common Issues

### JAX Installation
- CUDA 12: `pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- CUDA 11: `pip install jax[cuda11] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- CPU only: `pip install jax jaxlib`

### Memory Issues
- Reduce mesh subdivision levels
- Decrease batch size in config.yaml
- Use `simplify()` on geometry before optimization

### Visualization
- HTML outputs saved to `visualizations/` directory
- Use `backend='plotly'` for interactive plots
- Use `backend='pyvista'` for better performance with large meshes