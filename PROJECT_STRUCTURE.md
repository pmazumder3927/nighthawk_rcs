# Nighthawk RCS Project Structure

## Core Directories

### `/src/` - Source Code
- `geometry_3d.py` - 3D mesh handling with refinement capabilities
- `rcs_calc_3d.py` - Physical Optics RCS calculations with JAX
- `optimization_3d.py` - Topology optimization algorithms
- `visualization_3d.py` - 3D visualization utilities

### `/examples/` - Demonstration Scripts
- `rcs_3d_optimization_demo.py` - Main 3D optimization demo
- `mesh_refinement_demo.py` - Shows mesh resolution impact
- `echo1_manual_optimization.py` - 1970s manual optimization
- `modern_automated_optimization.py` - Modern automated optimization
- `shared_aircraft_geometry.py` - Common geometry for optimization demos
- `interactive_rcs_exploration.ipynb` - Interactive Jupyter notebook
- `jax_demo.ipynb` - JAX features demonstration

### `/tests/` - Test Suite
- `test_installation.py` - Installation verification
- `test_optimizations.py` - Optimization algorithm tests
- `test_sphere_rcs_validation.py` - RCS validation tests

### `/visualizations/` - Output Directory
Generated visualizations are saved here when running demos:
- PNG files for static plots
- HTML files for interactive 3D visualizations
- No STL files (removed to reduce clutter)

### `/docs/` - Documentation
- `README.md` - Main documentation

### `/scripts/` - Setup Scripts
- `setup.sh` - Environment setup
- `install_jax.sh` - JAX installation helper

## Configuration Files
- `config.yaml` - Optimization parameters
- `requirements.txt` - Python dependencies
- `CLAUDE.md` - AI assistant guidance
- `MESH_REFINEMENT_FEATURES.md` - Mesh refinement documentation
- `IMPLEMENTATION_SUMMARY.md` - Recent implementation summary

## Key Features Implemented

1. **Mesh Refinement**
   - Automatic quality warnings for electrically large objects
   - Subdivision and remeshing capabilities
   - Stationary phase approximation for very large objects

2. **Optimization Demos**
   - Manual (ECHO-1 style) vs automated comparison
   - Both start with same baseline geometry
   - Clear demonstration of computational advantages

3. **Clean Codebase**
   - No redundant files
   - No STL exports
   - Organized visualization outputs
   - Clear separation of concerns

## Running Demos

```bash
# Mesh refinement demonstration
python examples/mesh_refinement_demo.py

# Manual optimization (1970s style)
python examples/echo1_manual_optimization.py

# Modern automated optimization
python examples/modern_automated_optimization.py

# Main 3D optimization demo
python examples/rcs_3d_optimization_demo.py
```

All visualizations are saved to the `/visualizations/` directory.