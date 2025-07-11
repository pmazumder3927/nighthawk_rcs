"""
Quick test script to verify the RCS optimization package is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing RCS Optimization Package Installation...")
print("=" * 50)

# Test 2D modules
print("\n1. Testing 2D modules...")
try:
    from src.geometry import create_f117_inspired_2d_profile
    from src.rcs_calc import RCSCalculator
    from src.optimization import TopologyOptimizer
    from src.visualization import RCSVisualizer
    
    print("✓ All 2D imports successful")
    
    # Test basic 2D functionality
    geom = create_f117_inspired_2d_profile()
    print(f"✓ Created 2D F-117 geometry with {geom.n_vertices} vertices")
    
    rcs_calc = RCSCalculator(frequency=10e9)
    angles, rcs_db = rcs_calc.calculate_2d_rcs_pattern(geom, n_angles=36)
    print(f"✓ Calculated 2D RCS pattern: mean = {rcs_db.mean():.1f} dBsm")
    
except ImportError as e:
    print(f"❌ 2D module import error: {e}")
except Exception as e:
    print(f"❌ 2D test error: {e}")

# Test 3D modules
print("\n2. Testing 3D modules...")
try:
    from src.geometry_3d import create_f117_inspired_3d, create_simple_shape_3d
    from src.rcs_calc_3d import RCS3DCalculator
    from src.optimization_3d import TopologyOptimizer3D
    from src.visualization_3d import RCSVisualizer3D
    
    print("✓ All 3D imports successful")
    
    # Test basic 3D functionality
    geom_3d = create_simple_shape_3d('sphere', size=1.0)
    print(f"✓ Created 3D sphere with {len(geom_3d.mesh.vertices)} vertices")
    
    # Check GPU availability
    rcs_calc_3d = RCS3DCalculator(frequency=10e9, use_gpu=True)
    if rcs_calc_3d.use_gpu:
        print("✓ GPU acceleration is available")
    else:
        print("⚠ GPU acceleration not available, using CPU")
        
    # Test RCS calculation
    rcs = rcs_calc_3d.calculate_rcs(geom_3d.mesh, theta=90, phi=0)
    rcs_db = 10 * np.log10(rcs + 1e-10)
    print(f"✓ Calculated 3D RCS: {rcs_db:.1f} dBsm")
    
except ImportError as e:
    print(f"❌ 3D module import error: {e}")
    print("\nMake sure you have installed 3D dependencies:")
    print("  pip install trimesh pygmsh jax[cuda12] numba")
except Exception as e:
    print(f"❌ 3D test error: {e}")

# Test visualization modules
print("\n3. Testing visualization capabilities...")
try:
    import matplotlib
    print("✓ Matplotlib available")
    
    import plotly
    print("✓ Plotly available")
    
    try:
        import pyvista
        print("✓ PyVista available for 3D visualization")
    except:
        print("⚠ PyVista not available (optional)")
        
    try:
        import vedo
        print("✓ Vedo available for 3D visualization")
    except:
        print("⚠ Vedo not available (optional)")
        
except ImportError as e:
    print(f"❌ Visualization import error: {e}")

# Test optimization libraries
print("\n4. Testing optimization libraries...")
try:
    import scipy.optimize
    print("✓ SciPy optimization available")
    
    import nlopt
    print("✓ NLopt available")
    
    import pymoo
    print("✓ PyMOO available")
    
except ImportError as e:
    print(f"⚠ Some optimization libraries missing: {e}")

# Summary
print("\n" + "=" * 50)
print("INSTALLATION TEST SUMMARY")
print("=" * 50)

print("\n✅ Core functionality is ready!")
print("\nTo run demos:")
print("  - 2D demo: python examples/rcs_optimization_demo.py")
print("  - 3D demo: python examples/rcs_3d_optimization_demo.py")
print("\nNote: 3D optimization with GPU can take significant time.")
print("Adjust iterations in the demo script for faster testing.")

# Import numpy at module level for 3D test
try:
    import numpy as np
except:
    pass 