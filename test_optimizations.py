#!/usr/bin/env python3
"""
Simple test script to verify RCS optimization improvements.
"""

import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from geometry_3d import create_simple_shape_3d
from rcs_calc_3d import RCS3DCalculator
from optimization_3d import TopologyOptimizer3D

def test_rcs_calculation():
    """Test RCS calculation performance."""
    print("Testing RCS calculation performance...")
    
    # Create simple geometry
    geometry = create_simple_shape_3d('cube', size=2.0, subdivisions=1)
    print(f"Created geometry with {len(geometry.mesh.vertices)} vertices, {len(geometry.mesh.faces)} faces")
    
    # Initialize calculator
    rcs_calc = RCS3DCalculator(frequency=10e9, use_gpu=True)
    
    # Test single RCS calculation
    start_time = time.time()
    rcs = rcs_calc.calculate_rcs(geometry.mesh, 90, 0)
    single_time = time.time() - start_time
    print(f"Single RCS calculation: {rcs:.6f} m² in {single_time:.4f}s")
    
    # Test batch RCS calculation
    angles = [(90, phi) for phi in np.linspace(0, 360, 36, endpoint=False)]
    theta_angles = np.array([a[0] for a in angles])
    phi_angles = np.array([a[1] for a in angles])
    
    start_time = time.time()
    rcs_batch = rcs_calc.calculate_rcs_batch(geometry.mesh, theta_angles, phi_angles)
    batch_time = time.time() - start_time
    print(f"Batch RCS calculation ({len(angles)} angles): {batch_time:.4f}s")
    print(f"Speedup: {len(angles) * single_time / batch_time:.1f}x")
    
    return rcs_calc, geometry

def test_optimization():
    """Test optimization performance."""
    print("\nTesting optimization performance...")
    
    rcs_calc, geometry = test_rcs_calculation()
    
    # Create optimizer
    optimizer = TopologyOptimizer3D(
        rcs_calc,
        control_points=geometry.mesh.vertices,  # Use all vertices as control points
        max_displacement=0.5,
        volume_constraint=False,
        smoothness=0.5
    )
    
    # Test objective function
    target_angles = [(90, 0), (90, 90), (90, 180), (90, 270)]
    
    start_time = time.time()
    obj_value = optimizer.objective_function(geometry, target_angles)
    obj_time = time.time() - start_time
    print(f"Objective evaluation: {obj_value:.6f} in {obj_time:.4f}s")
    
    # Test gradient calculation
    start_time = time.time()
    gradient = optimizer._calculate_gradient_finite_diff(geometry, target_angles, epsilon=0.01)
    grad_time = time.time() - start_time
    print(f"Gradient calculation: {grad_time:.4f}s")
    print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")
    
    return optimizer, geometry

def main():
    """Run all tests."""
    print("=" * 60)
    print("RCS OPTIMIZATION PERFORMANCE TESTS")
    print("=" * 60)
    
    try:
        test_rcs_calculation()
        test_optimization()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 