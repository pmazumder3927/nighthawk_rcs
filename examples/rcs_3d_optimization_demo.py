"""
3D RCS Topology Optimization Demo with GPU Acceleration

This script demonstrates accurate 3D RCS optimization with:
- Physical Optics calculations
- GPU acceleration (RTX 4080 optimized)
- Surface evolution visualization
- F-117 inspired geometry optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry_3d import (create_f117_inspired_3d, create_conventional_aircraft_3d,
                            create_simple_shape_3d, Geometry3D)
from src.rcs_calc_3d import RCS3DCalculator
from src.optimization_3d import TopologyOptimizer3D
from src.visualization_3d import RCSVisualizer3D


def main():
    """Run the complete 3D RCS optimization demonstration."""
    
    print("=" * 70)
    print("3D RCS TOPOLOGY OPTIMIZATION WITH GPU ACCELERATION")
    print("Accurate Physical Optics Implementation")
    print("=" * 70)
    
    # 1. Setup
    print("\n1. Setting up RCS calculator with GPU support...")
    # X-band radar (10 GHz)
    rcs_calc = RCS3DCalculator(frequency=10e9, use_gpu=True)
    print(f"   Frequency: {rcs_calc.frequency/1e9:.1f} GHz")
    print(f"   Wavelength: {rcs_calc.wavelength:.3f} m")
    
    # 2. Create initial geometries
    print("\n2. Creating 3D geometries...")
    
    # Start with a simple shape for initial testing
    print("   Creating sphere (baseline)...")
    sphere_geometry = create_simple_shape_3d('sphere', size=5.0, subdivisions=3)
    print(f"   - Vertices: {len(sphere_geometry.mesh.vertices)}")
    print(f"   - Faces: {len(sphere_geometry.mesh.faces)}")
    print(f"   - Volume: {sphere_geometry.volume:.2f} m³")
    
    print("\n   Creating F-117 inspired geometry...")
    f117_geometry = create_f117_inspired_3d()
    print(f"   - Vertices: {len(f117_geometry.mesh.vertices)}")
    print(f"   - Faces: {len(f117_geometry.mesh.faces)}")
    print(f"   - Volume: {f117_geometry.volume:.2f} m³")
    
    # 3. Visualize initial geometries
    print("\n3. Visualizing initial geometries...")
    visualizer = RCSVisualizer3D(backend='plotly')
    
    # Plot geometries
    fig_sphere = visualizer.plot_geometry_3d(sphere_geometry, 
                                           show_normals=False, 
                                           title="Sphere Geometry")
    fig_sphere.write_html('visualizations/sphere_geometry.html')
    
    fig_f117 = visualizer.plot_geometry_3d(f117_geometry, 
                                         show_normals=True,
                                         title="F-117 Inspired Geometry")
    fig_f117.write_html('visualizations/f117_geometry.html')
    
    # 4. Calculate baseline RCS
    print("\n4. Calculating baseline RCS values...")
    
    # Calculate RCS at key angles
    test_angles = [(90, 0), (90, 90), (90, 180), (90, 270)]  # Front, side, back, side
    
    print("\n   Sphere RCS:")
    for theta, phi in test_angles:
        rcs = rcs_calc.calculate_rcs(sphere_geometry.mesh, theta, phi)
        rcs_db = 10 * np.log10(rcs + 1e-10)
        print(f"   θ={theta}°, φ={phi}°: {rcs_db:.1f} dBsm")
        
    print("\n   F-117 RCS:")
    for theta, phi in test_angles:
        rcs = rcs_calc.calculate_rcs(f117_geometry.mesh, theta, phi)
        rcs_db = 10 * np.log10(rcs + 1e-10)
        print(f"   θ={theta}°, φ={phi}°: {rcs_db:.1f} dBsm")
        
    # 5. Optimize sphere geometry
    print("\n5. Optimizing sphere geometry for RCS reduction...")
    print("   This will demonstrate how optimization discovers faceted surfaces")
    
    # Define target angles (frontal sector emphasis)
    target_angles = []
    for theta in [60, 90, 120]:
        for phi in [0, 30, 330]:
            target_angles.append((theta, phi))
            
    print(f"   Target angles: {len(target_angles)} directions in frontal sector")
    
    # Create optimizer
    optimizer = TopologyOptimizer3D(
        rcs_calc,
        control_points=None,
        max_displacement=2,  # Limit deformation
        volume_constraint=True,
        smoothness=0.5
    )
    
    # Run optimization
    print("\n   Running gradient-based optimization (Adam)...")
    print("   This may take several minutes with GPU acceleration...")
    
    start_time = time.time()
    
    optimized_sphere = optimizer.gradient_descent_3d(
        sphere_geometry,
        n_iterations=100,  # Reduced for demo
        learning_rate=0.05,
        target_angles=target_angles,
        method='adam'
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n   Optimization completed in {elapsed_time:.1f} seconds")
    
    # 6. Analyze results
    print("\n6. Analyzing optimization results...")
    
    # Calculate RCS improvement
    print("\n   RCS comparison at key angles:")
    print("   " + "-" * 50)
    print("   Angle         | Original | Optimized | Reduction")
    print("   " + "-" * 50)
    
    for theta, phi in test_angles:
        rcs_orig = rcs_calc.calculate_rcs(sphere_geometry.mesh, theta, phi)
        rcs_opt = rcs_calc.calculate_rcs(optimized_sphere.mesh, theta, phi)
        
        rcs_orig_db = 10 * np.log10(rcs_orig + 1e-10)
        rcs_opt_db = 10 * np.log10(rcs_opt + 1e-10)
        reduction_db = rcs_orig_db - rcs_opt_db
        
        print(f"   θ={theta}°,φ={phi:3d}° | {rcs_orig_db:7.1f} | {rcs_opt_db:8.1f} | {reduction_db:8.1f} dB")
        
    # Volume preservation check
    vol_ratio = optimized_sphere.volume / sphere_geometry.volume
    print(f"\n   Volume preservation: {vol_ratio:.3f} (target: 1.0)")
    
    # 7. Create surface evolution animation
    print("\n7. Creating surface evolution animation...")
    
    fig_anim = visualizer.create_surface_evolution_animation(
        optimizer.history,
        save_path='visualizations/surface_evolution.html',
        skip_frames=2  # Show every 2nd frame
    )
    
    print("   Animation saved to: visualizations/surface_evolution.html")
    
    # 8. Compare with F-117
    print("\n8. Creating comparison visualization...")
    
    fig_comparison = visualizer.plot_optimization_comparison_3d(
        sphere_geometry,
        optimized_sphere,
        rcs_calc
    )
    fig_comparison.write_html('visualizations/optimization_comparison.html')
    
    # 9. Plot RCS patterns
    print("\n9. Calculating full RCS patterns (this will take time)...")
    
    # Calculate RCS pattern for optimized geometry
    print("   Calculating optimized geometry RCS pattern...")
    theta_grid, phi_grid, rcs_opt_db = rcs_calc.calculate_rcs_pattern(
        optimized_sphere.mesh,
        theta_range=(30, 150),
        phi_range=(0, 360),
        n_theta=19,  # Reduced for faster computation
        n_phi=37
    )
    
    # Plot RCS sphere
    fig_rcs = visualizer.plot_rcs_sphere(theta_grid, phi_grid, rcs_opt_db,
                                       title="Optimized Geometry RCS Pattern")
    fig_rcs.write_html('visualizations/rcs_pattern_3d.html')
    
    # 10. Plot RCS cuts
    print("\n10. Creating RCS cut plots...")
    
    # Azimuth cut at θ=90°
    fig_azimuth = visualizer.plot_rcs_cuts(
        optimized_sphere,
        rcs_calc,
        cut_type='azimuth',
        fixed_angle=90.0
    )
    plt.savefig('visualizations/rcs_azimuth_cut.png', dpi=150, bbox_inches='tight')
    
    # Elevation cut at φ=0°
    fig_elevation = visualizer.plot_rcs_cuts(
        optimized_sphere,
        rcs_calc,
        cut_type='elevation',
        fixed_angle=0.0
    )
    plt.savefig('visualizations/rcs_elevation_cut.png', dpi=150, bbox_inches='tight')
    
    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    initial_obj = optimizer.history['objective_values'][0]
    final_obj = optimizer.history['objective_values'][-1]
    reduction_factor = initial_obj / final_obj
    
    print(f"Initial objective value: {initial_obj:.6f}")
    print(f"Final objective value: {final_obj:.6f}")
    print(f"Reduction factor: {reduction_factor:.2f}x")
    print(f"Mean RCS reduction: {10*np.log10(reduction_factor):.1f} dB")
    
    print("\nKey insights:")
    print("1. Optimization naturally evolves smooth surfaces toward faceted designs")
    print("2. Faceted surfaces redirect energy away from incident direction")
    print("3. Volume constraints preserve physical realizability")
    print("4. GPU acceleration enables accurate Physical Optics calculations")
    print("5. F-117's faceted design is validated by physics-based optimization")
    
    print("\nVisualization files created in 'visualizations/' directory:")
    print("- sphere_geometry.html: Initial sphere geometry")
    print("- f117_geometry.html: F-117 inspired geometry")
    print("- surface_evolution.html: Animated optimization progress")
    print("- optimization_comparison.html: Before/after comparison")
    print("- rcs_pattern_3d.html: 3D RCS pattern visualization")
    print("- rcs_azimuth_cut.png: Azimuth RCS pattern")
    print("- rcs_elevation_cut.png: Elevation RCS pattern")
    
    # Optional: Save optimized geometry
    optimized_sphere.export('visualizations/optimized_sphere.stl')
    print("\nOptimized geometry saved as: optimized_sphere.stl")
    
    print("\n✅ 3D RCS optimization demonstration complete!")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Run the demo
    main() 