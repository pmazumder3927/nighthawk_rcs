"""
3D RCS Topology Optimization Demo with GPU Acceleration

This script demonstrates accurate 3D RCS optimization with:
- Physical Optics calculations
- GPU acceleration (RTX 4080 optimized)
- Surface evolution visualization
- F-117 inspired geometry optimization
- Performance optimizations for faster execution
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import psutil
import GPUtil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry_3d import (create_f117_inspired_3d, create_conventional_aircraft_3d,
                            create_simple_shape_3d, Geometry3D)
from src.rcs_calc_3d import RCS3DCalculator
from src.optimization_3d import TopologyOptimizer3D
from src.visualization_3d import RCSVisualizer3D


def print_system_info():
    """Print system information for performance monitoring."""
    print("\nSystem Information:")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                print(f"GPU: {gpu.name}, Memory: {gpu.memoryTotal:.0f} MB")
    except:
        print("GPU info not available")
    print()


def main():
    """Run the complete 3D RCS optimization demonstration with performance optimizations."""
    
    print("=" * 70)
    print("3D RCS TOPOLOGY OPTIMIZATION WITH GPU ACCELERATION")
    print("Accurate Physical Optics Implementation - Optimized Version")
    print("=" * 70)
    
    # Print system info
    print_system_info()
    
    # 1. Setup
    print("1. Setting up RCS calculator with GPU support...")
    start_setup = time.time()
    
    # X-band radar (10 GHz)
    rcs_calc = RCS3DCalculator(frequency=10e9, use_gpu=True)
    print(f"   Frequency: {rcs_calc.frequency/1e9:.1f} GHz")
    print(f"   Wavelength: {rcs_calc.wavelength:.3f} m")
    print(f"   Setup time: {time.time() - start_setup:.2f} seconds")
    
    # 2. Create initial geometries
    print("\n2. Creating 3D geometries...")
    
    # Start with a simple shape for initial testing
    print("   Creating cube (baseline)...")
    start_geom = time.time()
    cube_geometry = create_simple_shape_3d('cube', size=5.0, subdivisions=2)
    print(f"   - Vertices: {len(cube_geometry.mesh.vertices)}")
    print(f"   - Faces: {len(cube_geometry.mesh.faces)}")
    print(f"   - Volume: {cube_geometry.volume:.2f} m³")
    print(f"   - Creation time: {time.time() - start_geom:.2f} seconds")
    
    print("\n   Creating F-117 inspired geometry...")
    start_f117 = time.time()
    f117_geometry = create_f117_inspired_3d()
    print(f"   - Vertices: {len(f117_geometry.mesh.vertices)}")
    print(f"   - Faces: {len(f117_geometry.mesh.faces)}")
    print(f"   - Volume: {f117_geometry.volume:.2f} m³")
    print(f"   - Creation time: {time.time() - start_f117:.2f} seconds")
    
    # 3. Visualize initial geometries
    print("\n3. Visualizing initial geometries...")
    visualizer = RCSVisualizer3D(backend='plotly')
    
    # Plot geometries
    fig_cube = visualizer.plot_geometry_3d(cube_geometry, 
                                          show_normals=False, 
                                          title="Cube Geometry")
    fig_cube.write_html('visualizations/cube_geometry.html')
    
    fig_f117 = visualizer.plot_geometry_3d(f117_geometry, 
                                         show_normals=True,
                                         title="F-117 Inspired Geometry")
    fig_f117.write_html('visualizations/f117_geometry.html')
    
    # 4. Calculate baseline RCS with performance timing
    print("\n4. Calculating baseline RCS values...")
    
    # Precompute mesh invariants for faster calculations
    print("   Precomputing mesh invariants...")
    rcs_calc._precompute_mesh_invariants(cube_geometry.mesh)
    
    # Calculate RCS at key angles
    test_angles = [(90, 0), (90, 90), (90, 180), (90, 270)]  # Front, side, back, side
    
    print("\n   Cube RCS (with timing):")
    for theta, phi in test_angles:
        start_rcs = time.time()
        rcs = rcs_calc.calculate_rcs(cube_geometry.mesh, theta, phi)
        calc_time = time.time() - start_rcs
        rcs_db = 10 * np.log10(rcs + 1e-10)
        print(f"   θ={theta}°, φ={phi}°: {rcs_db:.1f} dBsm (time: {calc_time*1000:.1f} ms)")
    
    # Batch calculation demonstration
    print("\n   Testing batch RCS calculation...")
    theta_batch = np.array([angle[0] for angle in test_angles])
    phi_batch = np.array([angle[1] for angle in test_angles])
    
    start_batch = time.time()
    rcs_batch = rcs_calc.calculate_rcs_batch(cube_geometry.mesh, theta_batch, phi_batch)
    batch_time = time.time() - start_batch
    print(f"   Batch calculation time: {batch_time*1000:.1f} ms for {len(test_angles)} angles")
    print(f"   Speedup: {len(test_angles) / (batch_time / calc_time):.1f}x")
        
    print("\n   F-117 RCS:")
    rcs_calc._precompute_mesh_invariants(f117_geometry.mesh)
    for theta, phi in test_angles:
        rcs = rcs_calc.calculate_rcs(f117_geometry.mesh, theta, phi)
        rcs_db = 10 * np.log10(rcs + 1e-10)
        print(f"   θ={theta}°, φ={phi}°: {rcs_db:.1f} dBsm")
        
    # 5. Optimize cube geometry with enhanced settings
    print("\n5. Optimizing cube geometry for RCS reduction...")
    print("   Using enhanced optimization with:")
    print("   - Parallel gradient computation")
    print("   - Adaptive learning rate")
    print("   - Early stopping")
    print("   - Farthest point sampling for control points")
    
    # Define target angles (frontal sector emphasis) - reduced for speed
    target_angles = []
    for theta in [60, 90, 120]:
        for phi in [0, 30, 330]:
            target_angles.append((theta, phi))
            
    print(f"   Target angles: {len(target_angles)} directions in frontal sector")
    
    # Create optimizer
    optimizer = TopologyOptimizer3D(
        rcs_calc,
        control_points=None,  # Will use FPS
        max_displacement=1.0,  # Reduced for stability
        volume_constraint=True,
        smoothness=0.5
    )
    
    # Run optimization with performance monitoring
    print("\n   Running gradient-based optimization (Adam) with enhancements...")
    print("   Monitoring performance metrics...")
    
    # Monitor initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    
    optimized_cube = optimizer.gradient_descent_3d(
        cube_geometry,
        n_iterations=50,  # Reduced for demo, but more effective iterations
        learning_rate=0.1,
        target_angles=target_angles,
        method='adam',
        adaptive_lr=True,
        checkpoint_interval=5
    )
    
    elapsed_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\n   Optimization completed in {elapsed_time:.1f} seconds")
    print(f"   Memory usage: {final_memory - initial_memory:.1f} MB")
    print(f"   Average time per iteration: {elapsed_time / optimizer.history['iterations']:.2f} seconds")
    
    # 6. Analyze results
    print("\n6. Analyzing optimization results...")
    
    # Calculate RCS improvement using batch evaluation
    print("\n   RCS comparison at key angles:")
    print("   " + "-" * 50)
    print("   Angle         | Original | Optimized | Reduction")
    print("   " + "-" * 50)
    
    # Batch calculate for efficiency
    theta_test = np.array([angle[0] for angle in test_angles])
    phi_test = np.array([angle[1] for angle in test_angles])
    
    rcs_orig_batch = rcs_calc.calculate_rcs_batch(cube_geometry.mesh, theta_test, phi_test)
    rcs_opt_batch = rcs_calc.calculate_rcs_batch(optimized_cube.mesh, theta_test, phi_test)
    
    for i, (theta, phi) in enumerate(test_angles):
        rcs_orig_db = 10 * np.log10(rcs_orig_batch[i] + 1e-10)
        rcs_opt_db = 10 * np.log10(rcs_opt_batch[i] + 1e-10)
        reduction_db = rcs_orig_db - rcs_opt_db
        
        print(f"   θ={theta}°,φ={phi:3d}° | {rcs_orig_db:7.1f} | {rcs_opt_db:8.1f} | {reduction_db:8.1f} dB")
        
    # Volume preservation check
    vol_ratio = optimized_cube.volume / cube_geometry.volume
    print(f"\n   Volume preservation: {vol_ratio:.3f} (target: 1.0)")
    
    # 7. Create surface evolution animation (only key frames)
    print("\n7. Creating surface evolution animation...")
    
    # Use only checkpointed geometries for faster animation
    fig_anim = visualizer.create_surface_evolution_animation(
        optimizer.history,
        save_path='visualizations/surface_evolution.html',
        skip_frames=1  # Use all checkpointed frames
    )
    
    print("   Animation saved to: visualizations/surface_evolution.html")
    
    # 8. Compare with F-117
    print("\n8. Creating comparison visualization...")
    
    fig_comparison = visualizer.plot_optimization_comparison_3d(
        cube_geometry,
        optimized_cube,
        rcs_calc
    )
    fig_comparison.write_html('visualizations/optimization_comparison.html')
    
    # 9. Plot RCS patterns (reduced resolution for speed)
    print("\n9. Calculating RCS patterns (reduced resolution for speed)...")
    
    # Calculate RCS pattern for optimized geometry
    print("   Calculating optimized geometry RCS pattern...")
    start_pattern = time.time()
    
    theta_grid, phi_grid, rcs_opt_db = rcs_calc.calculate_rcs_pattern(
        optimized_cube.mesh,
        theta_range=(30, 150),
        phi_range=(0, 360),
        n_theta=13,  # Further reduced for speed
        n_phi=25
    )
    
    pattern_time = time.time() - start_pattern
    print(f"   Pattern calculation time: {pattern_time:.1f} seconds")
    print(f"   Points per second: {13*25/pattern_time:.0f}")
    
    # Plot RCS sphere
    fig_rcs = visualizer.plot_rcs_sphere(theta_grid, phi_grid, rcs_opt_db,
                                       title="Optimized Geometry RCS Pattern")
    fig_rcs.write_html('visualizations/rcs_pattern_3d.html')
    
    # 10. Plot RCS cuts
    print("\n10. Creating RCS cut plots...")
    
    # Azimuth cut at θ=90°
    fig_azimuth = visualizer.plot_rcs_cuts(
        optimized_cube,
        rcs_calc,
        cut_type='azimuth',
        fixed_angle=90.0
    )
    plt.savefig('visualizations/rcs_azimuth_cut.png', dpi=150, bbox_inches='tight')
    
    # Elevation cut at φ=0°
    fig_elevation = visualizer.plot_rcs_cuts(
        optimized_cube,
        rcs_calc,
        cut_type='elevation',
        fixed_angle=0.0
    )
    plt.savefig('visualizations/rcs_elevation_cut.png', dpi=150, bbox_inches='tight')
    
    # Summary with performance metrics
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
    
    print("\nPerformance Summary:")
    print(f"Total execution time: {time.time() - start_time:.1f} seconds")
    print(f"Optimization time: {elapsed_time:.1f} seconds")
    print(f"RCS calculations per second: {len(optimizer.history['objective_values']) * len(target_angles) / elapsed_time:.0f}")
    
    print("\nKey optimizations applied:")
    print("1. Batch RCS calculations for multiple angles")
    print("2. Parallel gradient computation using ThreadPoolExecutor")
    print("3. Adaptive learning rate with cosine annealing")
    print("4. Early stopping to avoid unnecessary iterations")
    print("5. Farthest point sampling for better control points")
    print("6. Caching of objective function evaluations")
    print("7. Reduced mesh invariant recomputation")
    print("8. Optimized memory usage with checkpointing")
    
    print("\nVisualization files created in 'visualizations/' directory:")
    print("- cube_geometry.html: Initial cube geometry")
    print("- f117_geometry.html: F-117 inspired geometry")
    print("- surface_evolution.html: Animated optimization progress")
    print("- optimization_comparison.html: Before/after comparison")
    print("- rcs_pattern_3d.html: 3D RCS pattern visualization")
    print("- rcs_azimuth_cut.png: Azimuth RCS pattern")
    print("- rcs_elevation_cut.png: Elevation RCS pattern")
    
    # Optional: Save optimized geometry
    optimized_cube.export('visualizations/optimized_cube.stl')
    print("\nOptimized geometry saved as: optimized_cube.stl")
    
    # Clear GPU memory if used
    if rcs_calc.use_gpu:
        rcs_calc.clear_cache()
        print("\nGPU memory cleared")
    
    print("\n✅ 3D RCS optimization demonstration complete!")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Run the demo
    main() 