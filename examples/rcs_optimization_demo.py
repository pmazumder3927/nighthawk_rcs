"""
RCS Topology Optimization Demo

This script demonstrates the complete RCS optimization process
inspired by F-117 Nighthawk stealth design principles.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import create_f117_inspired_2d_profile, create_conventional_aircraft_2d_profile
from src.rcs_calc import RCSCalculator, compute_rcs_statistics
from src.optimization import TopologyOptimizer
from src.visualization import RCSVisualizer


def main():
    """Run the complete RCS optimization demonstration."""
    
    print("=" * 60)
    print("RCS TOPOLOGY OPTIMIZATION DEMO")
    print("Inspired by F-117 Nighthawk Design Principles")
    print("=" * 60)
    
    # 1. Setup
    print("\n1. Setting up RCS calculator...")
    # X-band radar (10 GHz) - commonly used for military applications
    rcs_calc = RCSCalculator(frequency=10e9)
    print(f"   Frequency: {rcs_calc.frequency/1e9:.1f} GHz")
    print(f"   Wavelength: {rcs_calc.wavelength:.3f} m")
    
    # 2. Create initial geometries
    print("\n2. Creating initial geometries...")
    f117_geometry = create_f117_inspired_2d_profile()
    conventional_geometry = create_conventional_aircraft_2d_profile()
    
    # 3. Compare initial RCS patterns
    print("\n3. Calculating initial RCS patterns...")
    visualizer = RCSVisualizer()
    
    # Calculate RCS for both geometries
    angles_f117, rcs_f117 = rcs_calc.calculate_2d_rcs_pattern(f117_geometry)
    angles_conv, rcs_conv = rcs_calc.calculate_2d_rcs_pattern(conventional_geometry)
    
    stats_f117 = compute_rcs_statistics(rcs_f117)
    stats_conv = compute_rcs_statistics(rcs_conv)
    
    print(f"\n   F-117 inspired geometry:")
    print(f"   - Mean RCS: {stats_f117['mean_rcs_db']:.2f} dBsm")
    print(f"   - Max RCS: {stats_f117['max_rcs_db']:.2f} dBsm")
    print(f"   - Min RCS: {stats_f117['min_rcs_db']:.2f} dBsm")
    
    print(f"\n   Conventional geometry:")
    print(f"   - Mean RCS: {stats_conv['mean_rcs_db']:.2f} dBsm")
    print(f"   - Max RCS: {stats_conv['max_rcs_db']:.2f} dBsm")
    print(f"   - Min RCS: {stats_conv['min_rcs_db']:.2f} dBsm")
    
    # Visualize comparison
    fig1 = visualizer.plot_geometry_comparison(
        [f117_geometry, conventional_geometry],
        ["F-117 Inspired", "Conventional"],
        rcs_calc
    )
    plt.savefig('visualizations/initial_comparison.png', dpi=150, bbox_inches='tight')
    
    # 4. Optimize the conventional geometry
    print("\n4. Optimizing conventional geometry...")
    print("   This demonstrates how topology optimization can transform")
    print("   a conventional shape into a stealthier configuration.")
    
    # Define target angles - frontal aspect is most important
    target_angles = np.array([0, 30, 330])  # Front and slight off-angles
    
    optimizer = TopologyOptimizer(rcs_calc, max_displacement=2.0)
    
    # Run gradient-based optimization
    print("\n   Running gradient-based optimization...")
    optimized_gradient = optimizer.gradient_descent_2d(
        conventional_geometry,
        n_iterations=30,
        learning_rate=0.2,
        target_angles=target_angles,
        smoothing=0.5
    )
    
    gradient_history = optimizer.history.copy()
    
    # Run evolutionary optimization for comparison
    print("\n   Running evolutionary optimization...")
    optimized_evolutionary = optimizer.evolutionary_optimization_2d(
        conventional_geometry,
        population_size=20,
        n_generations=30,
        mutation_rate=0.15,
        crossover_rate=0.7,
        target_angles=target_angles
    )
    
    evolutionary_history = optimizer.history.copy()
    
    # 5. Compare optimization methods
    print("\n5. Comparing optimization results...")
    
    # Calculate final RCS patterns
    _, rcs_gradient = rcs_calc.calculate_2d_rcs_pattern(optimized_gradient)
    _, rcs_evolutionary = rcs_calc.calculate_2d_rcs_pattern(optimized_evolutionary)
    
    stats_gradient = compute_rcs_statistics(rcs_gradient)
    stats_evolutionary = compute_rcs_statistics(rcs_evolutionary)
    
    print(f"\n   Gradient-based optimization:")
    print(f"   - Final mean RCS: {stats_gradient['mean_rcs_db']:.2f} dBsm")
    print(f"   - Reduction: {stats_conv['mean_rcs_db'] - stats_gradient['mean_rcs_db']:.2f} dB")
    
    print(f"\n   Evolutionary optimization:")
    print(f"   - Final mean RCS: {stats_evolutionary['mean_rcs_db']:.2f} dBsm")
    print(f"   - Reduction: {stats_conv['mean_rcs_db'] - stats_evolutionary['mean_rcs_db']:.2f} dB")
    
    # Visualize final comparison
    fig2 = visualizer.plot_geometry_comparison(
        [conventional_geometry, optimized_gradient, optimized_evolutionary, f117_geometry],
        ["Original", "Gradient Opt.", "Evolutionary Opt.", "F-117 Inspired"],
        rcs_calc
    )
    plt.savefig('visualizations/optimization_comparison.png', dpi=150, bbox_inches='tight')
    
    # 6. Create optimization animation
    print("\n6. Creating optimization animation...")
    fig3, anim = visualizer.create_optimization_animation(
        gradient_history,
        rcs_calc,
        save_path='visualizations/optimization_progress.gif'
    )
    
    # 7. Plot detailed metrics
    print("\n7. Generating optimization metrics...")
    fig4 = visualizer.plot_optimization_metrics(gradient_history)
    plt.savefig('visualizations/optimization_metrics.png', dpi=150, bbox_inches='tight')
    
    # 8. Analyze specific angle performance
    print("\n8. Analyzing angle-specific RCS reduction...")
    
    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Frontal aspect comparison
    frontal_angles = np.linspace(-30, 30, 61)
    rcs_original_frontal = rcs_calc.calculate_2d_rcs(conventional_geometry, frontal_angles)
    rcs_optimized_frontal = rcs_calc.calculate_2d_rcs(optimized_gradient, frontal_angles)
    rcs_f117_frontal = rcs_calc.calculate_2d_rcs(f117_geometry, frontal_angles)
    
    ax1.plot(frontal_angles, 10*np.log10(rcs_original_frontal), 'r-', 
             linewidth=2, label='Original')
    ax1.plot(frontal_angles, 10*np.log10(rcs_optimized_frontal), 'b-', 
             linewidth=2, label='Optimized')
    ax1.plot(frontal_angles, 10*np.log10(rcs_f117_frontal), 'g--', 
             linewidth=2, label='F-117 Inspired')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('RCS (dBsm)')
    ax1.set_title('Frontal Aspect RCS Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Full polar plot
    ax2 = plt.subplot(122, projection='polar')
    angles_rad = np.deg2rad(angles_conv)
    ax2.plot(angles_rad, rcs_conv, 'r-', linewidth=2, label='Original')
    ax2.plot(angles_rad, rcs_gradient, 'b-', linewidth=2, label='Optimized')
    ax2.plot(angles_rad, rcs_f117, 'g--', linewidth=2, label='F-117 Inspired')
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(-1)
    ax2.set_title('Full 360° RCS Pattern Comparison')
    ax2.grid(True)
    ax2.set_ylim(-40, np.max([np.max(rcs_conv), np.max(rcs_gradient)]) + 5)
    
    plt.tight_layout()
    plt.savefig('visualizations/angle_analysis.png', dpi=150, bbox_inches='tight')
    
    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Original mean RCS: {stats_conv['mean_rcs_db']:.2f} dBsm")
    print(f"Optimized mean RCS: {stats_gradient['mean_rcs_db']:.2f} dBsm")
    print(f"F-117 inspired mean RCS: {stats_f117['mean_rcs_db']:.2f} dBsm")
    print(f"\nTotal RCS reduction achieved: {stats_conv['mean_rcs_db'] - stats_gradient['mean_rcs_db']:.2f} dB")
    print(f"This represents a {10**((stats_conv['mean_rcs_db'] - stats_gradient['mean_rcs_db'])/10):.1f}x reduction in radar signature!")
    
    print("\nKey insights:")
    print("1. Faceted surfaces (like F-117) redirect radar energy away from source")
    print("2. Optimization naturally discovers angular/faceted shapes")
    print("3. Frontal aspect RCS is dramatically reduced")
    print("4. Trade-off exists between all-aspect and specific-angle stealth")
    
    plt.show()


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Run the demo
    main() 