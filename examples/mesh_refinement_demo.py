"""
Mesh Refinement Demo for Electrically Large Objects

This demonstrates how mesh resolution affects RCS calculations
and shows the automatic mesh refinement features.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry_3d import Geometry3D, create_simple_shape_3d
from src.rcs_calc_3d import RCS3DCalculator
from src.visualization_3d import RCSVisualizer3D


def demonstrate_mesh_refinement():
    """Show the importance of mesh refinement for electrically large objects."""
    
    print("=" * 70)
    print("MESH REFINEMENT DEMONSTRATION")
    print("Showing how mesh resolution affects RCS accuracy")
    print("=" * 70)
    
    # Create a 1m diameter sphere (more reasonable for mesh demo)
    frequency = 10e9  # X-band
    wavelength = 3e8 / frequency
    
    print(f"\nRadar parameters:")
    print(f"  Frequency: {frequency/1e9} GHz")
    print(f"  Wavelength: {wavelength*100:.1f} cm")
    
    # Create sphere with different mesh resolutions
    sphere_radius = 0.5  # 1m diameter
    
    print(f"\nObject: Sphere with {2*sphere_radius}m diameter")
    print(f"Electrical size: ka = {2*np.pi*sphere_radius/wavelength:.1f}")
    
    # Test different subdivision levels
    subdivision_levels = [0, 1, 2, 3]
    results = []
    
    for level in subdivision_levels:
        print(f"\n{'='*50}")
        print(f"Testing subdivision level {level}...")
        
        # Create sphere
        sphere = create_simple_shape_3d('sphere', size=sphere_radius, subdivisions=level)
        
        # Check mesh quality
        quality = sphere.check_mesh_resolution(wavelength)
        
        print(f"\nMesh statistics:")
        print(f"  Faces: {quality['num_faces']}")
        print(f"  Max edge length: {quality['max_edge_length']:.3f}m")
        print(f"  Edges per wavelength: {quality['edges_per_wavelength']:.1f}")
        print(f"  Resolution adequate: {quality['resolution_adequate']}")
        
        if 'warning' in quality:
            print(f"  ⚠️  {quality['warning']}")
        
        # Calculate RCS
        rcs_calc = RCS3DCalculator(frequency=frequency, check_mesh_quality=False)
        
        # Test angles
        phi_angles = np.linspace(0, 360, 73)
        theta = 90.0
        theta_array = np.full_like(phi_angles, theta)
        
        rcs_values = rcs_calc.calculate_rcs(sphere.mesh, theta_array, phi_angles)
        rcs_db = 10 * np.log10(rcs_values + 1e-10)
        
        # Theoretical sphere RCS
        theoretical_rcs = np.pi * sphere_radius**2
        theoretical_db = 10 * np.log10(theoretical_rcs)
        
        results.append({
            'level': level,
            'quality': quality,
            'rcs_values': rcs_values,
            'rcs_db': rcs_db,
            'mean_rcs_db': np.mean(rcs_db),
            'std_rcs_db': np.std(rcs_db),
            'sphere': sphere
        })
        
        print(f"\nRCS results:")
        print(f"  Mean RCS: {np.mean(rcs_db):.1f} dBsm")
        print(f"  Std deviation: {np.std(rcs_db):.1f} dB")
        print(f"  Theoretical: {theoretical_db:.1f} dBsm")
        print(f"  Error: {np.mean(rcs_db) - theoretical_db:.1f} dB")
    
    # Now show automatic refinement
    print(f"\n{'='*70}")
    print("AUTOMATIC MESH REFINEMENT")
    print(f"{'='*70}")
    
    # Create coarse sphere
    coarse_sphere = create_simple_shape_3d('sphere', size=sphere_radius, subdivisions=0)
    print(f"\nOriginal mesh: {len(coarse_sphere.mesh.faces)} faces")
    
    # Refine to target edge length
    target_edge = wavelength / 10
    print(f"\nRefining to target edge length: {target_edge:.3f}m")
    
    refined_sphere = coarse_sphere.remesh_to_target_edge_length(target_edge)
    print(f"Refined mesh: {len(refined_sphere.mesh.faces)} faces")
    
    # Check refined quality
    refined_quality = refined_sphere.check_mesh_resolution(wavelength)
    print(f"Max edge after refinement: {refined_quality['max_edge_length']:.3f}m")
    print(f"Edges per wavelength: {refined_quality['edges_per_wavelength']:.1f}")
    
    # Calculate RCS with refined mesh
    print("\nCalculating RCS with refined mesh...")
    rcs_calc_refined = RCS3DCalculator(frequency=frequency)
    rcs_refined = rcs_calc_refined.calculate_rcs(refined_sphere.mesh, theta_array, phi_angles)
    rcs_refined_db = 10 * np.log10(rcs_refined + 1e-10)
    
    print(f"\nRefined RCS results:")
    print(f"  Mean RCS: {np.mean(rcs_refined_db):.1f} dBsm")
    print(f"  Std deviation: {np.std(rcs_refined_db):.1f} dB")
    print(f"  Error from theoretical: {np.mean(rcs_refined_db) - theoretical_db:.1f} dB")
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. RCS patterns for different subdivisions
    for i, result in enumerate(results):
        ax1.plot(phi_angles, result['rcs_db'], label=f"Level {result['level']} ({result['quality']['num_faces']} faces)", 
                linewidth=2-i*0.3, alpha=0.8)
    
    ax1.axhline(y=theoretical_db, color='k', linestyle='--', label='Theoretical')
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('RCS (dBsm)')
    ax1.set_title('RCS Pattern vs Mesh Resolution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    
    # 2. RCS error vs subdivision level
    errors = [r['mean_rcs_db'] - theoretical_db for r in results]
    std_devs = [r['std_rcs_db'] for r in results]
    
    ax2.plot(subdivision_levels, errors, 'bo-', linewidth=2, markersize=8, label='Mean error')
    ax2.plot(subdivision_levels, std_devs, 'rs-', linewidth=2, markersize=8, label='Std deviation')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Subdivision Level')
    ax2.set_ylabel('dB')
    ax2.set_title('RCS Accuracy vs Mesh Resolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Mesh quality metrics
    faces_count = [r['quality']['num_faces'] for r in results]
    edges_per_lambda = [r['quality']['edges_per_wavelength'] for r in results]
    
    ax3_twin = ax3.twinx()
    
    bars = ax3.bar(subdivision_levels, faces_count, alpha=0.7, label='Face count')
    line = ax3_twin.plot(subdivision_levels, edges_per_lambda, 'ro-', linewidth=2, 
                        markersize=8, label='Edges/λ')
    
    ax3_twin.axhline(y=10, color='g', linestyle='--', label='Recommended (10 edges/λ)')
    
    ax3.set_xlabel('Subdivision Level')
    ax3.set_ylabel('Number of Faces', color='b')
    ax3_twin.set_ylabel('Edges per Wavelength', color='r')
    ax3.set_title('Mesh Quality Metrics')
    ax3.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 4. Automatic refinement comparison
    ax4.text(0.5, 0.9, 'Automatic Refinement Results', ha='center', 
             fontsize=14, weight='bold', transform=ax4.transAxes)
    
    comparison_text = f"""
Original Mesh:
  • Faces: {len(coarse_sphere.mesh.faces)}
  • Max edge: {results[0]['quality']['max_edge_length']:.3f}m
  • RCS error: {errors[0]:.1f} dB
  • Std dev: {std_devs[0]:.1f} dB

Refined Mesh:
  • Faces: {len(refined_sphere.mesh.faces)}
  • Max edge: {refined_quality['max_edge_length']:.3f}m  
  • RCS error: {np.mean(rcs_refined_db) - theoretical_db:.1f} dB
  • Std dev: {np.std(rcs_refined_db):.1f} dB

Improvement:
  • {len(refined_sphere.mesh.faces)/len(coarse_sphere.mesh.faces):.1f}x more faces
  • {abs(errors[0] - (np.mean(rcs_refined_db) - theoretical_db)):.1f} dB better accuracy
  • {(std_devs[0] - np.std(rcs_refined_db)):.1f} dB smoother pattern
"""
    
    ax4.text(0.1, 0.7, comparison_text, fontsize=10, verticalalignment='top',
             family='monospace', transform=ax4.transAxes)
    ax4.axis('off')
    
    plt.suptitle('Mesh Refinement Impact on RCS Calculations', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('../visualizations/mesh_refinement_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # (Removed STL export to reduce file clutter)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("1. Coarse meshes cause spiky, inaccurate RCS patterns")
    print("2. Need ~10 edges per wavelength for smooth results")  
    print("3. Automatic refinement can fix mesh resolution issues")
    print("4. Always check mesh quality for electrically large objects!")
    
    print("\nFiles generated:")
    print("  - mesh_refinement_demo.png")
    
    return results, refined_sphere


if __name__ == "__main__":
    os.makedirs('../visualizations', exist_ok=True)
    results, refined = demonstrate_mesh_refinement()