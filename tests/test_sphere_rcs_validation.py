"""
Test script to validate RCS calculations against analytical sphere solutions.

This script compares the RCS3DCalculator results with known Physical Optics
solutions for perfectly conducting spheres of various sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
import trimesh
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rcs_calc_3d import RCS3DCalculator


def analytical_sphere_rcs_po(radius, wavelength, theta_deg=0):
    """
    Calculate analytical Physical Optics RCS for a perfectly conducting sphere.
    
    For backscatter (monostatic), the PO approximation gives:
    σ = π * a² * |2 * j₁(2ka) / (2ka)|²
    
    where j₁ is the spherical Bessel function of the first kind.
    
    For large spheres (ka >> 1), this approaches the geometric optics limit: σ ≈ π * a²
    
    Args:
        radius: Sphere radius in meters
        wavelength: Wavelength in meters
        theta_deg: Angle from forward direction (0 for backscatter)
        
    Returns:
        RCS in square meters
    """
    k = 2 * np.pi / wavelength
    ka = k * radius
    
    # For backscatter at theta=0 (nose-on)
    if theta_deg == 0:
        # Use spherical Bessel function
        # j₁(x) = sin(x)/x² - cos(x)/x
        x = 2 * ka
        if x < 1e-6:
            # Small argument approximation
            j1_over_x = 1/3 - x**2/30
        else:
            j1_over_x = (np.sin(x) - x * np.cos(x)) / x**3
        
        # PO backscatter RCS
        rcs = np.pi * radius**2 * (2 * x * j1_over_x)**2
    else:
        # For other angles, use geometric optics approximation
        # This is valid for large spheres
        rcs = np.pi * radius**2
    
    return rcs


def create_sphere_mesh(radius, subdivisions=3):
    """Create a sphere mesh with given radius and subdivision level."""
    # Create icosphere for better uniformity
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return sphere


def test_sphere_rcs_trends():
    """Test RCS calculations for spheres of different sizes and compare with analytical solutions."""
    
    # Setup
    frequency = 10e9  # 10 GHz
    wavelength = 3e8 / frequency
    print(f"Testing at frequency: {frequency/1e9:.1f} GHz")
    print(f"Wavelength: {wavelength*1000:.1f} mm")
    
    # Initialize RCS calculator
    rcs_calc = RCS3DCalculator(frequency=frequency)
    
    # Test sphere sizes from 0.1λ to 10λ
    radii_wavelengths = np.logspace(-1, 1, 20)  # 0.1λ to 10λ
    radii = radii_wavelengths * wavelength
    
    # Storage for results
    rcs_numerical = []
    rcs_analytical = []
    rcs_optical = []  # Geometric optics limit
    
    print("\nCalculating RCS for spheres of different sizes...")
    print(f"{'Radius/λ':<10} {'Radius (mm)':<12} {'RCS Num (dBsm)':<15} {'RCS Anal (dBsm)':<15} {'RCS GO (dBsm)':<15} {'Error (dB)':<10}")
    print("-" * 90)
    
    for i, radius in enumerate(radii):
        # Create sphere mesh
        # More subdivisions for larger spheres
        subdivisions = 3 if radii_wavelengths[i] < 2 else 4
        sphere_mesh = create_sphere_mesh(radius, subdivisions=subdivisions)
        
        # Calculate numerical RCS
        rcs_num = rcs_calc.calculate_rcs(sphere_mesh, theta=0, phi=0, polarization='VV')
        rcs_numerical.append(rcs_num)
        
        # Calculate analytical RCS
        rcs_anal = analytical_sphere_rcs_po(radius, wavelength, theta_deg=0)
        rcs_analytical.append(rcs_anal)
        
        # Geometric optics limit (large sphere approximation)
        rcs_go = np.pi * radius**2
        rcs_optical.append(rcs_go)
        
        # Convert to dBsm for display
        rcs_num_db = 10 * np.log10(rcs_num)
        rcs_anal_db = 10 * np.log10(rcs_anal)
        rcs_go_db = 10 * np.log10(rcs_go)
        error_db = rcs_num_db - rcs_anal_db
        
        print(f"{radii_wavelengths[i]:>10.2f} {radius*1000:>12.2f} {rcs_num_db:>15.2f} {rcs_anal_db:>15.2f} {rcs_go_db:>15.2f} {error_db:>10.2f}")
    
    # Convert to arrays
    rcs_numerical = np.array(rcs_numerical)
    rcs_analytical = np.array(rcs_analytical)
    rcs_optical = np.array(rcs_optical)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot 1: RCS vs sphere size
    plt.subplot(2, 2, 1)
    plt.loglog(radii_wavelengths, rcs_numerical, 'b-o', label='Numerical (RCS3DCalculator)', markersize=6)
    plt.loglog(radii_wavelengths, rcs_analytical, 'r--', label='Analytical PO', linewidth=2)
    plt.loglog(radii_wavelengths, rcs_optical, 'g:', label='Geometric Optics (πa²)', linewidth=2)
    plt.xlabel('Sphere Radius (wavelengths)')
    plt.ylabel('RCS (m²)')
    plt.title('Sphere RCS Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: RCS in dBsm
    plt.subplot(2, 2, 2)
    plt.semilogx(radii_wavelengths, 10*np.log10(rcs_numerical), 'b-o', label='Numerical', markersize=6)
    plt.semilogx(radii_wavelengths, 10*np.log10(rcs_analytical), 'r--', label='Analytical PO', linewidth=2)
    plt.semilogx(radii_wavelengths, 10*np.log10(rcs_optical), 'g:', label='Geometric Optics', linewidth=2)
    plt.xlabel('Sphere Radius (wavelengths)')
    plt.ylabel('RCS (dBsm)')
    plt.title('Sphere RCS in dB scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Normalized RCS (σ/πa²)
    plt.subplot(2, 2, 3)
    normalized_numerical = rcs_numerical / (np.pi * radii**2)
    normalized_analytical = rcs_analytical / (np.pi * radii**2)
    plt.semilogx(radii_wavelengths, normalized_numerical, 'b-o', label='Numerical', markersize=6)
    plt.semilogx(radii_wavelengths, normalized_analytical, 'r--', label='Analytical PO', linewidth=2)
    plt.axhline(y=1, color='g', linestyle=':', label='GO limit', linewidth=2)
    plt.xlabel('Sphere Radius (wavelengths)')
    plt.ylabel('Normalized RCS (σ/πa²)')
    plt.title('Normalized Sphere RCS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.5])
    
    # Plot 4: Error analysis
    plt.subplot(2, 2, 4)
    error_db = 10*np.log10(rcs_numerical) - 10*np.log10(rcs_analytical)
    plt.semilogx(radii_wavelengths, error_db, 'k-o', label='Error (dB)', markersize=6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Sphere Radius (wavelengths)')
    plt.ylabel('Error (dB)')
    plt.title('Numerical vs Analytical Error')
    plt.grid(True, alpha=0.3)
    
    # Add text box with error statistics
    avg_error = np.mean(np.abs(error_db))
    max_error = np.max(np.abs(error_db))
    textstr = f'Avg Error: {avg_error:.2f} dB\nMax Error: {max_error:.2f} dB'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('sphere_rcs_validation.png', dpi=150)
    plt.show()
    
    # Summary statistics
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"Average absolute error: {avg_error:.2f} dB")
    print(f"Maximum absolute error: {max_error:.2f} dB")
    print(f"RMS error: {np.sqrt(np.mean(error_db**2)):.2f} dB")
    
    # Check trends
    small_sphere_idx = radii_wavelengths < 1
    large_sphere_idx = radii_wavelengths > 5
    
    if len(rcs_numerical[small_sphere_idx]) > 0:
        small_trend = np.polyfit(np.log10(radii_wavelengths[small_sphere_idx]), 
                                np.log10(rcs_numerical[small_sphere_idx]), 1)[0]
        print(f"\nSmall sphere trend (a < λ): RCS ∝ a^{small_trend:.2f} (expected: ~4 for Rayleigh)")
    
    if len(rcs_numerical[large_sphere_idx]) > 0:
        large_trend = np.polyfit(np.log10(radii_wavelengths[large_sphere_idx]), 
                                np.log10(rcs_numerical[large_sphere_idx]), 1)[0]
        print(f"Large sphere trend (a > 5λ): RCS ∝ a^{large_trend:.2f} (expected: 2 for geometric optics)")
    
    # Physical correctness checks
    print("\nPhysical Correctness Checks:")
    print(f"✓ All RCS values positive: {np.all(rcs_numerical > 0)}")
    print(f"✓ Monotonic increase with size: {np.all(np.diff(rcs_numerical) > 0)}")
    print(f"✓ Approaches GO limit for large spheres: {np.abs(normalized_numerical[-1] - 1) < 0.1}")
    
    return rcs_numerical, rcs_analytical, radii_wavelengths


def test_angular_pattern():
    """Test RCS angular pattern for a sphere (should be nearly isotropic)."""
    
    frequency = 10e9
    wavelength = 3e8 / frequency
    radius = 2 * wavelength  # 2λ radius sphere
    
    print(f"\nTesting angular pattern for {radius/wavelength:.1f}λ sphere...")
    
    # Create sphere
    sphere_mesh = create_sphere_mesh(radius, subdivisions=4)
    
    # Initialize calculator
    rcs_calc = RCS3DCalculator(frequency=frequency)
    
    # Calculate RCS pattern in elevation
    theta_angles = np.linspace(0, 180, 37)
    rcs_theta = []
    
    for theta in theta_angles:
        rcs = rcs_calc.calculate_rcs(sphere_mesh, theta=theta, phi=0, polarization='VV')
        rcs_theta.append(rcs)
    
    rcs_theta = np.array(rcs_theta)
    
    # Plot angular pattern
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(theta_angles, 10*np.log10(rcs_theta), 'b-', linewidth=2)
    plt.xlabel('Theta (degrees)')
    plt.ylabel('RCS (dBsm)')
    plt.title(f'Sphere RCS Pattern ({radius/wavelength:.1f}λ radius)')
    plt.grid(True, alpha=0.3)
    
    # Polar plot
    plt.subplot(1, 2, 2, projection='polar')
    plt.plot(np.deg2rad(theta_angles), 10*np.log10(rcs_theta), 'b-', linewidth=2)
    plt.title('Polar Pattern')
    
    plt.tight_layout()
    plt.savefig('sphere_angular_pattern.png', dpi=150)
    plt.show()
    
    # Check isotropy
    variation_db = np.max(10*np.log10(rcs_theta)) - np.min(10*np.log10(rcs_theta))
    print(f"Angular variation: {variation_db:.2f} dB")
    print(f"Nearly isotropic: {variation_db < 3} (< 3 dB variation)")


if __name__ == "__main__":
    print("Sphere RCS Validation Test")
    print("=" * 50)
    
    # Run main validation test
    rcs_num, rcs_anal, radii_wl = test_sphere_rcs_trends()
    
    # Run angular pattern test
    test_angular_pattern()
    
    print("\nTest completed. Check generated plots for detailed results.")