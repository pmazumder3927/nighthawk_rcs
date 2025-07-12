#!/usr/bin/env python3
"""
Demonstration of RCS calculations using the repository's tools.

This script shows how to use the Physical Optics implementation to calculate
and visualize RCS patterns for different geometries, demonstrating the concepts
from the stealth aircraft article.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    
    from rcs_calc_3d import RCS3DCalculator
    from geometry_3d import Geometry3D
    from visualization_3d import RCSVisualizer3D
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the repository root directory")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

class ArticleConceptDemo:
    """Demonstrates the article's concepts using actual RCS calculations."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.frequency = 10e9  # X-band radar (10 GHz)
        self.rcs_calculator = RCS3DCalculator(frequency=self.frequency)
        self.visualizer = RCSVisualizer3D()
        
    def create_sphere_vs_faceted_comparison(self):
        """
        Create a practical comparison between sphere and faceted object RCS.
        This demonstrates the "disco ball effect" mentioned in the article.
        """
        print("Creating sphere vs faceted comparison...")
        
        # Create a sphere geometry
        sphere_geom = Geometry3D.create_sphere(radius=1.0, resolution=50)
        
        # Create a faceted geometry (simplified diamond shape)
        vertices = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        faces = np.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            [0, 3, 5], [3, 2, 5], [2, 1, 5], [1, 0, 5]
        ])
        faceted_geom = Geometry3D.from_vertices_faces(vertices, faces)
        
        # Calculate RCS patterns
        angles = np.linspace(0, 360, 73)
        
        # Sphere RCS (should be relatively constant)
        sphere_rcs = []
        for angle in angles:
            rcs = self.rcs_calculator.calculate_rcs(sphere_geom.mesh, 90, angle)
            sphere_rcs.append(10 * np.log10(rcs))  # Convert to dBsm
        
        # Faceted RCS (should be highly directional)
        faceted_rcs = []
        for angle in angles:
            rcs = self.rcs_calculator.calculate_rcs(faceted_geom.mesh, 90, angle)
            faceted_rcs.append(10 * np.log10(rcs))  # Convert to dBsm
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                       subplot_kw=dict(projection='polar'))
        
        # Sphere pattern
        ax1.plot(np.deg2rad(angles), sphere_rcs, 'b-', linewidth=2, label='Sphere')
        ax1.set_title('Sphere RCS Pattern\n(Isotropic - "Disco Ball Effect")', fontsize=14)
        ax1.set_ylim(min(sphere_rcs + faceted_rcs), max(sphere_rcs + faceted_rcs))
        ax1.grid(True)
        
        # Faceted pattern
        ax2.plot(np.deg2rad(angles), faceted_rcs, 'r-', linewidth=2, label='Faceted')
        ax2.set_title('Faceted RCS Pattern\n(Directional - Good for Stealth)', fontsize=14)
        ax2.set_ylim(min(sphere_rcs + faceted_rcs), max(sphere_rcs + faceted_rcs))
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/article-vis/sphere_vs_faceted_rcs.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return sphere_rcs, faceted_rcs, angles
    
    def demonstrate_po_assumptions(self):
        """
        Demonstrate the Physical Optics assumptions using actual calculations.
        Show how only illuminated surfaces contribute to scattering.
        """
        print("Demonstrating Physical Optics assumptions...")
        
        # Create a simple cube to show illuminated vs shadowed regions
        cube_geom = Geometry3D.create_cube(size=2.0)
        
        # Calculate RCS for different incident angles
        elevation_angles = np.linspace(0, 180, 37)
        azimuth_angle = 0  # Fixed azimuth
        
        cube_rcs = []
        for theta in elevation_angles:
            try:
                rcs = self.rcs_calculator.calculate_rcs(cube_geom.mesh, theta, azimuth_angle)
                cube_rcs.append(10 * np.log10(rcs))
            except:
                cube_rcs.append(-50)  # Very low RCS for invalid calculations
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(elevation_angles, cube_rcs, 'g-', linewidth=3, marker='o', markersize=6)
        ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
        ax.set_ylabel('RCS (dBsm)', fontsize=12)
        ax.set_title('Cube RCS vs Elevation Angle\n(Demonstrating PO Illumination Effect)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.annotate('Front face illuminated', xy=(0, max(cube_rcs)), xytext=(20, max(cube_rcs)-5),
                   arrowprops=dict(arrowstyle='->', color='blue'), fontsize=12)
        ax.annotate('Edge effects', xy=(90, cube_rcs[len(cube_rcs)//2]), xytext=(110, cube_rcs[len(cube_rcs)//2]+5),
                   arrowprops=dict(arrowstyle='->', color='red'), fontsize=12)
        ax.annotate('Back face illuminated', xy=(180, max(cube_rcs)), xytext=(160, max(cube_rcs)-5),
                   arrowprops=dict(arrowstyle='->', color='blue'), fontsize=12)
        
        plt.tight_layout()
        plt.savefig('visualizations/article-vis/po_assumptions_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return elevation_angles, cube_rcs
    
    def frequency_scaling_demonstration(self):
        """
        Demonstrate how RCS scales with frequency (wavelength).
        This shows the high-frequency assumption of Physical Optics.
        """
        print("Demonstrating frequency scaling...")
        
        # Create a simple sphere
        sphere_geom = Geometry3D.create_sphere(radius=0.5, resolution=30)
        
        # Test different frequencies
        frequencies = np.logspace(8, 11, 20)  # 100 MHz to 100 GHz
        wavelengths = 3e8 / frequencies
        
        sphere_rcs_freq = []
        for freq in frequencies:
            calc = RCS3DCalculator(frequency=freq)
            try:
                rcs = calc.calculate_rcs(sphere_geom.mesh, 90, 0)
                sphere_rcs_freq.append(rcs)
            except:
                sphere_rcs_freq.append(1e-10)  # Very small RCS for failed calculations
        
        # Theoretical RCS for sphere (high frequency limit)
        radius = 0.5
        theoretical_rcs = np.pi * radius**2 * np.ones_like(frequencies)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # RCS vs Frequency
        ax1.loglog(frequencies/1e9, sphere_rcs_freq, 'b-', linewidth=2, label='PO Calculation')
        ax1.loglog(frequencies/1e9, theoretical_rcs, 'r--', linewidth=2, label='Theoretical (High Freq)')
        ax1.set_xlabel('Frequency (GHz)', fontsize=12)
        ax1.set_ylabel('RCS (m²)', fontsize=12)
        ax1.set_title('RCS vs Frequency\n(Sphere, radius=0.5m)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RCS vs ka (electrical size)
        k = 2 * np.pi / wavelengths
        ka = k * radius
        
        ax2.loglog(ka, sphere_rcs_freq, 'b-', linewidth=2, label='PO Calculation')
        ax2.loglog(ka, theoretical_rcs, 'r--', linewidth=2, label='Theoretical (High Freq)')
        ax2.axvline(x=1, color='g', linestyle=':', label='ka = 1 (Rayleigh limit)')
        ax2.set_xlabel('ka (electrical size)', fontsize=12)
        ax2.set_ylabel('RCS (m²)', fontsize=12)
        ax2.set_title('RCS vs Electrical Size\n(ka = 2πr/λ)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/article-vis/frequency_scaling_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return frequencies, sphere_rcs_freq, theoretical_rcs
    
    def create_stealth_shape_analysis(self):
        """
        Analyze different stealth-relevant shapes and their RCS patterns.
        This demonstrates the principles discussed in the article.
        """
        print("Creating stealth shape analysis...")
        
        # Create different geometric shapes
        shapes = {}
        
        # Sphere (bad for stealth)
        shapes['Sphere'] = Geometry3D.create_sphere(radius=1.0, resolution=30)
        
        # Flat plate (good when angled)
        vertices = np.array([
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        shapes['Flat Plate'] = Geometry3D.from_vertices_faces(vertices, faces)
        
        # Diamond (faceted, good for stealth)
        vertices = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        faces = np.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            [0, 3, 5], [3, 2, 5], [2, 1, 5], [1, 0, 5]
        ])
        shapes['Diamond'] = Geometry3D.from_vertices_faces(vertices, faces)
        
        # Calculate RCS patterns for each shape
        angles = np.linspace(0, 360, 73)
        rcs_patterns = {}
        
        for name, geometry in shapes.items():
            pattern = []
            for angle in angles:
                try:
                    rcs = self.rcs_calculator.calculate_rcs(geometry.mesh, 90, angle)
                    pattern.append(10 * np.log10(rcs))
                except:
                    pattern.append(-50)  # Very low RCS
            rcs_patterns[name] = pattern
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        colors = ['blue', 'red', 'green']
        for i, (name, pattern) in enumerate(rcs_patterns.items()):
            ax.plot(np.deg2rad(angles), pattern, color=colors[i], 
                   linewidth=3, label=name)
        
        ax.set_title('RCS Pattern Comparison\n(Different Geometries)', fontsize=16)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/article-vis/stealth_shape_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return rcs_patterns, angles
    
    def create_interactive_rcs_explorer(self):
        """
        Create an interactive RCS explorer that allows users to see how
        different parameters affect RCS calculations.
        """
        print("Creating interactive RCS explorer...")
        
        # Create a simple faceted geometry
        vertices = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        faces = np.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            [0, 3, 5], [3, 2, 5], [2, 1, 5], [1, 0, 5]
        ])
        geometry = Geometry3D.from_vertices_faces(vertices, faces)
        
        # Calculate RCS for different elevation and azimuth angles
        theta_angles = np.linspace(0, 180, 19)
        phi_angles = np.linspace(0, 360, 37)
        
        # Create meshgrid for 3D plotting
        THETA, PHI = np.meshgrid(theta_angles, phi_angles)
        RCS_dB = np.zeros_like(THETA)
        
        for i, theta in enumerate(theta_angles):
            for j, phi in enumerate(phi_angles):
                try:
                    rcs = self.rcs_calculator.calculate_rcs(geometry.mesh, theta, phi)
                    RCS_dB[j, i] = 10 * np.log10(rcs)
                except:
                    RCS_dB[j, i] = -50
        
        # Convert to spherical coordinates for 3D plotting
        r = 1.0  # Unit sphere
        x = r * np.sin(np.deg2rad(THETA)) * np.cos(np.deg2rad(PHI))
        y = r * np.sin(np.deg2rad(THETA)) * np.sin(np.deg2rad(PHI))
        z = r * np.cos(np.deg2rad(THETA))
        
        # Create interactive 3D plot
        fig = go.Figure(data=[
            go.Surface(
                x=x, y=y, z=z,
                surfacecolor=RCS_dB,
                colorscale='Viridis',
                colorbar=dict(title='RCS (dBsm)'),
                name='RCS Pattern'
            )
        ])
        
        fig.update_layout(
            title='Interactive RCS Pattern Explorer<br>Diamond Geometry',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        fig.write_html('visualizations/article-vis/interactive_rcs_explorer.html')
        print("Interactive RCS explorer saved as 'interactive_rcs_explorer.html'")
        
        return THETA, PHI, RCS_dB
    
    def run_all_demonstrations(self):
        """Run all demonstrations."""
        print("🚀 Running Article Concept Demonstrations")
        print("=" * 50)
        
        try:
            # 1. Sphere vs Faceted comparison
            print("\n1. Sphere vs Faceted Comparison...")
            self.create_sphere_vs_faceted_comparison()
            
            # 2. Physical Optics assumptions
            print("\n2. Physical Optics Assumptions...")
            self.demonstrate_po_assumptions()
            
            # 3. Frequency scaling
            print("\n3. Frequency Scaling...")
            self.frequency_scaling_demonstration()
            
            # 4. Stealth shape analysis
            print("\n4. Stealth Shape Analysis...")
            self.create_stealth_shape_analysis()
            
            # 5. Interactive RCS explorer
            print("\n5. Interactive RCS Explorer...")
            self.create_interactive_rcs_explorer()
            
            print("\n✅ All demonstrations completed successfully!")
            print("📁 Files saved in: visualizations/article-vis/")
            
        except Exception as e:
            print(f"❌ Error during demonstration: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    demo = ArticleConceptDemo()
    demo.run_all_demonstrations() 