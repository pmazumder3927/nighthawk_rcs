#!/usr/bin/env python3
"""
Simple demonstration of stealth aircraft mathematics concepts.

This script creates visualizations that demonstrate the key mathematical concepts
from the article using basic numerical methods, without requiring complex imports.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SimpleStealthDemo:
    """Simple demonstration of stealth concepts."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.frequency = 10e9  # X-band radar (10 GHz)
        self.wavelength = 3e8 / self.frequency
        self.k = 2 * np.pi / self.wavelength
        
    def analytical_sphere_rcs(self, radius: float, frequency: float) -> float:
        """
        Calculate theoretical RCS of a sphere in the high-frequency limit.
        
        Args:
            radius: Sphere radius in meters
            frequency: Frequency in Hz
            
        Returns:
            RCS in square meters
        """
        # High-frequency limit: RCS = π * radius²
        return np.pi * radius**2
    
    def analytical_plate_rcs(self, area: float, angle: float) -> float:
        """
        Calculate theoretical RCS of a flat plate.
        
        Args:
            area: Plate area in square meters
            angle: Angle from normal in degrees
            
        Returns:
            RCS in square meters
        """
        angle_rad = np.deg2rad(angle)
        # Simplified model: RCS = 4π * area² * cos²(θ) / λ²
        return 4 * np.pi * area**2 * np.cos(angle_rad)**2 / self.wavelength**2
    
    def create_rcs_comparison(self):
        """Create a comparison of RCS patterns for different shapes."""
        print("Creating RCS comparison...")
        
        # Angles for calculation
        angles = np.linspace(0, 180, 180)
        
        # Sphere RCS (constant)
        sphere_radius = 1.0
        sphere_rcs = np.full_like(angles, self.analytical_sphere_rcs(sphere_radius, self.frequency))
        sphere_rcs_db = 10 * np.log10(sphere_rcs)
        
        # Flat plate RCS (angle-dependent)
        plate_area = np.pi * sphere_radius**2  # Same area as sphere
        plate_rcs = []
        for angle in angles:
            rcs = self.analytical_plate_rcs(plate_area, angle)
            plate_rcs.append(max(rcs, 1e-10))  # Avoid log(0)
        plate_rcs_db = 10 * np.log10(plate_rcs)
        
        # Faceted surface (simplified model)
        faceted_rcs_db = []
        for angle in angles:
            # Simplified faceted model with discrete lobes
            lobe_strength = 0
            for facet_angle in [0, 45, 90, 135, 180]:
                if abs(angle - facet_angle) < 15:
                    lobe_strength += np.exp(-(angle - facet_angle)**2 / 50)
            faceted_rcs_db.append(10 * np.log10(max(lobe_strength * sphere_rcs[0], 1e-10)))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Cartesian plot
        ax1.plot(angles, sphere_rcs_db, 'b-', linewidth=3, label='Sphere (Isotropic)')
        ax1.plot(angles, plate_rcs_db, 'r-', linewidth=3, label='Flat Plate (Directive)')
        ax1.plot(angles, faceted_rcs_db, 'g-', linewidth=3, label='Faceted Surface')
        
        ax1.set_xlabel('Angle (degrees)', fontsize=12)
        ax1.set_ylabel('RCS (dBsm)', fontsize=12)
        ax1.set_title('RCS Patterns: Different Geometries', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Polar plot
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(np.deg2rad(angles), np.maximum(sphere_rcs_db, -50), 'b-', linewidth=3, label='Sphere')
        ax2.plot(np.deg2rad(angles), np.maximum(plate_rcs_db, -50), 'r-', linewidth=3, label='Flat Plate')
        ax2.plot(np.deg2rad(angles), np.maximum(faceted_rcs_db, -50), 'g-', linewidth=3, label='Faceted')
        
        ax2.set_title('RCS Patterns (Polar View)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/article-vis/simple_rcs_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return angles, sphere_rcs_db, plate_rcs_db, faceted_rcs_db
    
    def demonstrate_frequency_effects(self):
        """Demonstrate how RCS changes with frequency."""
        print("Demonstrating frequency effects...")
        
        # Frequency range from 1 GHz to 100 GHz
        frequencies = np.logspace(9, 11, 50)
        
        # Sphere parameters
        sphere_radius = 0.5  # meters
        
        # Calculate RCS for each frequency
        sphere_rcs = []
        plate_rcs = []
        
        for freq in frequencies:
            # Sphere RCS (high frequency limit)
            rcs_sphere = self.analytical_sphere_rcs(sphere_radius, freq)
            sphere_rcs.append(rcs_sphere)
            
            # Flat plate RCS (normal incidence)
            wavelength = 3e8 / freq
            plate_area = np.pi * sphere_radius**2
            rcs_plate = 4 * np.pi * plate_area**2 / wavelength**2
            plate_rcs.append(rcs_plate)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # RCS vs Frequency
        ax1.loglog(frequencies/1e9, sphere_rcs, 'b-', linewidth=3, label='Sphere')
        ax1.loglog(frequencies/1e9, plate_rcs, 'r-', linewidth=3, label='Flat Plate')
        ax1.set_xlabel('Frequency (GHz)', fontsize=12)
        ax1.set_ylabel('RCS (m²)', fontsize=12)
        ax1.set_title('RCS vs Frequency', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RCS vs Electrical Size (ka)
        wavelengths = 3e8 / frequencies
        ka_values = 2 * np.pi * sphere_radius / wavelengths
        
        ax2.loglog(ka_values, sphere_rcs, 'b-', linewidth=3, label='Sphere')
        ax2.loglog(ka_values, plate_rcs, 'r-', linewidth=3, label='Flat Plate')
        ax2.axvline(x=1, color='g', linestyle='--', alpha=0.7, label='ka = 1')
        ax2.set_xlabel('ka (electrical size)', fontsize=12)
        ax2.set_ylabel('RCS (m²)', fontsize=12)
        ax2.set_title('RCS vs Electrical Size', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/article-vis/simple_frequency_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return frequencies, sphere_rcs, plate_rcs
    
    def create_stealth_effectiveness_demo(self):
        """Demonstrate stealth effectiveness with actual numbers."""
        print("Creating stealth effectiveness demonstration...")
        
        # Aircraft data from the article
        aircraft_data = {
            'Small Bird': {'rcs_m2': 0.01, 'rcs_dbsm': -20},
            'Human': {'rcs_m2': 1.0, 'rcs_dbsm': 0},
            'F-15 Eagle': {'rcs_m2': 12.0, 'rcs_dbsm': 10.8},
            'SR-71 Blackbird': {'rcs_m2': 10.0, 'rcs_dbsm': 10.0},
            'F-117 Nighthawk': {'rcs_m2': 0.05, 'rcs_dbsm': -13.0}
        }
        
        # Calculate detection distances (normalized)
        base_distance = 100  # km, baseline detection distance
        detection_distances = {}
        
        for aircraft, data in aircraft_data.items():
            # Detection distance ∝ RCS^(1/4)
            relative_rcs = data['rcs_m2'] / aircraft_data['Human']['rcs_m2']
            detection_distance = base_distance * (relative_rcs**(1/4))
            detection_distances[aircraft] = detection_distance
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # RCS comparison
        names = list(aircraft_data.keys())
        rcs_values = [data['rcs_m2'] for data in aircraft_data.values()]
        colors = ['green', 'blue', 'orange', 'purple', 'red']
        
        bars1 = ax1.bar(range(len(names)), rcs_values, color=colors, alpha=0.7)
        ax1.set_yscale('log')
        ax1.set_xlabel('Aircraft/Object', fontsize=12)
        ax1.set_ylabel('RCS (m²)', fontsize=12)
        ax1.set_title('Radar Cross Section Comparison', fontsize=14)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rcs in zip(bars1, rcs_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{rcs:.3f} m²', ha='center', va='bottom', fontsize=10)
        
        # Detection distance comparison
        distances = [detection_distances[name] for name in names]
        bars2 = ax2.bar(range(len(names)), distances, color=colors, alpha=0.7)
        ax2.set_xlabel('Aircraft/Object', fontsize=12)
        ax2.set_ylabel('Detection Distance (km)', fontsize=12)
        ax2.set_title('Detection Distance (∝ RCS^1/4)', fontsize=14)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, dist in zip(bars2, distances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{dist:.1f} km', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('visualizations/article-vis/simple_stealth_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return aircraft_data, detection_distances
    
    def create_physical_optics_demo(self):
        """Demonstrate Physical Optics concepts."""
        print("Creating Physical Optics demonstration...")
        
        # Simulate illuminated vs shadowed regions
        angles = np.linspace(0, 360, 360)
        
        # Cylinder with incident wave from 45 degrees
        incident_angle = 45
        illuminated_angles = []
        shadowed_angles = []
        
        for angle in angles:
            # Determine if surface element is illuminated
            angle_diff = abs(angle - incident_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff <= 90:
                illuminated_angles.append(angle)
            else:
                shadowed_angles.append(angle)
        
        # Create RCS pattern based on PO assumptions
        po_rcs_pattern = []
        for angle in angles:
            angle_diff = abs(angle - incident_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff <= 90:
                # Illuminated region - cosine dependence
                rcs = np.cos(np.deg2rad(angle_diff))**2
            else:
                # Shadowed region - no contribution
                rcs = 0
            po_rcs_pattern.append(rcs)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Illuminated vs shadowed regions
        ax1 = plt.subplot(121, projection='polar')
        
        # Draw cylinder
        theta_cyl = np.linspace(0, 2*np.pi, 100)
        r_cyl = np.ones_like(theta_cyl)
        ax1.plot(theta_cyl, r_cyl, 'k-', linewidth=3, label='Cylinder')
        
        # Show illuminated region
        theta_ill = np.deg2rad(illuminated_angles)
        r_ill = np.ones_like(theta_ill)
        ax1.scatter(theta_ill, r_ill, c='yellow', s=20, alpha=0.7, label='Illuminated')
        
        # Show shadowed region
        theta_shad = np.deg2rad(shadowed_angles)
        r_shad = np.ones_like(theta_shad)
        ax1.scatter(theta_shad, r_shad, c='purple', s=20, alpha=0.7, label='Shadowed')
        
        # Show incident direction
        ax1.arrow(0, 0, np.deg2rad(incident_angle), 1.5, head_width=0.1, head_length=0.1, 
                 fc='blue', ec='blue', alpha=0.8)
        ax1.text(np.deg2rad(incident_angle), 1.7, 'Incident Wave', ha='center', fontsize=10)
        
        ax1.set_title('PO Illumination Model', fontsize=14)
        ax1.legend()
        
        # RCS pattern
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(np.deg2rad(angles), po_rcs_pattern, 'r-', linewidth=3, label='PO RCS Pattern')
        ax2.set_title('Physical Optics RCS Pattern', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/article-vis/simple_physical_optics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return angles, po_rcs_pattern
    
    def run_all_demos(self):
        """Run all demonstrations."""
        print("🚀 Running Simple Stealth Mathematics Demonstrations")
        print("=" * 60)
        
        try:
            print("\n1. RCS Comparison...")
            self.create_rcs_comparison()
            
            print("\n2. Frequency Effects...")
            self.demonstrate_frequency_effects()
            
            print("\n3. Stealth Effectiveness...")
            self.create_stealth_effectiveness_demo()
            
            print("\n4. Physical Optics...")
            self.create_physical_optics_demo()
            
            print("\n✅ All demonstrations completed successfully!")
            print("📁 Files saved in: visualizations/article-vis/")
            
        except Exception as e:
            print(f"❌ Error during demonstration: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    demo = SimpleStealthDemo()
    demo.run_all_demos() 