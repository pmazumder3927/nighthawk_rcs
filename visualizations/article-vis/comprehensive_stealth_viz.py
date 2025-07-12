#!/usr/bin/env python3
"""
Comprehensive visualizations for stealth aircraft mathematics.

This script creates advanced, thorough visualizations for every mathematical 
concept in the stealth aircraft article, inspired by 3Blue1Brown's style.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import os

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StealthMathViz:
    """
    Comprehensive visualization suite for stealth aircraft mathematics.
    """
    
    def __init__(self, output_dir: str = "visualizations/article-vis/"):
        """Initialize the visualization suite."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Physical constants
        self.c = 3e8  # Speed of light (m/s)
        self.eta = 377.0  # Impedance of free space (Ohms)
        
        # Color scheme (inspired by 3Blue1Brown)
        self.colors = {
            'blue': '#3498db',
            'red': '#e74c3c',
            'green': '#2ecc71',
            'yellow': '#f39c12',
            'purple': '#9b59b6',
            'orange': '#e67e22',
            'dark_blue': '#2980b9',
            'dark_red': '#c0392b'
        }
        
    def rcs_concept_visualization(self):
        """
        Create a comprehensive visualization of the RCS concept.
        Shows radar waves interacting with different objects.
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create a 2x3 grid of subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Basic RCS concept
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_rcs_concept_basic(ax1)
        
        # 2. Sphere vs Flat Plate
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_sphere_vs_plate(ax2)
        
        # 3. Faceted surface
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_faceted_surface(ax3)
        
        # 4. RCS comparison chart
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_rcs_comparison_chart(ax4)
        
        plt.suptitle('Radar Cross Section (RCS) Fundamentals', 
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/1_rcs_concept.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create an interactive 3D version
        self._create_interactive_rcs_concept()
        
    def _plot_rcs_concept_basic(self, ax):
        """Plot the basic RCS concept with radar and target."""
        # Radar position
        radar_x, radar_y = -5, 0
        target_x, target_y = 5, 0
        
        # Draw radar
        radar = Circle((radar_x, radar_y), 0.5, color=self.colors['blue'], alpha=0.8)
        ax.add_patch(radar)
        ax.text(radar_x, radar_y-1, 'Radar', ha='center', fontsize=12, fontweight='bold')
        
        # Draw target
        target = Circle((target_x, target_y), 0.8, color=self.colors['red'], alpha=0.8)
        ax.add_patch(target)
        ax.text(target_x, target_y-1.5, 'Target', ha='center', fontsize=12, fontweight='bold')
        
        # Draw incident wave
        x_incident = np.linspace(radar_x + 0.5, target_x - 0.8, 100)
        y_incident = np.zeros_like(x_incident)
        
        # Add wave pattern
        wave_amp = 0.2
        wave_freq = 8
        y_wave = wave_amp * np.sin(wave_freq * x_incident)
        
        ax.plot(x_incident, y_wave, color=self.colors['blue'], linewidth=2, label='Incident Wave')
        ax.arrow(radar_x + 0.5, 0, 2, 0, head_width=0.2, head_length=0.3, 
                fc=self.colors['blue'], ec=self.colors['blue'])
        
        # Draw scattered wave
        x_scattered = np.linspace(target_x - 0.8, radar_x + 0.5, 100)
        y_scattered = np.zeros_like(x_scattered)
        y_wave_back = wave_amp * np.sin(wave_freq * x_scattered + np.pi)
        
        ax.plot(x_scattered, y_wave_back + 0.5, color=self.colors['red'], linewidth=2, 
                label='Scattered Wave')
        ax.arrow(target_x - 0.8, 0.5, -2, 0, head_width=0.2, head_length=0.3, 
                fc=self.colors['red'], ec=self.colors['red'])
        
        # Add RCS formula
        ax.text(0, 3, r'$\sigma = \lim_{r \to \infty} 4\pi r^2 \left| \frac{E_s}{E_i} \right|^2$', 
                ha='center', fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        
        ax.set_xlim(-7, 7)
        ax.set_ylim(-3, 4)
        ax.set_aspect('equal')
        ax.set_title('Basic RCS Concept', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_sphere_vs_plate(self, ax):
        """Compare RCS patterns of sphere vs flat plate."""
        # Create angle array
        angles = np.linspace(0, 2*np.pi, 360)
        
        # Sphere RCS pattern (isotropic)
        sphere_rcs = np.ones_like(angles)
        
        # Flat plate RCS pattern (directive)
        plate_rcs = np.abs(np.cos(angles - np.pi/2))**4
        plate_rcs[np.abs(angles - np.pi/2) > np.pi/2] = 0.001
        
        # Plot polar patterns
        ax_polar = plt.subplot(111, projection='polar')
        ax_polar.plot(angles, sphere_rcs, color=self.colors['blue'], linewidth=3, 
                     label='Sphere (Isotropic)')
        ax_polar.plot(angles, plate_rcs, color=self.colors['red'], linewidth=3, 
                     label='Flat Plate (Directive)')
        
        ax_polar.set_ylim(0, 1.2)
        ax_polar.set_title('RCS Patterns: Sphere vs Flat Plate', fontsize=14, fontweight='bold')
        ax_polar.legend(loc='upper right')
        ax_polar.grid(True, alpha=0.3)
        
        # Remove the original axes
        ax.remove()
        
    def _plot_faceted_surface(self, ax):
        """Show how faceted surfaces redirect radar energy."""
        # Create a faceted surface
        vertices = np.array([
            [-2, -1], [-1, -1.5], [0, -1], [1, -1.5], [2, -1],
            [2, 1], [1, 1.5], [0, 1], [-1, 1.5], [-2, 1]
        ])
        
        # Draw the faceted surface
        ax.plot(vertices[:, 0], vertices[:, 1], color=self.colors['purple'], 
                linewidth=3, marker='o', markersize=6)
        ax.fill(vertices[:, 0], vertices[:, 1], color=self.colors['purple'], alpha=0.3)
        
        # Draw incident rays
        for i in range(5):
            x_start = -4
            y_start = -2 + i * 1
            x_end = vertices[i, 0]
            y_end = vertices[i, 1]
            
            ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
                    head_width=0.1, head_length=0.2, fc=self.colors['blue'], 
                    ec=self.colors['blue'], alpha=0.7)
        
        # Draw reflected rays (specular reflection)
        for i in range(5):
            # Calculate reflection direction
            edge_vec = vertices[i+1] - vertices[i]
            edge_vec = edge_vec / np.linalg.norm(edge_vec)
            normal = np.array([-edge_vec[1], edge_vec[0]])
            
            # Incident direction
            incident = np.array([1, 0])
            
            # Reflected direction
            reflected = incident - 2 * np.dot(incident, normal) * normal
            
            x_start = vertices[i, 0]
            y_start = vertices[i, 1]
            x_end = x_start + 2 * reflected[0]
            y_end = y_start + 2 * reflected[1]
            
            ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
                    head_width=0.1, head_length=0.2, fc=self.colors['red'], 
                    ec=self.colors['red'], alpha=0.7)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title('Faceted Surface Reflection', fontsize=14, fontweight='bold')
        ax.text(0, -2.5, 'Incident rays redirected away from source', 
                ha='center', fontsize=12, style='italic')
        ax.grid(True, alpha=0.3)
        
    def _plot_rcs_comparison_chart(self, ax):
        """Create the RCS comparison chart from the article."""
        # Data from the article
        objects = ['Small Bird', 'Human', 'F-15 Eagle', 'SR-71 Blackbird', 'F-117 Nighthawk']
        rcs_m2 = [0.01, 1, 12, 10, 0.05]  # Using middle value for F-117
        rcs_dbsm = [10 * np.log10(rcs) for rcs in rcs_m2]
        
        # Create bar chart
        bars = ax.bar(objects, rcs_dbsm, color=[self.colors['green'], self.colors['blue'], 
                                               self.colors['orange'], self.colors['purple'], 
                                               self.colors['red']], alpha=0.8)
        
        # Add value labels on bars
        for bar, val_m2, val_db in zip(bars, rcs_m2, rcs_dbsm):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{val_m2:.2f} m²\n{val_db:.1f} dBsm', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('RCS (dBsm)', fontsize=12, fontweight='bold')
        ax.set_title('Radar Cross Section Comparison', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add detection distance annotation
        ax.text(0.5, 0.95, 'Detection distance ∝ RCS^(1/4)', 
                transform=ax.transAxes, ha='center', va='top', 
                fontsize=12, style='italic', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    def _create_interactive_rcs_concept(self):
        """Create an interactive 3D visualization of RCS concept."""
        # Create a 3D scene
        fig = go.Figure()
        
        # Add sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, 
                                colorscale='Blues', opacity=0.7, name='Sphere'))
        
        # Add incident wave vectors
        n_rays = 20
        for i in range(n_rays):
            angle = 2 * np.pi * i / n_rays
            x_start = 3 * np.cos(angle)
            y_start = 3 * np.sin(angle)
            z_start = 0
            
            fig.add_trace(go.Scatter3d(
                x=[x_start, 0], y=[y_start, 0], z=[z_start, 0],
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=3),
                name='Incident Wave' if i == 0 else None,
                showlegend=True if i == 0 else False
            ))
        
        # Add scattered wave vectors
        for i in range(n_rays):
            angle = 2 * np.pi * i / n_rays
            x_end = 3 * np.cos(angle)
            y_end = 3 * np.sin(angle)
            z_end = 0
            
            fig.add_trace(go.Scatter3d(
                x=[0, x_end], y=[0, y_end], z=[0, z_end],
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=2),
                name='Scattered Wave' if i == 0 else None,
                showlegend=True if i == 0 else False
            ))
        
        fig.update_layout(
            title='Interactive RCS Concept - Sphere Scattering',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        fig.write_html(f'{self.output_dir}/1_rcs_concept_interactive.html')
        
    def rcs_equation_breakdown(self):
        """
        Visualize the components of the RCS equation.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distance scaling (r^2 effect)
        distances = np.linspace(1, 20, 100)
        field_strength = 1 / distances**2
        
        ax1.plot(distances, field_strength, color=self.colors['blue'], linewidth=3)
        ax1.set_xlabel('Distance (r)', fontsize=12)
        ax1.set_ylabel('Field Strength (∝ 1/r²)', fontsize=12)
        ax1.set_title('Distance Scaling in RCS Formula', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.text(0.5, 0.8, r'$E \propto \frac{1}{r}$, Power $\propto \frac{1}{r^2}$', 
                transform=ax1.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        
        # 2. Scattered vs Incident field ratio
        incident_angles = np.linspace(0, 180, 180)
        # Simulate different scattering patterns
        sphere_ratio = np.ones_like(incident_angles)  # Isotropic
        plate_ratio = np.abs(np.cos(np.deg2rad(incident_angles)))**2
        
        ax2.plot(incident_angles, sphere_ratio, color=self.colors['blue'], 
                linewidth=3, label='Sphere')
        ax2.plot(incident_angles, plate_ratio, color=self.colors['red'], 
                linewidth=3, label='Flat Plate')
        ax2.set_xlabel('Angle (degrees)', fontsize=12)
        ax2.set_ylabel('|Es/Ei|²', fontsize=12)
        ax2.set_title('Scattered to Incident Field Ratio', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Effective area concept
        actual_areas = np.array([0.01, 1, 12, 10, 0.05])  # From the article
        objects = ['Bird', 'Human', 'F-15', 'SR-71', 'F-117']
        
        bars = ax3.bar(objects, actual_areas, color=[self.colors['green'], self.colors['blue'], 
                                                    self.colors['orange'], self.colors['purple'], 
                                                    self.colors['red']], alpha=0.8)
        ax3.set_ylabel('Effective Area (m²)', fontsize=12)
        ax3.set_title('RCS as Effective Area', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Add annotations
        for bar, val in zip(bars, actual_areas):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{val:.2f} m²', ha='center', va='bottom', fontsize=10)
        
        # 4. Complete RCS equation visualization
        ax4.text(0.5, 0.8, r'$\sigma = \lim_{r \to \infty} 4\pi r^2 \left| \frac{E_s}{E_i} \right|^2$',
                ha='center', va='center', transform=ax4.transAxes, fontsize=20, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['yellow'], alpha=0.8))
        
        # Break down the equation
        ax4.text(0.5, 0.6, r'$4\pi r^2$ = Surface area of sphere at distance r',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.5, 0.5, r'$\left| \frac{E_s}{E_i} \right|^2$ = Power ratio (scattered/incident)',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.5, 0.4, r'$\sigma$ = Effective scattering area (m²)',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        ax4.text(0.5, 0.2, 'Think of it as: "What size metal sphere would\nscatter the same amount of energy?"',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12, 
                style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('RCS Equation Breakdown', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/2_rcs_equation_breakdown.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def detection_distance_visualization(self):
        """
        Visualize the fourth root relationship between RCS and detection distance.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # RCS values and corresponding detection distances
        rcs_values = np.logspace(-3, 2, 100)  # From 0.001 to 100 m²
        detection_distance = rcs_values**(1/4)  # Fourth root relationship
        
        # Plot 1: RCS vs Detection Distance
        ax1.loglog(rcs_values, detection_distance, color=self.colors['blue'], linewidth=3)
        ax1.set_xlabel('RCS (m²)', fontsize=12)
        ax1.set_ylabel('Relative Detection Distance', fontsize=12)
        ax1.set_title('Detection Distance vs RCS\n(Fourth Root Relationship)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add specific aircraft points
        aircraft_rcs = [0.01, 1, 12, 10, 0.05]  # From article
        aircraft_names = ['Small Bird', 'Human', 'F-15', 'SR-71', 'F-117']
        aircraft_distances = [rcs**(1/4) for rcs in aircraft_rcs]
        
        for i, (rcs, dist, name) in enumerate(zip(aircraft_rcs, aircraft_distances, aircraft_names)):
            ax1.scatter(rcs, dist, s=100, color=self.colors['red'], zorder=5)
            ax1.annotate(name, (rcs, dist), xytext=(10, 10), textcoords='offset points',
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add formula
        ax1.text(0.02, 0.95, r'$R_{detection} \propto \sigma^{1/4}$', 
                transform=ax1.transAxes, fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # Plot 2: Impact of RCS reduction
        base_rcs = 10  # SR-71 baseline
        rcs_reductions = np.array([1, 10, 100, 1000, 10000])
        reduced_rcs = base_rcs / rcs_reductions
        distance_ratios = (reduced_rcs / base_rcs)**(1/4)
        
        bars = ax2.bar(range(len(rcs_reductions)), distance_ratios, 
                      color=[self.colors['blue'], self.colors['green'], self.colors['orange'], 
                             self.colors['purple'], self.colors['red']], alpha=0.8)
        
        ax2.set_xlabel('RCS Reduction Factor', fontsize=12)
        ax2.set_ylabel('Detection Distance Ratio', fontsize=12)
        ax2.set_title('Impact of RCS Reduction\non Detection Distance', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(rcs_reductions)))
        ax2.set_xticklabels([f'{r}×' for r in rcs_reductions])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for bar, ratio in zip(bars, distance_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{ratio:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add insight box
        ax2.text(0.5, 0.8, 'Cutting RCS by 100×\nonly reduces detection\ndistance to 56% of original', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/3_detection_distance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def physical_optics_demonstration(self):
        """
        Demonstrate Physical Optics approximation vs exact solution.
        Shows surface currents and the key assumptions of PO.
        """
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. PO Assumptions
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_po_assumptions(ax1)
        
        # 2. Surface Currents
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_surface_currents(ax2)
        
        # 3. Illuminated vs Shadowed regions
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_illuminated_regions(ax3)
        
        # 4. PO vs Exact Solution Comparison
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_po_vs_exact(ax4)
        
        # 5. Computational Complexity
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_computational_complexity(ax5)
        
        # 6. PO Integral Visualization
        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_po_integral(ax6)
        
        plt.suptitle('Physical Optics (PO) Approximation', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/4_physical_optics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create interactive PO integral visualization
        self._create_interactive_po_integral()
        
    def _plot_po_assumptions(self, ax):
        """Plot the key assumptions of Physical Optics."""
        ax.text(0.5, 0.9, 'Physical Optics Assumptions', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        assumptions = [
            '1. Perfect Electrical Conductor (PEC)',
            '2. Only illuminated surfaces contribute',
            '3. Specular reflection only',
            '4. No multiple reflections',
            '5. High frequency approximation'
        ]
        
        for i, assumption in enumerate(assumptions):
            ax.text(0.1, 0.75 - i*0.12, assumption, ha='left', va='center', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['blue'], alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    def _plot_surface_currents(self, ax):
        """Visualize surface currents on a conducting surface."""
        # Create a simple conducting surface
        x = np.linspace(-2, 2, 100)
        y = np.sin(x)
        
        ax.plot(x, y, color=self.colors['dark_blue'], linewidth=3, label='Conducting Surface')
        ax.fill_between(x, y, y-0.2, color=self.colors['blue'], alpha=0.3)
        
        # Add surface current arrows
        x_arrows = np.linspace(-1.5, 1.5, 8)
        y_arrows = np.sin(x_arrows)
        
        # Normal vectors
        dx = x_arrows[1] - x_arrows[0]
        dy_dx = np.cos(x_arrows)
        normals = np.array([-dy_dx, np.ones_like(dy_dx)]).T
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Surface current vectors (perpendicular to normal)
        currents = np.array([normals[:, 1], -normals[:, 0]]).T
        
        for i, (x_pos, y_pos, curr) in enumerate(zip(x_arrows, y_arrows, currents)):
            ax.arrow(x_pos, y_pos, curr[0]*0.3, curr[1]*0.3, 
                    head_width=0.05, head_length=0.05, 
                    fc=self.colors['red'], ec=self.colors['red'])
        
        # Add incident field
        ax.arrow(-3, 1, 1, -0.5, head_width=0.1, head_length=0.1, 
                fc=self.colors['green'], ec=self.colors['green'], linewidth=2)
        ax.text(-3, 1.2, r'$\vec{H}_i$', fontsize=12, color=self.colors['green'])
        
        # Add PO formula
        ax.text(0.5, 0.9, r'$\vec{J}_s = 2\hat{n} \times \vec{H}_i$', 
                transform=ax.transAxes, ha='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-1, 2)
        ax.set_title('Surface Currents on PEC', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    def _plot_illuminated_regions(self, ax):
        """Show illuminated vs shadowed regions."""
        # Create a 3D-like object (cylinder cross-section)
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        
        # Draw the object
        ax.plot(x_circle, y_circle, color=self.colors['dark_blue'], linewidth=3)
        
        # Incident direction
        incident_angle = np.pi/4  # 45 degrees
        ki_x = np.cos(incident_angle)
        ki_y = np.sin(incident_angle)
        
        # Determine illuminated region
        illuminated = np.dot(np.array([x_circle, y_circle]).T, np.array([ki_x, ki_y])) > 0
        
        # Fill illuminated region
        ax.fill(x_circle[illuminated], y_circle[illuminated], 
                color=self.colors['yellow'], alpha=0.6, label='Illuminated')
        
        # Fill shadowed region
        ax.fill(x_circle[~illuminated], y_circle[~illuminated], 
                color=self.colors['purple'], alpha=0.6, label='Shadowed')
        
        # Add incident rays
        for angle in np.linspace(0, 2*np.pi, 16):
            x_start = 2.5 * np.cos(angle + incident_angle)
            y_start = 2.5 * np.sin(angle + incident_angle)
            x_end = np.cos(angle)
            y_end = np.sin(angle)
            
            # Only draw if hitting illuminated side
            if np.dot(np.array([x_end, y_end]), np.array([ki_x, ki_y])) > 0:
                ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
                        head_width=0.05, head_length=0.05, 
                        fc=self.colors['blue'], ec=self.colors['blue'], alpha=0.7)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title('Illuminated vs Shadowed Regions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_po_vs_exact(self, ax):
        """Compare PO approximation with exact solution."""
        # Simulate RCS patterns for a sphere
        angles = np.linspace(0, 180, 180)
        
        # Exact solution (Mie scattering for large sphere)
        exact_rcs = np.ones_like(angles)  # Simplified - actual Mie is more complex
        
        # PO approximation
        po_rcs = np.abs(np.cos(np.deg2rad(angles)))**2
        po_rcs[angles > 90] = 0  # No backscattering from shadowed side
        
        # Plot both
        ax.plot(angles, exact_rcs, color=self.colors['blue'], linewidth=3, 
                label='Exact Solution', linestyle='-')
        ax.plot(angles, po_rcs, color=self.colors['red'], linewidth=3, 
                label='PO Approximation', linestyle='--')
        
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Normalized RCS', fontsize=12)
        ax.set_title('PO vs Exact Solution Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.annotate('PO misses edge effects', xy=(90, 0), xytext=(120, 0.3),
                   arrowprops=dict(arrowstyle='->', color=self.colors['red']),
                   fontsize=12, color=self.colors['red'])
        
        ax.annotate('Good agreement in forward direction', xy=(30, 0.7), xytext=(50, 0.9),
                   arrowprops=dict(arrowstyle='->', color=self.colors['green']),
                   fontsize=12, color=self.colors['green'])
        
    def _plot_computational_complexity(self, ax):
        """Show computational complexity of different methods."""
        n_vertices = np.logspace(2, 4, 50)  # 100 to 10,000 vertices
        
        # Exact solution: O(N^2) for Green's function
        exact_complexity = n_vertices**2
        
        # PO approximation: O(N)
        po_complexity = n_vertices
        
        ax.loglog(n_vertices, exact_complexity, color=self.colors['blue'], 
                 linewidth=3, label='Exact Solution O(N²)')
        ax.loglog(n_vertices, po_complexity, color=self.colors['red'], 
                 linewidth=3, label='PO Approximation O(N)')
        
        ax.set_xlabel('Number of Surface Elements', fontsize=12)
        ax.set_ylabel('Computational Cost', fontsize=12)
        ax.set_title('Computational Complexity', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add memory annotation
        ax.text(0.5, 0.8, '1970s computers had ~1MB memory\n1000 vertices → 9MB for exact solution!', 
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
    def _plot_po_integral(self, ax):
        """Visualize the PO integral over the surface."""
        # Create a sample surface
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-1, 1, 25)
        X, Y = np.meshgrid(x, y)
        Z = 0.2 * np.sin(2*X) * np.cos(2*Y)
        
        # Plot surface
        im = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        
        # Add PO integral formula
        ax.text(0.5, 0.95, r'$\vec{E}_s = \frac{jk\eta}{4\pi} \int_{S_{lit}} \left[\hat{k}_s \times (\hat{k}_s \times \vec{J}_s)\right] e^{jk(\hat{k}_i - \hat{k}_s) \cdot \vec{r}} dS$',
                transform=ax.transAxes, ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # Add sample integration points
        n_points = 10
        x_points = np.random.uniform(-2, 2, n_points)
        y_points = np.random.uniform(-1, 1, n_points)
        
        for i, (xi, yi) in enumerate(zip(x_points, y_points)):
            ax.plot(xi, yi, 'ro', markersize=8)
            if i < 3:  # Show phase for first few points
                phase = np.random.uniform(0, 2*np.pi)
                ax.annotate(f'φ={phase:.1f}', (xi, yi), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('PO Integral: Phase Contributions from Surface Patches', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Surface Height')
        
    def _create_interactive_po_integral(self):
        """Create an interactive visualization of the PO integral."""
        # Create a simple surface with controllable parameters
        fig = go.Figure()
        
        # Create surface mesh
        x = np.linspace(-2, 2, 30)
        y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x, y)
        Z = 0.3 * np.sin(2*X) * np.cos(2*Y)
        
        # Add surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.8,
            name='Surface'
        ))
        
        # Add sample integration points
        n_points = 20
        x_points = np.random.uniform(-2, 2, n_points)
        y_points = np.random.uniform(-2, 2, n_points)
        z_points = 0.3 * np.sin(2*x_points) * np.cos(2*y_points)
        
        # Calculate phase contributions
        k = 20  # Wave number
        phases = k * (x_points + y_points)  # Simplified phase calculation
        
        fig.add_trace(go.Scatter3d(
            x=x_points, y=y_points, z=z_points,
            mode='markers',
            marker=dict(
                size=8,
                color=phases,
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title='Phase')
            ),
            name='Integration Points'
        ))
        
        fig.update_layout(
            title='Interactive PO Integral - Phase Contributions',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        fig.write_html(f'{self.output_dir}/4_po_integral_interactive.html')
        
    def surface_current_visualization(self):
        """
        Detailed visualization of surface currents and their effects.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Surface current formula derivation
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_surface_current_derivation(ax1)
        
        # 2. Current distribution on different shapes
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_current_distribution(ax2)
        
        # 3. Boundary conditions
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_boundary_conditions(ax3)
        
        # 4. Scattered field from currents
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_scattered_field_from_currents(ax4)
        
        plt.suptitle('Surface Currents and Electromagnetic Scattering', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/5_surface_currents.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_surface_current_derivation(self, ax):
        """Show the derivation of surface current formula."""
        ax.text(0.5, 0.9, 'Surface Current Derivation', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        steps = [
            r'1. Tangential E-field = 0 on PEC',
            r'2. $\vec{E}_t = \vec{E}_i + \vec{E}_r = 0$',
            r'3. $\vec{E}_r = -\vec{E}_i$',
            r'4. $\vec{H}_r = \vec{H}_i$ (at surface)',
            r'5. $\vec{J}_s = \hat{n} \times (\vec{H}_r - \vec{H}_i)$',
            r'6. $\vec{J}_s = 2\hat{n} \times \vec{H}_i$'
        ]
        
        for i, step in enumerate(steps):
            ax.text(0.1, 0.75 - i*0.1, step, ha='left', va='center',
                   transform=ax.transAxes, fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    def _plot_current_distribution(self, ax):
        """Show current distribution on different shapes."""
        # Create a flat plate
        x_plate = np.linspace(-1, 1, 20)
        y_plate = np.zeros_like(x_plate)
        
        # Current vectors (parallel to plate)
        for i, (x, y) in enumerate(zip(x_plate, y_plate)):
            if i % 2 == 0:  # Show every other arrow
                ax.arrow(x, y, 0, 0.3, head_width=0.05, head_length=0.05,
                        fc=self.colors['red'], ec=self.colors['red'])
        
        ax.plot(x_plate, y_plate, color=self.colors['blue'], linewidth=4, label='Flat Plate')
        
        # Add cylinder
        theta = np.linspace(0, np.pi, 20)  # Upper half
        x_cyl = np.cos(theta)
        y_cyl = np.sin(theta) + 1.5
        
        # Current vectors (tangential to cylinder)
        for i, (x, y, th) in enumerate(zip(x_cyl, y_cyl, theta)):
            if i % 2 == 0:
                # Tangent direction
                tx = -np.sin(th)
                ty = np.cos(th)
                ax.arrow(x, y, tx*0.2, ty*0.2, head_width=0.05, head_length=0.05,
                        fc=self.colors['green'], ec=self.colors['green'])
        
        ax.plot(x_cyl, y_cyl, color=self.colors['blue'], linewidth=4, label='Cylinder')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title('Current Distribution on Different Shapes', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_boundary_conditions(self, ax):
        """Visualize boundary conditions at conductor surface."""
        # Draw conductor surface
        ax.fill_between([-2, 2], [-0.5, -0.5], [0, 0], color=self.colors['blue'], alpha=0.3, label='Conductor')
        ax.plot([-2, 2], [0, 0], color=self.colors['blue'], linewidth=3)
        
        # Electric field lines (terminated at surface)
        for x in np.linspace(-1.5, 1.5, 8):
            # Above surface
            ax.arrow(x, 0.5, 0, -0.4, head_width=0.05, head_length=0.05,
                    fc=self.colors['red'], ec=self.colors['red'])
            ax.text(x, 0.6, 'E', ha='center', fontsize=10, color=self.colors['red'])
        
        # Magnetic field lines (parallel to surface)
        for y in [0.3, 0.7, 1.0]:
            ax.arrow(-1.5, y, 3, 0, head_width=0.05, head_length=0.1,
                    fc=self.colors['green'], ec=self.colors['green'])
            ax.text(-1.7, y, 'H', ha='center', fontsize=10, color=self.colors['green'])
        
        # Surface current
        ax.arrow(-1, 0, 2, 0, head_width=0.08, head_length=0.1,
                fc=self.colors['purple'], ec=self.colors['purple'], linewidth=3)
        ax.text(0, -0.2, r'$\vec{J}_s$', ha='center', fontsize=12, color=self.colors['purple'])
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.7, 1.3)
        ax.set_title('Boundary Conditions at PEC Surface', fontsize=12, fontweight='bold')
        ax.text(0, 1.1, r'$\vec{E}_t = 0$, $\vec{H}_t$ continuous', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        ax.grid(True, alpha=0.3)
        
    def _plot_scattered_field_from_currents(self, ax):
        """Show how surface currents generate scattered fields."""
        # Create a conducting surface
        x_surf = np.linspace(-2, 2, 100)
        y_surf = 0.2 * np.sin(2*x_surf)
        
        ax.plot(x_surf, y_surf, color=self.colors['blue'], linewidth=3, label='Conducting Surface')
        ax.fill_between(x_surf, y_surf, y_surf-0.1, color=self.colors['blue'], alpha=0.3)
        
        # Add current elements
        x_currents = np.linspace(-1.5, 1.5, 8)
        y_currents = 0.2 * np.sin(2*x_currents)
        
        for i, (x, y) in enumerate(zip(x_currents, y_currents)):
            # Current direction (tangential)
            dx = 0.2
            dy = 0.2 * 2 * np.cos(2*x) * dx
            norm = np.sqrt(dx**2 + dy**2)
            dx, dy = dx/norm, dy/norm
            
            ax.arrow(x, y, dx*0.2, dy*0.2, head_width=0.05, head_length=0.05,
                    fc=self.colors['red'], ec=self.colors['red'])
            
            # Radiated field (perpendicular to current)
            rad_x, rad_y = -dy, dx
            ax.arrow(x, y, rad_x*0.8, rad_y*0.8, head_width=0.03, head_length=0.03,
                    fc=self.colors['green'], ec=self.colors['green'], alpha=0.7)
        
        # Add scattered field pattern
        angles = np.linspace(0, 2*np.pi, 100)
        r_pattern = 1 + 0.5 * np.cos(2*angles)  # Simplified pattern
        x_pattern = r_pattern * np.cos(angles)
        y_pattern = r_pattern * np.sin(angles) + 1
        
        ax.plot(x_pattern, y_pattern, color=self.colors['purple'], linewidth=2, 
                linestyle='--', label='Scattered Field Pattern')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        ax.set_title('Scattered Field Generation from Surface Currents', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add equation
        ax.text(0.5, 0.95, r'$\vec{E}_s = \frac{jk\eta}{4\pi} \int \vec{J}_s \frac{e^{jkR}}{R} dS$',
                transform=ax.transAxes, ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))

    def faceted_vs_curved_comparison(self):
        """
        Compare RCS patterns between faceted and curved surfaces.
        Demonstrates the 'disco ball effect' of curved surfaces.
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Disco ball effect visualization
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_disco_ball_effect(ax1)
        
        # 2. Faceted surface reflection
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_faceted_reflection(ax2)
        
        # 3. RCS pattern comparison
        ax3 = fig.add_subplot(gs[0, 2], projection='polar')
        self._plot_rcs_pattern_comparison(ax3)
        
        # 4. Curvature effect on reflection
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_curvature_effect(ax4)
        
        # 5. F-117 geometry analysis
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_f117_geometry(ax5)
        
        # 6. Stealth design principles
        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_stealth_design_principles(ax6)
        
        plt.suptitle('Faceted vs Curved Surfaces: The Science of Stealth Shaping', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/6_faceted_vs_curved.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create 3D interactive comparison
        self._create_3d_surface_comparison()
        
    def _plot_disco_ball_effect(self, ax):
        """Demonstrate why curved surfaces are bad for stealth."""
        # Create a sphere
        theta = np.linspace(0, 2*np.pi, 100)
        r = 1
        x_sphere = r * np.cos(theta)
        y_sphere = r * np.sin(theta)
        
        ax.plot(x_sphere, y_sphere, color=self.colors['blue'], linewidth=3, label='Curved Surface')
        ax.fill(x_sphere, y_sphere, color=self.colors['blue'], alpha=0.2)
        
        # Add incident rays from various angles
        n_rays = 12
        for i in range(n_rays):
            angle = 2 * np.pi * i / n_rays
            
            # Incident ray
            x_start = 2.5 * np.cos(angle)
            y_start = 2.5 * np.sin(angle)
            x_contact = r * np.cos(angle)
            y_contact = r * np.sin(angle)
            
            # Draw incident ray
            ax.arrow(x_start, y_start, x_contact - x_start, y_contact - y_start,
                    head_width=0.05, head_length=0.05, fc=self.colors['green'], 
                    ec=self.colors['green'], alpha=0.7)
            
            # Reflected ray (specular reflection)
            # Normal at contact point
            normal_x, normal_y = np.cos(angle), np.sin(angle)
            # Incident direction
            incident_x = x_contact - x_start
            incident_y = y_contact - y_start
            incident_mag = np.sqrt(incident_x**2 + incident_y**2)
            incident_x /= incident_mag
            incident_y /= incident_mag
            
            # Reflected direction
            dot_product = incident_x * normal_x + incident_y * normal_y
            reflected_x = incident_x - 2 * dot_product * normal_x
            reflected_y = incident_y - 2 * dot_product * normal_y
            
            # Draw reflected ray
            ax.arrow(x_contact, y_contact, reflected_x * 1.5, reflected_y * 1.5,
                    head_width=0.05, head_length=0.05, fc=self.colors['red'], 
                    ec=self.colors['red'], alpha=0.7)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title('Curved Surface: "Disco Ball Effect"', fontsize=14, fontweight='bold')
        ax.text(0, -2.5, 'Energy scattered in all directions', ha='center', fontsize=12, 
                style='italic', color=self.colors['red'])
        ax.grid(True, alpha=0.3)
        
    def _plot_faceted_reflection(self, ax):
        """Show how faceted surfaces direct energy away from source."""
        # Create a faceted surface (diamond-like shape)
        vertices = np.array([
            [-1.5, 0], [-0.5, 1], [0.5, 1], [1.5, 0],
            [0.5, -1], [-0.5, -1], [-1.5, 0]
        ])
        
        ax.plot(vertices[:, 0], vertices[:, 1], color=self.colors['purple'], 
                linewidth=3, marker='o', markersize=8, label='Faceted Surface')
        ax.fill(vertices[:, 0], vertices[:, 1], color=self.colors['purple'], alpha=0.3)
        
        # Add incident rays
        n_rays = 6
        incident_angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
        
        for i, angle in enumerate(incident_angles):
            # Find the closest facet
            facet_idx = i % (len(vertices) - 1)
            x_contact = (vertices[facet_idx, 0] + vertices[facet_idx+1, 0]) / 2
            y_contact = (vertices[facet_idx, 1] + vertices[facet_idx+1, 1]) / 2
            
            # Incident ray
            x_start = x_contact - 2 * np.cos(angle)
            y_start = y_contact - 2 * np.sin(angle)
            
            ax.arrow(x_start, y_start, x_contact - x_start, y_contact - y_start,
                    head_width=0.05, head_length=0.05, fc=self.colors['green'], 
                    ec=self.colors['green'], alpha=0.7)
            
            # Facet normal
            facet_vec = vertices[facet_idx+1] - vertices[facet_idx]
            normal_vec = np.array([-facet_vec[1], facet_vec[0]])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            
            # Incident direction
            incident_vec = np.array([np.cos(angle), np.sin(angle)])
            
            # Reflected direction
            reflected_vec = incident_vec - 2 * np.dot(incident_vec, normal_vec) * normal_vec
            
            # Draw reflected ray
            ax.arrow(x_contact, y_contact, reflected_vec[0] * 1.5, reflected_vec[1] * 1.5,
                    head_width=0.05, head_length=0.05, fc=self.colors['red'], 
                    ec=self.colors['red'], alpha=0.7)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title('Faceted Surface: Controlled Reflection', fontsize=14, fontweight='bold')
        ax.text(0, -2.5, 'Energy directed away from source', ha='center', fontsize=12, 
                style='italic', color=self.colors['green'])
        ax.grid(True, alpha=0.3)
        
    def _plot_rcs_pattern_comparison(self, ax):
        """Compare RCS patterns in polar coordinates."""
        angles = np.linspace(0, 2*np.pi, 360)
        
        # Sphere pattern (isotropic)
        sphere_pattern = np.ones_like(angles)
        
        # Faceted surface pattern (highly directional)
        faceted_pattern = np.zeros_like(angles)
        # Add discrete lobes for each facet
        for i in range(6):  # 6 facets
            center_angle = i * np.pi / 3
            lobe_angles = np.abs(angles - center_angle)
            lobe_angles = np.minimum(lobe_angles, 2*np.pi - lobe_angles)
            faceted_pattern += 0.5 * np.exp(-10 * lobe_angles**2)
        
        ax.plot(angles, sphere_pattern, color=self.colors['blue'], linewidth=3, 
                label='Sphere (Isotropic)')
        ax.plot(angles, faceted_pattern, color=self.colors['red'], linewidth=3, 
                label='Faceted (Directional)')
        
        ax.set_ylim(0, 1.5)
        ax.set_title('RCS Pattern Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_curvature_effect(self, ax):
        """Show how local curvature affects scattering."""
        # Create surfaces with different curvatures
        x = np.linspace(-2, 2, 100)
        
        # Flat surface
        y_flat = np.zeros_like(x)
        
        # Slightly curved surface
        y_curved = 0.1 * x**2
        
        # Highly curved surface
        y_highly_curved = 0.5 * x**2
        
        ax.plot(x, y_flat, color=self.colors['blue'], linewidth=3, label='Flat Surface')
        ax.plot(x, y_curved + 1, color=self.colors['green'], linewidth=3, label='Slightly Curved')
        ax.plot(x, y_highly_curved + 2, color=self.colors['red'], linewidth=3, label='Highly Curved')
        
        # Add incident rays
        for surface_y, color in [(0, self.colors['blue']), (1, self.colors['green']), (2, self.colors['red'])]:
            for xi in [-1, 0, 1]:
                # Incident ray
                ax.arrow(xi, surface_y + 2, 0, -1.5, head_width=0.05, head_length=0.05,
                        fc=color, ec=color, alpha=0.5)
                
                # Reflected ray direction depends on local curvature
                if surface_y == 0:  # Flat
                    ax.arrow(xi, surface_y, 0, 1.5, head_width=0.05, head_length=0.05,
                            fc=color, ec=color, alpha=0.5)
                else:  # Curved
                    # Local slope
                    if surface_y == 1:  # Slightly curved
                        slope = 0.2 * xi
                    else:  # Highly curved
                        slope = xi
                    
                    # Reflection angle based on slope
                    reflect_angle = np.arctan(slope)
                    ax.arrow(xi, surface_y + 0.1*xi**2 if surface_y == 1 else 0.5*xi**2, 
                            1.5*np.sin(reflect_angle), 1.5*np.cos(reflect_angle),
                            head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.5)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.5, 4)
        ax.set_title('Effect of Surface Curvature on Scattering', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add mathematical note
        ax.text(0.5, 0.95, r'Higher curvature → More scattered directions', 
                transform=ax.transAxes, ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
    def _plot_f117_geometry(self, ax):
        """Show F-117 geometry and design principles."""
        # Simplified F-117 cross-section
        vertices = np.array([
            [0, 0.5], [0.8, 0.3], [1.2, 0], [0.8, -0.3], [0, -0.5],
            [-0.8, -0.3], [-1.2, 0], [-0.8, 0.3], [0, 0.5]
        ])
        
        ax.plot(vertices[:, 0], vertices[:, 1], color=self.colors['dark_blue'], 
                linewidth=4, label='F-117 Cross-section')
        ax.fill(vertices[:, 0], vertices[:, 1], color=self.colors['dark_blue'], alpha=0.3)
        
        # Add facet normals
        for i in range(len(vertices) - 1):
            mid_x = (vertices[i, 0] + vertices[i+1, 0]) / 2
            mid_y = (vertices[i, 1] + vertices[i+1, 1]) / 2
            
            # Facet vector
            facet_vec = vertices[i+1] - vertices[i]
            # Normal vector
            normal = np.array([-facet_vec[1], facet_vec[0]])
            normal = normal / np.linalg.norm(normal)
            
            ax.arrow(mid_x, mid_y, normal[0] * 0.2, normal[1] * 0.2,
                    head_width=0.02, head_length=0.02, fc=self.colors['red'], 
                    ec=self.colors['red'], alpha=0.7)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_title('F-117 Faceted Design', fontsize=14, fontweight='bold')
        ax.text(0, -0.8, 'All facets angled to deflect radar', ha='center', fontsize=12, 
                style='italic')
        ax.grid(True, alpha=0.3)
        
    def _plot_stealth_design_principles(self, ax):
        """Illustrate key stealth design principles."""
        # Create a comparison chart
        principles = [
            'Faceted Surfaces',
            'Eliminate Curves',
            'Radar-Absorbing Materials',
            'Edge Alignment',
            'Minimize Cavities',
            'Angled Surfaces'
        ]
        
        effectiveness = [0.9, 0.8, 0.7, 0.85, 0.75, 0.95]
        complexity = [0.6, 0.4, 0.8, 0.7, 0.5, 0.3]
        
        x = np.arange(len(principles))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, effectiveness, width, label='Effectiveness', 
                      color=self.colors['green'], alpha=0.8)
        bars2 = ax.bar(x + width/2, complexity, width, label='Implementation Complexity', 
                      color=self.colors['red'], alpha=0.8)
        
        ax.set_xlabel('Stealth Design Principles', fontsize=12)
        ax.set_ylabel('Relative Score', fontsize=12)
        ax.set_title('Stealth Design Principles Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(principles, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
    def _create_3d_surface_comparison(self):
        """Create 3D interactive comparison of surface types."""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Curved Surface (Bad for Stealth)', 'Faceted Surface (Good for Stealth)')
        )
        
        # Create curved surface (sphere)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(
            go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, 
                      colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # Create faceted surface (diamond)
        vertices = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        
        faces = [
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            [0, 3, 5], [3, 2, 5], [2, 1, 5], [1, 0, 5]
        ]
        
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                color='red', opacity=0.7, showscale=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='3D Surface Comparison: Curved vs Faceted',
            scene=dict(aspectmode='cube'),
            scene2=dict(aspectmode='cube'),
            width=1000,
            height=500
        )
        
        fig.write_html(f'{self.output_dir}/6_surface_comparison_3d.html')
        
    def stealth_evolution_timeline(self):
        """
        Create a visualization showing the evolution of stealth technology.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Timeline data
        aircraft_data = {
            'SR-71 Blackbird': {'year': 1964, 'rcs_m2': 10, 'rcs_dbsm': 10, 'technology': 'Speed + Altitude'},
            'F-117 Nighthawk': {'year': 1981, 'rcs_m2': 0.05, 'rcs_dbsm': -13, 'technology': 'Faceted Shaping'},
            'B-2 Spirit': {'year': 1989, 'rcs_m2': 0.0001, 'rcs_dbsm': -40, 'technology': 'Flying Wing + RAM'},
            'F-22 Raptor': {'year': 1997, 'rcs_m2': 0.0001, 'rcs_dbsm': -40, 'technology': 'Curved Stealth'},
            'F-35 Lightning II': {'year': 2006, 'rcs_m2': 0.001, 'rcs_dbsm': -30, 'technology': 'Multi-role Stealth'}
        }
        
        # Plot 1: RCS evolution over time
        years = [data['year'] for data in aircraft_data.values()]
        rcs_values = [data['rcs_m2'] for data in aircraft_data.values()]
        names = list(aircraft_data.keys())
        
        ax1.semilogy(years, rcs_values, 'o-', linewidth=3, markersize=10, 
                    color=self.colors['blue'], label='RCS Evolution')
        
        # Add aircraft labels
        for i, (year, rcs, name) in enumerate(zip(years, rcs_values, names)):
            ax1.annotate(name, (year, rcs), xytext=(10, 10), textcoords='offset points',
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('RCS (m²)', fontsize=12)
        ax1.set_title('Evolution of Stealth Aircraft RCS', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add improvement annotations
        ax1.annotate('', xy=(1981, 0.05), xytext=(1964, 10),
                    arrowprops=dict(arrowstyle='->', color=self.colors['red'], lw=2))
        ax1.text(1972, 1, '200× RCS\nReduction', ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # Plot 2: Technology comparison
        technologies = [data['technology'] for data in aircraft_data.values()]
        rcs_dbsm = [data['rcs_dbsm'] for data in aircraft_data.values()]
        
        bars = ax2.bar(names, rcs_dbsm, color=[self.colors['purple'], self.colors['red'], 
                                              self.colors['green'], self.colors['blue'], 
                                              self.colors['orange']], alpha=0.8)
        
        ax2.set_ylabel('RCS (dBsm)', fontsize=12)
        ax2.set_title('Stealth Technology Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add technology labels
        for bar, tech in zip(bars, technologies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height - 2,
                   tech, ha='center', va='top', fontsize=10, rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Rotate x-axis labels
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/7_stealth_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create interactive timeline
        self._create_interactive_timeline(aircraft_data)
        
    def _create_interactive_timeline(self, aircraft_data):
        """Create an interactive timeline of stealth evolution."""
        fig = go.Figure()
        
        years = [data['year'] for data in aircraft_data.values()]
        rcs_values = [data['rcs_m2'] for data in aircraft_data.values()]
        names = list(aircraft_data.keys())
        technologies = [data['technology'] for data in aircraft_data.values()]
        
        # Add the main timeline
        fig.add_trace(go.Scatter(
            x=years, y=rcs_values,
            mode='lines+markers',
            name='RCS Evolution',
            line=dict(color='blue', width=3),
            marker=dict(size=12, color='red'),
            hovertemplate='<b>%{text}</b><br>' +
                         'Year: %{x}<br>' +
                         'RCS: %{y:.4f} m²<br>' +
                         'Technology: %{customdata}<br>' +
                         '<extra></extra>',
            text=names,
            customdata=technologies
        ))
        
        # Add annotations for each aircraft
        for i, (year, rcs, name, tech) in enumerate(zip(years, rcs_values, names, technologies)):
            fig.add_annotation(
                x=year, y=rcs,
                text=f"{name}<br>{tech}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
                xshift=20 if i % 2 == 0 else -20,
                yshift=20 if i % 2 == 0 else -20
            )
        
        fig.update_layout(
            title='Interactive Stealth Aircraft Evolution Timeline',
            xaxis_title='Year',
            yaxis_title='RCS (m²)',
            yaxis_type='log',
            width=1000,
            height=600,
            showlegend=False
        )
        
        fig.write_html(f'{self.output_dir}/7_stealth_evolution_interactive.html')
        
    def create_complete_visualization_suite(self):
        """
        Create the complete suite of visualizations for the stealth article.
        """
        print("🚀 Creating Comprehensive Stealth Mathematics Visualizations")
        print("=" * 60)
        
        try:
            print("1. 📊 RCS Concept Visualization...")
            self.rcs_concept_visualization()
            
            print("2. 🧮 RCS Equation Breakdown...")
            self.rcs_equation_breakdown()
            
            print("3. 📏 Detection Distance Analysis...")
            self.detection_distance_visualization()
            
            print("4. 🔬 Physical Optics Demonstration...")
            self.physical_optics_demonstration()
            
            print("5. ⚡ Surface Current Visualization...")
            self.surface_current_visualization()
            
            print("6. 🔺 Faceted vs Curved Comparison...")
            self.faceted_vs_curved_comparison()
            
            print("7. 🛩️ Stealth Evolution Timeline...")
            self.stealth_evolution_timeline()
            
            print("8. 📋 Creating Summary Report...")
            self._create_summary_report()
            
            print("✅ All visualizations completed successfully!")
            print(f"📁 Files saved in: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Error during visualization creation: {e}")
            import traceback
            traceback.print_exc()
            
    def _create_summary_report(self):
        """Create a summary report of all visualizations."""
        report = f"""
# Stealth Aircraft Mathematics Visualizations

## Overview
This visualization suite provides comprehensive illustrations of the mathematical concepts behind stealth aircraft technology, as described in the article "Black Diamonds and Bouncing Photons."

## Visualizations Created

### 1. RCS Concept Visualization
- **File**: 1_rcs_concept.png, 1_rcs_concept_interactive.html
- **Concepts**: Basic radar cross section, wave scattering, effective area
- **Key Insights**: How RCS relates to detection and why it's measured in effective area

### 2. RCS Equation Breakdown
- **File**: 2_rcs_equation_breakdown.png
- **Concepts**: Distance scaling, field ratios, effective area interpretation
- **Key Insights**: Mathematical components of the RCS formula σ = lim(r→∞) 4πr²|Es/Ei|²

### 3. Detection Distance Analysis
- **File**: 3_detection_distance.png
- **Concepts**: Fourth root relationship, RCS reduction impact
- **Key Insights**: Why cutting RCS by 100× only reduces detection distance to 56%

### 4. Physical Optics Demonstration
- **File**: 4_physical_optics.png, 4_po_integral_interactive.html
- **Concepts**: PO assumptions, surface currents, computational complexity
- **Key Insights**: How PO simplifies the problem while maintaining accuracy

### 5. Surface Current Visualization
- **File**: 5_surface_currents.png
- **Concepts**: Boundary conditions, current distribution, scattered fields
- **Key Insights**: How surface currents generate scattered electromagnetic fields

### 6. Faceted vs Curved Comparison
- **File**: 6_faceted_vs_curved.png, 6_surface_comparison_3d.html
- **Concepts**: Disco ball effect, controlled reflection, F-117 design
- **Key Insights**: Why faceted surfaces are superior for stealth applications

### 7. Stealth Evolution Timeline
- **File**: 7_stealth_evolution.png, 7_stealth_evolution_interactive.html
- **Concepts**: Technology progression, RCS improvements over time
- **Key Insights**: Evolution from speed-based to shape-based stealth

## Mathematical Concepts Covered

1. **Radar Cross Section (RCS)**: σ = lim(r→∞) 4πr²|Es/Ei|²
2. **Physical Optics Surface Current**: J̃s = 2n̂ × H̃i
3. **PO Scattered Field**: Ẽs = (jkη/4π) ∫S_lit [k̂s × (k̂s × J̃s)] e^(jk(k̂i-k̂s)·r̃) dS
4. **Detection Distance**: R_detection ∝ σ^(1/4)

## Usage Notes

- All static images are saved as high-resolution PNG files (300 DPI)
- Interactive visualizations are saved as HTML files for web viewing
- Color scheme follows a 3Blue1Brown-inspired palette for clarity
- Mathematical formulas are rendered using LaTeX notation

## Dependencies

- numpy, matplotlib, plotly, seaborn
- For interactive features: plotly, ipywidgets
- For 3D visualization: pyvista, vedo (optional)

Generated on: {np.datetime64('now')}
"""
        
        with open(f'{self.output_dir}/README.md', 'w') as f:
            f.write(report)


# Example usage
if __name__ == "__main__":
    viz = StealthMathViz()
    viz.create_complete_visualization_suite() 