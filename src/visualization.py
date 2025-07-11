"""
Visualization tools for RCS patterns and geometry.

This module provides functions for visualizing RCS calculations,
optimization progress, and geometry evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import plotly.graph_objects as go
from typing import List, Optional, Tuple
from .geometry import Geometry2D, Geometry3D
from .rcs_calc import RCSCalculator


class RCSVisualizer:
    """Visualization tools for RCS analysis and optimization."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        
    def plot_2d_rcs_pattern(self, angles: np.ndarray, rcs_db: np.ndarray,
                           title: str = "RCS Pattern", 
                           reference_rcs_db: Optional[np.ndarray] = None):
        """
        Plot 2D RCS pattern in polar coordinates.
        
        Args:
            angles: Angles in degrees
            rcs_db: RCS values in dBsm
            title: Plot title
            reference_rcs_db: Reference RCS pattern for comparison
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Convert angles to radians
        angles_rad = np.deg2rad(angles)
        
        # Plot main RCS pattern
        ax.plot(angles_rad, rcs_db, 'b-', linewidth=2, label='Current')
        
        # Plot reference if provided
        if reference_rcs_db is not None:
            ax.plot(angles_rad, reference_rcs_db, 'r--', linewidth=2, 
                   label='Reference', alpha=0.7)
            
        # Customize plot
        ax.set_theta_zero_location('E')  # 0° at East (right)
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_title(title, y=1.08)
        ax.grid(True)
        
        # Add range rings
        ax.set_ylim(-40, np.max(rcs_db) + 5)
        
        if reference_rcs_db is not None:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
        return fig, ax
        
    def plot_geometry_comparison(self, geometries: List[Geometry2D], 
                               labels: List[str],
                               rcs_calculator: Optional[RCSCalculator] = None):
        """
        Compare multiple 2D geometries and their RCS patterns.
        
        Args:
            geometries: List of Geometry2D objects
            labels: List of labels for each geometry
            rcs_calculator: Optional RCS calculator for pattern comparison
        """
        n_geoms = len(geometries)
        
        if rcs_calculator is None:
            # Single row for geometry only
            fig, axes = plt.subplots(1, n_geoms, figsize=(5*n_geoms, 5))
            if n_geoms == 1:
                axes = [axes]
                
            for i, (geom, label) in enumerate(zip(geometries, labels)):
                geom.plot(ax=axes[i])
                axes[i].set_title(label)
                
        else:
            # Two rows: geometry and RCS pattern
            fig, axes = plt.subplots(2, n_geoms, figsize=(5*n_geoms, 10))
            if n_geoms == 1:
                axes = axes.reshape(-1, 1)
                
            for i, (geom, label) in enumerate(zip(geometries, labels)):
                # Plot geometry
                geom.plot(ax=axes[0, i])
                axes[0, i].set_title(f"{label} - Geometry")
                
                # Calculate and plot RCS pattern
                angles, rcs_db = rcs_calculator.calculate_2d_rcs_pattern(geom)
                ax_polar = plt.subplot(2, n_geoms, n_geoms + i + 1, 
                                     projection='polar')
                angles_rad = np.deg2rad(angles)
                ax_polar.plot(angles_rad, rcs_db, 'b-', linewidth=2)
                ax_polar.set_theta_zero_location('E')
                ax_polar.set_theta_direction(-1)
                ax_polar.set_title(f"{label} - RCS Pattern")
                ax_polar.grid(True)
                ax_polar.set_ylim(-40, np.max(rcs_db) + 5)
                
        plt.tight_layout()
        return fig
        
    def create_optimization_animation(self, optimization_history: dict,
                                    rcs_calculator: RCSCalculator,
                                    save_path: Optional[str] = None):
        """
        Create animation showing optimization progress.
        
        Args:
            optimization_history: Dictionary with 'geometries' and 'rcs_values'
            rcs_calculator: RCS calculator instance
            save_path: Optional path to save animation
        """
        geometries = optimization_history['geometries']
        rcs_values = optimization_history['rcs_values']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Initialize plots
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Geometry Evolution')
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mean RCS (dBsm)')
        ax2.set_title('Optimization Progress')
        ax2.grid(True, alpha=0.3)
        
        # Plot optimization curve
        ax2.plot(rcs_values, 'b-', linewidth=2)
        point, = ax2.plot([], [], 'ro', markersize=8)
        
        # Animation update function
        def update(frame):
            ax1.clear()
            geometries[frame].plot(ax=ax1)
            ax1.set_title(f'Geometry Evolution - Iteration {frame}')
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            
            # Update progress marker
            point.set_data([frame], [rcs_values[frame]])
            
            return ax1.patches + ax1.lines + [point]
            
        anim = FuncAnimation(fig, update, frames=len(geometries),
                           interval=100, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            
        return fig, anim
        
    def plot_3d_rcs_sphere(self, geometry: Geometry3D, 
                         rcs_calculator: RCSCalculator,
                         n_samples: int = 30):
        """
        Plot 3D RCS pattern as a sphere colored by RCS values.
        
        Args:
            geometry: 3D geometry object
            rcs_calculator: RCS calculator instance
            n_samples: Number of samples in theta and phi
        """
        # Create spherical grid
        theta = np.linspace(0, 180, n_samples)
        phi = np.linspace(0, 360, n_samples)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        # Calculate RCS for each direction
        rcs_grid = np.zeros_like(theta_grid)
        for i in range(n_samples):
            for j in range(n_samples):
                rcs_grid[i, j] = rcs_calculator.calculate_3d_rcs(
                    geometry, theta_grid[i, j], phi_grid[i, j])
                    
        # Convert to dBsm
        rcs_db = 10 * np.log10(rcs_grid + 1e-10)
        
        # Create 3D plot using plotly
        fig = go.Figure()
        
        # Add RCS sphere
        fig.add_trace(go.Surface(
            x=np.sin(np.deg2rad(theta_grid)) * np.cos(np.deg2rad(phi_grid)),
            y=np.sin(np.deg2rad(theta_grid)) * np.sin(np.deg2rad(phi_grid)),
            z=np.cos(np.deg2rad(theta_grid)),
            surfacecolor=rcs_db,
            colorscale='Viridis',
            colorbar=dict(title='RCS (dBsm)'),
            name='RCS Pattern'
        ))
        
        fig.update_layout(
            title='3D RCS Pattern',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            )
        )
        
        return fig
        
    def plot_optimization_metrics(self, optimization_history: dict):
        """
        Plot detailed optimization metrics.
        
        Args:
            optimization_history: Dictionary with optimization data
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        iterations = range(len(optimization_history['rcs_values']))
        
        # Mean RCS
        axes[0, 0].plot(iterations, optimization_history['rcs_values'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Mean RCS (dBsm)')
        axes[0, 0].set_title('Mean RCS Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Max RCS
        if 'max_rcs' in optimization_history:
            axes[0, 1].plot(iterations, optimization_history['max_rcs'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Max RCS (dBsm)')
            axes[0, 1].set_title('Maximum RCS Evolution')
            axes[0, 1].grid(True, alpha=0.3)
            
        # RCS reduction
        initial_rcs = optimization_history['rcs_values'][0]
        reduction = initial_rcs - np.array(optimization_history['rcs_values'])
        axes[1, 0].plot(iterations, reduction, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('RCS Reduction (dB)')
        axes[1, 0].set_title('Cumulative RCS Reduction')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence rate
        if len(iterations) > 1:
            conv_rate = np.abs(np.diff(optimization_history['rcs_values']))
            axes[1, 1].semilogy(iterations[1:], conv_rate, 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('|ΔRCS| (dB)')
            axes[1, 1].set_title('Convergence Rate')
            axes[1, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        return fig 