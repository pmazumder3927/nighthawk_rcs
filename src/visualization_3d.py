"""
3D visualization tools for RCS optimization with surface evolution.

This module provides advanced 3D visualization including:
- Real-time 3D mesh rendering
- RCS pattern visualization on spheres
- Surface evolution animations
- Interactive 3D plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv
import vedo
from typing import List, Optional, Tuple, Dict
import imageio
from tqdm import tqdm

from .geometry_3d import Geometry3D
from .rcs_calc_3d import RCS3DCalculator


class RCSVisualizer3D:
    """Advanced 3D visualization for RCS analysis and optimization."""
    
    def __init__(self, backend: str = 'plotly'):
        """
        Initialize 3D visualizer.
        
        Args:
            backend: 'plotly', 'pyvista', or 'vedo'
        """
        self.backend = backend
        
    def plot_geometry_3d(self, geometry: Geometry3D, 
                        show_normals: bool = False,
                        title: str = "3D Geometry") -> go.Figure:
        """
        Plot 3D geometry using Plotly.
        
        Args:
            geometry: 3D geometry object
            show_normals: Whether to show face normals
            title: Plot title
            
        Returns:
            Plotly figure
        """
        mesh = geometry.mesh
        
        # Create mesh plot
        fig = go.Figure(data=[
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color='lightblue',
                opacity=0.8,
                name='Geometry'
            )
        ])
        
        if show_normals:
            # Add normal vectors
            centers = mesh.vertices[mesh.faces].mean(axis=1)
            normals = mesh.face_normals
            
            # Create arrow traces
            arrows = []
            for i in range(0, len(centers), 10):  # Show every 10th normal
                start = centers[i]
                end = centers[i] + 0.5 * normals[i]
                
                arrows.append(go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False
                ))
                
            fig.add_traces(arrows)
            
        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        return fig
        
    def plot_rcs_sphere(self, theta_grid: np.ndarray, 
                       phi_grid: np.ndarray,
                       rcs_db: np.ndarray,
                       title: str = "3D RCS Pattern") -> go.Figure:
        """
        Plot RCS pattern on a sphere.
        
        Args:
            theta_grid: Theta values (elevation)
            phi_grid: Phi values (azimuth)
            rcs_db: RCS values in dBsm
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Convert to Cartesian coordinates
        r = 1.0  # Unit sphere
        x = r * np.sin(np.deg2rad(theta_grid)) * np.cos(np.deg2rad(phi_grid))
        y = r * np.sin(np.deg2rad(theta_grid)) * np.sin(np.deg2rad(phi_grid))
        z = r * np.cos(np.deg2rad(theta_grid))
        
        fig = go.Figure(data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=rcs_db,
                colorscale='Viridis',
                colorbar=dict(title='RCS (dBsm)'),
                name='RCS Pattern'
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='cube',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        return fig
        
    def create_surface_evolution_animation(self, optimization_history: Dict,
                                         save_path: Optional[str] = None,
                                         skip_frames: int = 1) -> go.Figure:
        """
        Create animation of surface evolution during optimization.
        
        Args:
            optimization_history: Dictionary with optimization data
            save_path: Path to save animation
            skip_frames: Show every nth frame
            
        Returns:
            Plotly figure with animation
        """
        geometries = optimization_history['geometries'][::skip_frames]
        objective_values = optimization_history['objective_values'][::skip_frames]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'scatter'}]],
            subplot_titles=('3D Geometry Evolution', 'Objective Value')
        )
        
        # Prepare frames
        frames = []
        
        for i, geom in enumerate(geometries):
            mesh = geom.mesh
            
            # Geometry trace
            mesh_trace = go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color='lightblue',
                opacity=0.8,
                name=f'Iteration {i*skip_frames}'
            )
            
            # Objective curve trace
            obj_trace = go.Scatter(
                x=list(range(i+1)),
                y=objective_values[:i+1],
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            )
            
            frame = go.Frame(
                data=[mesh_trace, obj_trace],
                name=str(i)
            )
            frames.append(frame)
            
        # Add initial frame data
        initial_mesh = geometries[0].mesh
        fig.add_trace(
            go.Mesh3d(
                x=initial_mesh.vertices[:, 0],
                y=initial_mesh.vertices[:, 1],
                z=initial_mesh.vertices[:, 2],
                i=initial_mesh.faces[:, 0],
                j=initial_mesh.faces[:, 1],
                k=initial_mesh.faces[:, 2],
                color='lightblue',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[objective_values[0]],
                mode='lines+markers',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Add frames
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[str(i)], {
                            'frame': {'duration': 100, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': f'{i*skip_frames}',
                        'method': 'animate'
                    }
                    for i in range(len(frames))
                ],
                'active': 0,
                'y': 0,
                'len': 0.9,
                'x': 0.05,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        fig.update_layout(
            title='RCS Optimization Surface Evolution',
            scene=dict(aspectmode='data')
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def plot_optimization_comparison_3d(self, initial_geometry: Geometry3D,
                                      optimized_geometry: Geometry3D,
                                      rcs_calculator: RCS3DCalculator) -> go.Figure:
        """
        Compare initial and optimized geometries with RCS patterns.
        
        Args:
            initial_geometry: Initial geometry
            optimized_geometry: Optimized geometry
            rcs_calculator: RCS calculator for pattern computation
            
        Returns:
            Plotly figure with comparison
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Initial Geometry', 'Optimized Geometry',
                          'Initial RCS Pattern', 'Optimized RCS Pattern'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Plot geometries
        for idx, (geom, name) in enumerate([(initial_geometry, 'Initial'), 
                                           (optimized_geometry, 'Optimized')]):
            mesh = geom.mesh
            fig.add_trace(
                go.Mesh3d(
                    x=mesh.vertices[:, 0],
                    y=mesh.vertices[:, 1],
                    z=mesh.vertices[:, 2],
                    i=mesh.faces[:, 0],
                    j=mesh.faces[:, 1],
                    k=mesh.faces[:, 2],
                    color='lightblue' if idx == 0 else 'lightgreen',
                    opacity=0.8,
                    name=name
                ),
                row=1, col=idx+1
            )
            
        # Calculate and plot RCS patterns
        print("Calculating RCS patterns for comparison...")
        theta_range = (30, 150)
        phi_range = (0, 360)
        n_samples = 25  # Reduced for faster computation
        
        for idx, geom in enumerate([initial_geometry, optimized_geometry]):
            theta_grid, phi_grid, rcs_db = rcs_calculator.calculate_rcs_pattern(
                geom.mesh, theta_range, phi_range, n_samples, n_samples)
                
            # Convert to Cartesian for sphere plot
            r = 1.0
            x = r * np.sin(np.deg2rad(theta_grid)) * np.cos(np.deg2rad(phi_grid))
            y = r * np.sin(np.deg2rad(theta_grid)) * np.sin(np.deg2rad(phi_grid))
            z = r * np.cos(np.deg2rad(theta_grid))
            
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=rcs_db,
                    colorscale='Viridis',
                    colorbar=dict(title='RCS (dBsm)', x=1.1 if idx==1 else -0.1),
                    name=f'RCS {idx}'
                ),
                row=2, col=idx+1
            )
            
        fig.update_layout(
            title='3D RCS Optimization Results',
            height=1000,
            showlegend=False
        )
        
        # Update 3D scene properties
        for i in range(1, 5):
            fig.update_scenes(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                row=(i-1)//2 + 1,
                col=(i-1)%2 + 1
            )
            
        return fig
        
    def create_pyvista_animation(self, optimization_history: Dict,
                               save_path: Optional[str] = None,
                               skip_frames: int = 1):
        """
        Create high-quality animation using PyVista.
        
        Args:
            optimization_history: Optimization history
            save_path: Path to save video
            skip_frames: Show every nth frame
        """
        geometries = optimization_history['geometries'][::skip_frames]
        rcs_values = optimization_history['rcs_values'][::skip_frames]
        
        # Create plotter
        plotter = pv.Plotter(notebook=False, off_screen=save_path is not None)
        plotter.add_axes()
        
        # Initialize with first geometry
        mesh_pv = pv.PolyData(geometries[0].mesh.vertices, 
                            np.column_stack([
                                np.full(len(geometries[0].mesh.faces), 3),
                                geometries[0].mesh.faces
                            ]))
        
        actor = plotter.add_mesh(mesh_pv, color='lightblue', 
                               smooth_shading=True, show_edges=True)
        
        # Add text
        text_actor = plotter.add_text(f"Iteration: 0\nRCS: {rcs_values[0]:.1f} dBsm",
                                    position='upper_right')
        
        plotter.show(auto_close=False)
        
        # Animation function
        if save_path:
            plotter.open_movie(save_path)
            
        for i, geom in enumerate(tqdm(geometries, desc="Creating animation")):
            # Update mesh
            mesh_pv = pv.PolyData(geom.mesh.vertices,
                                np.column_stack([
                                    np.full(len(geom.mesh.faces), 3),
                                    geom.mesh.faces
                                ]))
            
            plotter.remove_actor(actor)
            actor = plotter.add_mesh(mesh_pv, color='lightblue',
                                   smooth_shading=True, show_edges=True)
            
            # Update text
            plotter.remove_actor(text_actor)
            text_actor = plotter.add_text(
                f"Iteration: {i*skip_frames}\nRCS: {rcs_values[i]:.1f} dBsm",
                position='upper_right'
            )
            
            if save_path:
                plotter.write_frame()
            else:
                plotter.render()
                
        if save_path:
            plotter.close()
            
    def plot_rcs_cuts(self, geometry: Geometry3D,
                     rcs_calculator: RCS3DCalculator,
                     cut_type: str = 'azimuth',
                     fixed_angle: float = 90.0) -> plt.Figure:
        """
        Plot 2D cuts of RCS pattern.
        
        Args:
            geometry: 3D geometry
            rcs_calculator: RCS calculator
            cut_type: 'azimuth' or 'elevation'
            fixed_angle: Fixed angle for the cut
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        if cut_type == 'azimuth':
            # Azimuth cut at fixed elevation
            phi = np.linspace(0, 360, 73)
            rcs_vv = []
            rcs_hh = []
            
            for p in phi:
                rcs_vv.append(rcs_calculator.calculate_rcs(
                    geometry.mesh, fixed_angle, p, 'VV'))
                rcs_hh.append(rcs_calculator.calculate_rcs(
                    geometry.mesh, fixed_angle, p, 'HH'))
                    
            rcs_vv = 10 * np.log10(np.array(rcs_vv) + 1e-10)
            rcs_hh = 10 * np.log10(np.array(rcs_hh) + 1e-10)
            
            # Cartesian plot
            ax1.plot(phi, rcs_vv, 'b-', linewidth=2, label='VV')
            ax1.plot(phi, rcs_hh, 'r--', linewidth=2, label='HH')
            ax1.set_xlabel('Azimuth (degrees)')
            ax1.set_ylabel('RCS (dBsm)')
            ax1.set_title(f'Azimuth Cut at θ={fixed_angle}°')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Polar plot
            ax2 = plt.subplot(122, projection='polar')
            phi_rad = np.deg2rad(phi)
            ax2.plot(phi_rad, rcs_vv, 'b-', linewidth=2, label='VV')
            ax2.plot(phi_rad, rcs_hh, 'r--', linewidth=2, label='HH')
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(-1)
            ax2.set_title(f'Polar Plot at θ={fixed_angle}°')
            ax2.grid(True)
            
        else:  # elevation cut
            # Elevation cut at fixed azimuth
            theta = np.linspace(0, 180, 37)
            rcs_vv = []
            rcs_hh = []
            
            for t in theta:
                rcs_vv.append(rcs_calculator.calculate_rcs(
                    geometry.mesh, t, fixed_angle, 'VV'))
                rcs_hh.append(rcs_calculator.calculate_rcs(
                    geometry.mesh, t, fixed_angle, 'HH'))
                    
            rcs_vv = 10 * np.log10(np.array(rcs_vv) + 1e-10)
            rcs_hh = 10 * np.log10(np.array(rcs_hh) + 1e-10)
            
            ax1.plot(theta, rcs_vv, 'b-', linewidth=2, label='VV')
            ax1.plot(theta, rcs_hh, 'r--', linewidth=2, label='HH')
            ax1.set_xlabel('Elevation (degrees)')
            ax1.set_ylabel('RCS (dBsm)')
            ax1.set_title(f'Elevation Cut at φ={fixed_angle}°')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_xlim(0, 180)
            
            # Polar plot (elevation)
            ax2 = plt.subplot(122, projection='polar')
            theta_rad = np.deg2rad(theta)
            ax2.plot(theta_rad, rcs_vv, 'b-', linewidth=2, label='VV')
            ax2.plot(theta_rad, rcs_hh, 'r--', linewidth=2, label='HH')
            ax2.set_theta_zero_location('N')
            ax2.set_title(f'Elevation Pattern at φ={fixed_angle}°')
            ax2.grid(True)
            ax2.set_theta_direction(-1)
            ax2.set_thetamin(0)
            ax2.set_thetamax(180)
            
        plt.tight_layout()
        return fig 