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
                       title: str = "3D RCS Pattern",
                       target_angles: Optional[List[Tuple[float, float]]] = None,
                       show_axes: bool = True,
                       close_surface: bool = True) -> go.Figure:
        """
        Plot RCS pattern on a sphere with radar angle indicators.
        
        Args:
            theta_grid: Theta values (elevation)
            phi_grid: Phi values (azimuth)
            rcs_db: RCS values in dBsm
            title: Plot title
            target_angles: List of (theta, phi) tuples for radar positions
            show_axes: Whether to show coordinate axes
            close_surface: Whether to close the surface at poles
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add RCS surface using the helper method
        fig.add_trace(self._create_rcs_surface_trace(theta_grid, phi_grid, rcs_db, showscale=True))
        
        # Add coordinate axes if requested
        if show_axes:
            axis_length = 1.8
            # X-axis (red)
            fig.add_trace(go.Scatter3d(
                x=[0, axis_length], y=[0, 0], z=[0, 0],
                mode='lines+text',
                line=dict(color='red', width=4),
                text=['', 'X'],
                textposition='top center',
                showlegend=False
            ))
            # Y-axis (green)
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[0, axis_length], z=[0, 0],
                mode='lines+text',
                line=dict(color='green', width=4),
                text=['', 'Y'],
                textposition='top center',
                showlegend=False
            ))
            # Z-axis (blue)
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[0, axis_length],
                mode='lines+text',
                line=dict(color='blue', width=4),
                text=['', 'Z'],
                textposition='top center',
                showlegend=False
            ))
        
        # Add radar angle indicators
        if target_angles:
            for i, (theta, phi) in enumerate(target_angles):
                # Convert angle to cartesian for radar position
                radar_distance = 2.5  # Place radar icons outside the RCS sphere
                rx = radar_distance * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
                ry = radar_distance * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
                rz = radar_distance * np.cos(np.deg2rad(theta))
                
                # Add radar indicator (cone pointing to origin)
                fig.add_trace(go.Scatter3d(
                    x=[rx], y=[ry], z=[rz],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color='orange',
                        symbol='diamond'
                    ),
                    text=[f'θ={theta}°<br>φ={phi}°'],
                    textposition='top center',
                    name=f'Radar {i+1}',
                    showlegend=True
                ))
                
                # Add beam line from radar to origin
                fig.add_trace(go.Scatter3d(
                    x=[rx, 0], y=[ry, 0], z=[rz, 0],
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dash'),
                    showlegend=False,
                    opacity=0.5
                ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20)
            ),
            scene=dict(
                aspectmode='cube',
                xaxis=dict(
                    title='X (m)',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title='Y (m)',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                zaxis=dict(
                    title='Z (m)',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)'
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
                
            # Close the surface properly
            if phi_grid[0, 0] != phi_grid[0, -1]:
                theta_grid = np.hstack([theta_grid, theta_grid[:, 0:1]])
                phi_grid = np.hstack([phi_grid, phi_grid[:, 0:1]])
                rcs_db = np.hstack([rcs_db, rcs_db[:, 0:1]])
            
            # Normalize RCS for radius visualization
            rcs_normalized = rcs_db - np.min(rcs_db)
            if np.max(rcs_normalized) > 0:
                rcs_normalized = 0.2 + 1.3 * (rcs_normalized / np.max(rcs_normalized))
            else:
                rcs_normalized = np.ones_like(rcs_db)
            
            # Convert to Cartesian for sphere plot
            x = rcs_normalized * np.sin(np.deg2rad(theta_grid)) * np.cos(np.deg2rad(phi_grid))
            y = rcs_normalized * np.sin(np.deg2rad(theta_grid)) * np.sin(np.deg2rad(phi_grid))
            z = rcs_normalized * np.cos(np.deg2rad(theta_grid))
            
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=rcs_db,
                    colorscale='RdYlBu_r',
                    colorbar=dict(title='RCS (dBsm)', x=1.1 if idx==1 else -0.1),
                    name=f'RCS {idx}',
                    opacity=1.0,
                    contours=dict(
                        z=dict(show=False),
                        x=dict(show=False),
                        y=dict(show=False)
                    ),
                    lighting=dict(
                        ambient=0.8,
                        diffuse=0.8,
                        fresnel=0.2,
                        specular=0.05,
                        roughness=0.5
                    ),
                    lightposition=dict(x=1000, y=1000, z=1000)
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
    
    def plot_geometry_with_radar_beams(self, geometry: Geometry3D,
                                     target_angles: List[Tuple[float, float]],
                                     beam_width: float = 15.0,
                                     title: str = "Geometry with Radar Illumination") -> go.Figure:
        """
        Plot 3D geometry with radar beam cones showing illumination angles.
        
        Args:
            geometry: 3D geometry object
            target_angles: List of (theta, phi) tuples for radar positions
            beam_width: Radar beam width in degrees
            title: Plot title
            
        Returns:
            Plotly figure
        """
        mesh = geometry.mesh
        
        fig = go.Figure()
        
        # Add geometry mesh
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            color='lightblue',
            opacity=0.7,
            name='Target Geometry',
            showscale=False
        ))
        
        # Add radar beams
        for i, (theta, phi) in enumerate(target_angles):
            # Radar position
            radar_distance = 30.0  # Place radar far from target
            rx = radar_distance * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
            ry = radar_distance * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
            rz = radar_distance * np.cos(np.deg2rad(theta))
            
            # Create cone representing radar beam
            cone_length = 25.0
            cone_samples = 20
            
            # Generate cone surface
            u = np.linspace(0, 2*np.pi, cone_samples)
            cone_heights = np.linspace(0, cone_length, 10)
            
            # Direction vector from radar to origin
            dir_x = -rx / radar_distance
            dir_y = -ry / radar_distance
            dir_z = -rz / radar_distance
            
            for h in cone_heights[1:]:
                radius = h * np.tan(np.deg2rad(beam_width/2))
                
                # Create perpendicular vectors for cone base
                if abs(dir_z) < 0.9:
                    perp1 = np.cross([dir_x, dir_y, dir_z], [0, 0, 1])
                else:
                    perp1 = np.cross([dir_x, dir_y, dir_z], [1, 0, 0])
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross([dir_x, dir_y, dir_z], perp1)
                
                # Generate circle points
                cx = rx + h * dir_x + radius * (np.cos(u)[:, np.newaxis] * perp1[0] + 
                                                 np.sin(u)[:, np.newaxis] * perp2[0]).flatten()
                cy = ry + h * dir_y + radius * (np.cos(u)[:, np.newaxis] * perp1[1] + 
                                                 np.sin(u)[:, np.newaxis] * perp2[1]).flatten()
                cz = rz + h * dir_z + radius * (np.cos(u)[:, np.newaxis] * perp1[2] + 
                                                 np.sin(u)[:, np.newaxis] * perp2[2]).flatten()
                
                # Add cone outline
                if h == cone_heights[-1]:  # Only show the outermost cone
                    fig.add_trace(go.Scatter3d(
                        x=cx, y=cy, z=cz,
                        mode='lines',
                        line=dict(color=f'hsl({i*60}, 70%, 50%)', width=3),
                        name=f'Radar {i+1} (\u03b8={theta}\u00b0, \u03c6={phi}\u00b0)',
                        showlegend=True,
                        opacity=0.4
                    ))
            
            # Add radar position marker
            fig.add_trace(go.Scatter3d(
                x=[rx], y=[ry], z=[rz],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=f'hsl({i*60}, 70%, 50%)',
                    symbol='diamond'
                ),
                text=[f'Radar {i+1}'],
                textposition='top center',
                showlegend=False
            ))
            
            # Add center beam line
            fig.add_trace(go.Scatter3d(
                x=[rx, rx + cone_length * dir_x],
                y=[ry, ry + cone_length * dir_y],
                z=[rz, rz + cone_length * dir_z],
                mode='lines',
                line=dict(color=f'hsl({i*60}, 70%, 50%)', width=4),
                showlegend=False,
                opacity=0.8
            ))
        
        # Add coordinate axes
        axis_length = 10.0
        # X-axis
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='red', width=3),
            text=['', 'X'],
            textposition='top center',
            showlegend=False
        ))
        # Y-axis
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines+text',
            line=dict(color='green', width=3),
            text=['', 'Y'],
            textposition='top center',
            showlegend=False
        ))
        # Z-axis
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines+text',
            line=dict(color='blue', width=3),
            text=['', 'Z'],
            textposition='top center',
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20)
            ),
            scene=dict(
                aspectmode='data',
                xaxis=dict(
                    title='X (m)',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title='Y (m)', 
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                zaxis=dict(
                    title='Z (m)',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8)
                )
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
        
        return fig
    
    def plot_combined_view(self, geometry: Geometry3D,
                          rcs_calculator: RCS3DCalculator,
                          target_angles: List[Tuple[float, float]],
                          title: str = "RCS Analysis Combined View") -> go.Figure:
        """
        Create a combined view with geometry and RCS pattern side by side.
        
        Args:
            geometry: 3D geometry object
            rcs_calculator: RCS calculator
            target_angles: List of (theta, phi) tuples for evaluation
            title: Plot title
            
        Returns:
            Plotly figure with subplots
        """
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Target Geometry with Radar Beams', 'RCS Pattern (dBsm)'),
            horizontal_spacing=0.15
        )
        
        mesh = geometry.mesh
        
        # Left plot: Geometry with radar beams
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color='lightblue',
                opacity=0.8,
                name='Geometry',
                showscale=False
            ),
            row=1, col=1
        )
        
        # Add radar indicators on geometry plot
        for i, (theta, phi) in enumerate(target_angles):
            radar_distance = 15.0
            rx = radar_distance * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
            ry = radar_distance * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
            rz = radar_distance * np.cos(np.deg2rad(theta))
            
            fig.add_trace(
                go.Scatter3d(
                    x=[rx, 0], y=[ry, 0], z=[rz, 0],
                    mode='lines+markers',
                    line=dict(color='orange', width=2),
                    marker=dict(size=[8, 0], color='orange'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Right plot: Calculate and show RCS pattern
        print("Calculating RCS pattern...")
        theta_range = (0, 180)
        phi_range = (0, 360)
        n_samples = 30
        
        theta_grid, phi_grid, rcs_db = rcs_calculator.calculate_rcs_pattern(
            geometry.mesh, theta_range, phi_range, n_samples, n_samples
        )
        
        # Close the surface properly
        if phi_grid[0, 0] != phi_grid[0, -1]:
            theta_grid = np.hstack([theta_grid, theta_grid[:, 0:1]])
            phi_grid = np.hstack([phi_grid, phi_grid[:, 0:1]])
            rcs_db = np.hstack([rcs_db, rcs_db[:, 0:1]])
        
        # Normalize RCS for radius visualization
        rcs_normalized = rcs_db - np.min(rcs_db)
        if np.max(rcs_normalized) > 0:
            rcs_normalized = 0.2 + 1.3 * (rcs_normalized / np.max(rcs_normalized))
        else:
            rcs_normalized = np.ones_like(rcs_db)
        
        # Convert to Cartesian
        x = rcs_normalized * np.sin(np.deg2rad(theta_grid)) * np.cos(np.deg2rad(phi_grid))
        y = rcs_normalized * np.sin(np.deg2rad(theta_grid)) * np.sin(np.deg2rad(phi_grid))
        z = rcs_normalized * np.cos(np.deg2rad(theta_grid))
        
        fig.add_trace(
            go.Surface(
                x=x, y=y, z=z,
                surfacecolor=rcs_db,
                colorscale='RdYlBu_r',
                colorbar=dict(
                    title='RCS (dBsm)',
                    x=1.15,
                    thickness=20
                ),
                contours=dict(
                    z=dict(show=False),
                    x=dict(show=False),
                    y=dict(show=False)
                ),
                opacity=1.0,  # Full opacity
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.8,
                    fresnel=0.2,
                    specular=0.05,
                    roughness=0.5
                ),
                lightposition=dict(x=1000, y=1000, z=1000)
            ),
            row=1, col=2
        )
        
        # Add radar positions on RCS plot
        for i, (theta, phi) in enumerate(target_angles):
            radar_distance = 2.0
            rx = radar_distance * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
            ry = radar_distance * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
            rz = radar_distance * np.cos(np.deg2rad(theta))
            
            fig.add_trace(
                go.Scatter3d(
                    x=[rx], y=[ry], z=[rz],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='diamond'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=22),
                x=0.5
            ),
            showlegend=False,
            height=700
        )
        
        # Update both 3D scenes
        fig.update_scenes(
            aspectmode='data',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            zaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        return fig
    
    def _create_rcs_surface_trace(self, theta_grid: np.ndarray, phi_grid: np.ndarray, 
                                  rcs_db: np.ndarray, showscale: bool = True) -> go.Surface:
        """Helper method to create RCS surface trace with consistent settings."""
        # Ensure grids are properly shaped - exact same logic as plot_rcs_sphere
        theta_grid_closed = theta_grid
        phi_grid_closed = phi_grid
        rcs_db_closed = rcs_db
        
        # Check if phi wraps around (0 to 360 is already wrapped, don't add extra column)
        if phi_grid[0, 0] != phi_grid[0, -1] and not (phi_grid[0, 0] == 0 and phi_grid[0, -1] == 360):
            # Add wrapped column to close the surface
            theta_grid_closed = np.hstack([theta_grid, theta_grid[:, 0:1]])
            phi_grid_closed = np.hstack([phi_grid, phi_grid[:, 0:1]])
            rcs_db_closed = np.hstack([rcs_db, rcs_db[:, 0:1]])
        
        # Normalize RCS values to radius (0.1 to 1.5 for better visualization)
        rcs_normalized = rcs_db_closed - np.min(rcs_db_closed)
        if np.max(rcs_normalized) > 0:
            rcs_normalized = 0.1 + 1.4 * (rcs_normalized / np.max(rcs_normalized))
        else:
            rcs_normalized = np.ones_like(rcs_db_closed)
        
        # Convert to Cartesian coordinates with RCS as radius
        x = rcs_normalized * np.sin(np.deg2rad(theta_grid_closed)) * np.cos(np.deg2rad(phi_grid_closed))
        y = rcs_normalized * np.sin(np.deg2rad(theta_grid_closed)) * np.sin(np.deg2rad(phi_grid_closed))
        z = rcs_normalized * np.cos(np.deg2rad(theta_grid_closed))
        
        return go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=rcs_db_closed,
            colorscale='RdYlBu_r',  # Red for high RCS, blue for low
            colorbar=dict(
                title='RCS (dBsm)',
                x=1.02,
                thickness=20
            ) if showscale else None,
            name='RCS Pattern',
            opacity=1.0,  # Full opacity to prevent see-through
            showscale=showscale,
            contours=dict(
                z=dict(show=False),  # Disable contour lines to remove artifacts
                x=dict(show=False),
                y=dict(show=False)
            ),
            lighting=dict(
                ambient=0.8,
                diffuse=0.8,
                fresnel=0.2,
                specular=0.05,
                roughness=0.5
            ),
            lightposition=dict(
                x=1000,
                y=1000,
                z=1000
            )
        )
    
    def create_comprehensive_evolution(self, optimization_history: Dict,
                                     rcs_calculator: RCS3DCalculator,
                                     target_angles: List[Tuple[float, float]],
                                     skip_frames: int = 1,
                                     n_rcs_samples: int = 20) -> go.Figure:
        """
        Create comprehensive evolution animation showing geometry, RCS pattern, and metrics.
        
        Args:
            optimization_history: Dictionary with optimization data
            rcs_calculator: RCS calculator for pattern computation
            target_angles: Radar angles to visualize
            skip_frames: Show every nth frame
            n_rcs_samples: Resolution for RCS pattern calculation
            
        Returns:
            Plotly figure with comprehensive animation
        """
        from plotly.subplots import make_subplots
        
        geometries = optimization_history['geometries'][::skip_frames]
        objective_values = optimization_history['objective_values'][::skip_frames]
        
        print(f"Creating comprehensive evolution animation ({len(geometries)} frames)...")
        
        # Create figure with subplots - 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'surface'}, {'type': 'surface'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ],
            subplot_titles=('Geometry Evolution', 
                          'RCS Pattern Evolution',
                          'RCS at Target Angles',
                          'Objective Function'),
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )
        
        # Calculate RCS patterns for all frames
        print("Calculating RCS patterns for all frames...")
        rcs_patterns = []
        rcs_at_targets = []  # Store RCS values at target angles
        
        for i, geom in enumerate(geometries):
            print(f"  Frame {i+1}/{len(geometries)}...", end='\r')
            theta_grid, phi_grid, rcs_db = rcs_calculator.calculate_rcs_pattern(
                geom.mesh, 
                theta_range=(0, 180),
                phi_range=(0, 360),
                n_theta=n_rcs_samples,
                n_phi=n_rcs_samples
            )
            rcs_patterns.append((theta_grid, phi_grid, rcs_db))
            
            # Calculate RCS at each target angle
            target_rcs = []
            for theta, phi in target_angles:
                rcs_val = rcs_calculator.calculate_rcs(geom.mesh, theta, phi)
                target_rcs.append(10 * np.log10(rcs_val + 1e-10))
            rcs_at_targets.append(target_rcs)
        
        # Prepare frames
        frames = []
        
        for frame_idx, (geom, (theta_grid, phi_grid, rcs_db)) in enumerate(zip(geometries, rcs_patterns)):
            mesh = geom.mesh
            frame_data = []
            
            # 1. TOP LEFT: Geometry with radar beams
            # Add mesh
            frame_data.append(go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color='lightblue',
                opacity=0.8,
                name=f'Iteration {frame_idx*skip_frames}',
                showscale=False
            ))
            
            # Add radar beam indicators
            for j, (theta, phi) in enumerate(target_angles):
                radar_distance = 15.0
                rx = radar_distance * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
                ry = radar_distance * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
                rz = radar_distance * np.cos(np.deg2rad(theta))
                
                # Radar position marker
                frame_data.append(go.Scatter3d(
                    x=[rx], y=[ry], z=[rz],
                    mode='markers+text',
                    marker=dict(size=10, color=f'hsl({j*60}, 70%, 50%)', symbol='diamond'),
                    text=[f'R{j+1}'],
                    textposition='top center',
                    showlegend=False
                ))
                
                # Radar beam line
                frame_data.append(go.Scatter3d(
                    x=[rx, 0], y=[ry, 0], z=[rz, 0],
                    mode='lines',
                    line=dict(color=f'hsl({j*60}, 70%, 50%)', width=2, dash='dash'),
                    showlegend=False
                ))
            
            # 2. TOP RIGHT: RCS Pattern
            # Create surface directly using the helper method
            rcs_surface = self._create_rcs_surface_trace(theta_grid, phi_grid, rcs_db, showscale=True if frame_idx == 0 else False)
            frame_data.append(rcs_surface)
            
            # Add radar positions on RCS sphere
            for j, (theta, phi) in enumerate(target_angles):
                r = 2.0
                rx = r * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
                ry = r * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
                rz = r * np.cos(np.deg2rad(theta))
                
                frame_data.append(go.Scatter3d(
                    x=[rx], y=[ry], z=[rz],
                    mode='markers+text',
                    marker=dict(size=8, color=f'hsl({j*60}, 70%, 50%)', symbol='diamond'),
                    text=[f'R{j+1}'],
                    textposition='top center',
                    showlegend=False
                ))
            
            # 3. BOTTOM LEFT: RCS values at target angles
            # Create bar chart or line plot for each target angle
            angle_labels = [f'R{j+1}\n({theta}°,{phi}°)' for j, (theta, phi) in enumerate(target_angles)]
            current_rcs = rcs_at_targets[frame_idx]
            
            # Create grouped bar chart showing RCS evolution
            frame_data.append(go.Bar(
                x=angle_labels,
                y=current_rcs,
                marker=dict(
                    color=[f'hsl({j*60}, 70%, 50%)' for j in range(len(target_angles))],
                    line=dict(color='black', width=1)
                ),
                text=[f'{val:.1f}' for val in current_rcs],
                textposition='outside',
                showlegend=False
            ))
            
            # 4. BOTTOM RIGHT: Objective function plot
            frame_data.append(go.Scatter(
                x=list(range(frame_idx+1)),
                y=objective_values[:frame_idx+1],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=6),
                name='Objective',
                showlegend=False
            ))
            
            # Highlight current point
            frame_data.append(go.Scatter(
                x=[frame_idx],
                y=[objective_values[frame_idx]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                showlegend=False
            ))
            
            frame = go.Frame(data=frame_data, name=str(frame_idx))
            frames.append(frame)
        
        # Add initial frame data to all subplots
        initial_mesh = geometries[0].mesh
        initial_theta, initial_phi, initial_rcs_db = rcs_patterns[0]
        
        # 1. TOP LEFT: Initial geometry with radar indicators
        fig.add_trace(
            go.Mesh3d(
                x=initial_mesh.vertices[:, 0],
                y=initial_mesh.vertices[:, 1],
                z=initial_mesh.vertices[:, 2],
                i=initial_mesh.faces[:, 0],
                j=initial_mesh.faces[:, 1],
                k=initial_mesh.faces[:, 2],
                color='lightblue',
                opacity=0.8,
                showscale=False
            ),
            row=1, col=1
        )
        
        # Add initial radar indicators
        for j, (theta, phi) in enumerate(target_angles):
            radar_distance = 15.0
            rx = radar_distance * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
            ry = radar_distance * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
            rz = radar_distance * np.cos(np.deg2rad(theta))
            
            fig.add_trace(
                go.Scatter3d(
                    x=[rx], y=[ry], z=[rz],
                    mode='markers+text',
                    marker=dict(size=10, color=f'hsl({j*60}, 70%, 50%)', symbol='diamond'),
                    text=[f'R{j+1}'],
                    textposition='top center',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter3d(
                    x=[rx, 0], y=[ry, 0], z=[rz, 0],
                    mode='lines',
                    line=dict(color=f'hsl({j*60}, 70%, 50%)', width=2, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. TOP RIGHT: Initial RCS pattern
        # Create surface directly using the helper method
        initial_rcs_surface = self._create_rcs_surface_trace(initial_theta, initial_phi, initial_rcs_db, showscale=True)
        fig.add_trace(initial_rcs_surface, row=1, col=2)
        
        # Add radar positions on RCS
        for j, (theta, phi) in enumerate(target_angles):
            r = 2.0
            rx = r * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
            ry = r * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
            rz = r * np.cos(np.deg2rad(theta))
            
            fig.add_trace(
                go.Scatter3d(
                    x=[rx], y=[ry], z=[rz],
                    mode='markers+text',
                    marker=dict(size=8, color=f'hsl({j*60}, 70%, 50%)', symbol='diamond'),
                    text=[f'R{j+1}'],
                    textposition='top center',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. BOTTOM LEFT: Initial RCS at target angles
        initial_rcs_values = rcs_at_targets[0]
        angle_labels = [f'R{j+1}\n({theta}°,{phi}°)' for j, (theta, phi) in enumerate(target_angles)]
        
        fig.add_trace(
            go.Bar(
                x=angle_labels,
                y=initial_rcs_values,
                marker=dict(
                    color=[f'hsl({j*60}, 70%, 50%)' for j in range(len(target_angles))],
                    line=dict(color='black', width=1)
                ),
                text=[f'{val:.1f}' for val in initial_rcs_values],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. BOTTOM RIGHT: Initial objective plot
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[objective_values[0]],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=6),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add frames
        fig.frames = frames
        
        # Animation controls
        fig.update_layout(
            title=dict(
                text='Comprehensive RCS Optimization Evolution',
                font=dict(size=20),
                x=0.5,
                y=0.98
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 1.0,
                'x': 0.02,
                'xanchor': 'left',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 400, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 200}
                        }]
                    },
                    {
                        'label': '❚❚ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[str(i)], {
                            'frame': {'duration': 400, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 200}
                        }],
                        'label': f'Iter {i*skip_frames}',
                        'method': 'animate'
                    }
                    for i in range(len(frames))
                ],
                'active': 0,
                'y': 0,
                'len': 0.95,
                'x': 0.025,
                'xanchor': 'left',
                'yanchor': 'top',
                'transition': {'duration': 200},
                'currentvalue': {
                    'prefix': 'Iteration: ',
                    'visible': True,
                    'xanchor': 'right'
                }
            }],
            height=900,
            showlegend=False
        )
        
        # Update 3D scenes (only top row)
        for col in [1, 2]:
            # Use 'data' for geometry (col 1) and 'cube' for RCS sphere (col 2)
            aspectmode = 'data' if col == 1 else 'cube'
            
            # Different camera settings for RCS pattern
            if col == 2:
                camera_eye = dict(x=2.0, y=2.0, z=2.0)  # Further away for RCS sphere
            else:
                camera_eye = dict(x=1.5, y=1.5, z=1.2)
                
            fig.update_scenes(
                dict(
                    aspectmode=aspectmode,
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='lightgray', 
                        title='X',
                        showbackground=True,
                        backgroundcolor='rgba(245,245,245,0.8)',
                        range=[-2, 2] if col == 2 else None  # Fixed range for RCS
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='lightgray', 
                        title='Y',
                        showbackground=True,
                        backgroundcolor='rgba(245,245,245,0.8)',
                        range=[-2, 2] if col == 2 else None  # Fixed range for RCS
                    ),
                    zaxis=dict(
                        showgrid=True, 
                        gridcolor='lightgray', 
                        title='Z',
                        showbackground=True,
                        backgroundcolor='rgba(245,245,245,0.8)',
                        range=[-2, 2] if col == 2 else None  # Fixed range for RCS
                    ),
                    camera=dict(eye=camera_eye)
                ),
                row=1, col=col
            )
        
        # Update 2D axes for RCS bar chart
        fig.update_xaxes(
            title_text="Target Angles", 
            showgrid=False,
            tickangle=-45,
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="RCS (dBsm)", 
            showgrid=True, 
            gridcolor='lightgray',
            row=2, col=1
        )
        
        # Update 2D axes for objective plot
        fig.update_xaxes(
            title_text="Iteration", 
            showgrid=True, 
            gridcolor='lightgray',
            row=2, col=2
        )
        fig.update_yaxes(
            title_text="Objective Value", 
            showgrid=True, 
            gridcolor='lightgray',
            row=2, col=2
        )
        
        return fig