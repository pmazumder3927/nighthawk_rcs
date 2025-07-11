"""
Geometry representation for RCS calculations.

This module provides classes for representing 2D and 3D geometries
suitable for RCS calculations and topology optimization.
"""

import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class Geometry2D:
    """
    2D geometry representation using polygonal boundaries.
    
    Attributes:
        vertices: Array of shape (N, 2) containing vertex coordinates
        edges: Array of shape (M, 2) containing vertex indices for each edge
    """
    
    def __init__(self, vertices: np.ndarray):
        """
        Initialize 2D geometry from vertices.
        
        Args:
            vertices: Array of shape (N, 2) containing x, y coordinates
        """
        self.vertices = np.array(vertices)
        self.n_vertices = len(vertices)
        
        # Create edges by connecting consecutive vertices
        self.edges = np.array([(i, (i + 1) % self.n_vertices) 
                              for i in range(self.n_vertices)])
        
        # Compute edge normals and centers
        self._compute_edge_properties()
        
    def _compute_edge_properties(self):
        """Compute edge normals and centers for RCS calculations."""
        self.edge_centers = []
        self.edge_normals = []
        self.edge_lengths = []
        
        for edge in self.edges:
            v1, v2 = self.vertices[edge[0]], self.vertices[edge[1]]
            
            # Edge center
            center = (v1 + v2) / 2
            self.edge_centers.append(center)
            
            # Edge vector and length
            edge_vec = v2 - v1
            length = np.linalg.norm(edge_vec)
            self.edge_lengths.append(length)
            
            # Outward normal (assuming counter-clockwise vertices)
            normal = np.array([-edge_vec[1], edge_vec[0]])
            normal = normal / np.linalg.norm(normal)
            self.edge_normals.append(normal)
            
        self.edge_centers = np.array(self.edge_centers)
        self.edge_normals = np.array(self.edge_normals)
        self.edge_lengths = np.array(self.edge_lengths)
        
    def plot(self, ax=None, **kwargs):
        """Plot the 2D geometry."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
        # Default styling
        kwargs.setdefault('facecolor', 'lightgray')
        kwargs.setdefault('edgecolor', 'black')
        kwargs.setdefault('linewidth', 2)
        
        polygon = Polygon(self.vertices, **kwargs)
        ax.add_patch(polygon)
        
        # Plot normals
        for i, (center, normal) in enumerate(zip(self.edge_centers, self.edge_normals)):
            ax.arrow(center[0], center[1], 
                    0.1 * normal[0], 0.1 * normal[1],
                    head_width=0.05, head_length=0.02, 
                    fc='red', ec='red', alpha=0.5)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        return ax
        
    def perturb_vertices(self, perturbations: np.ndarray):
        """
        Apply perturbations to vertices for optimization.
        
        Args:
            perturbations: Array of shape (N, 2) with vertex displacements
            
        Returns:
            New Geometry2D object with perturbed vertices
        """
        new_vertices = self.vertices + perturbations
        return Geometry2D(new_vertices)


class Geometry3D:
    """
    3D geometry representation using triangular mesh.
    
    Attributes:
        vertices: Array of shape (N, 3) containing vertex coordinates
        faces: Array of shape (M, 3) containing vertex indices for each triangle
    """
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Initialize 3D geometry from vertices and faces.
        
        Args:
            vertices: Array of shape (N, 3) containing x, y, z coordinates
            faces: Array of shape (M, 3) containing vertex indices for triangles
        """
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        
        # Compute face properties
        self._compute_face_properties()
        
    def _compute_face_properties(self):
        """Compute face normals and centers for RCS calculations."""
        self.face_centers = []
        self.face_normals = []
        self.face_areas = []
        
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            
            # Face center
            center = (v0 + v1 + v2) / 3
            self.face_centers.append(center)
            
            # Face normal (right-hand rule)
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            self.face_areas.append(area)
            
            normal = normal / np.linalg.norm(normal)
            self.face_normals.append(normal)
            
        self.face_centers = np.array(self.face_centers)
        self.face_normals = np.array(self.face_normals)
        self.face_areas = np.array(self.face_areas)


def create_f117_inspired_2d_profile() -> Geometry2D:
    """
    Create a 2D cross-section inspired by F-117 Nighthawk design principles.
    
    The F-117 used flat faceted surfaces to reflect radar energy away from
    the source. This creates a simplified 2D profile with similar principles.
    
    Returns:
        Geometry2D object representing the cross-section
    """
    # Create a faceted diamond-like profile
    # Dimensions in meters (scaled down)
    vertices = np.array([
        [0.0, 0.5],      # Top vertex
        [1.0, 0.2],      # Front right
        [0.8, -0.3],     # Bottom right
        [0.0, -0.4],     # Bottom center
        [-0.8, -0.3],    # Bottom left
        [-1.0, 0.2],     # Front left
    ])
    
    # Scale to realistic size (approx 15m wingspan)
    vertices *= 7.5
    
    return Geometry2D(vertices)


def create_conventional_aircraft_2d_profile() -> Geometry2D:
    """
    Create a conventional aircraft 2D profile for comparison.
    
    Returns:
        Geometry2D object representing a smooth aircraft profile
    """
    # Create smooth elliptical profile
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    vertices = np.column_stack([
        8.0 * np.cos(t),  # 16m width
        3.0 * np.sin(t)   # 6m height
    ])
    
    return Geometry2D(vertices) 