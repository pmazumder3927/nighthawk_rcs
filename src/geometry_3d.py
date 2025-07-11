"""
3D geometry representation and manipulation for RCS optimization.

This module provides advanced 3D mesh handling including:
- Mesh generation and loading
- Smooth deformation fields
- Topology-preserving modifications
- F-117 inspired 3D geometries
"""

import numpy as np
import trimesh
from typing import Optional, Tuple, List
import pygmsh
from scipy.spatial import Delaunay
from scipy.interpolate import RBFInterpolator


class Geometry3D:
    """
    Advanced 3D geometry class with deformation capabilities.
    """
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize with a trimesh object.
        
        Args:
            mesh: Trimesh object
        """
        self.mesh = mesh
        self.original_vertices = mesh.vertices.copy()
        
        # Compute additional properties
        self._compute_properties()
        
    def _compute_properties(self):
        """Compute mesh properties for optimization."""
        # Ensure normals are computed
        self.mesh.face_normals
        self.mesh.vertex_normals
        
        # Compute bounds
        self.bounds = self.mesh.bounds
        self.center = self.mesh.centroid
        self._scale = np.max(self.bounds[1] - self.bounds[0])
        
        # Compute quality metrics
        self.edge_lengths = self.mesh.edges_unique_length
        self.mean_edge_length = np.mean(self.edge_lengths)
        
    def apply_deformation(self, control_points: np.ndarray, 
                         displacements: np.ndarray,
                         smoothing: float = 1.0) -> 'Geometry3D':
        """
        Apply smooth deformation using radial basis functions.
        
        Args:
            control_points: Nx3 array of control point locations
            displacements: Nx3 array of displacement vectors
            smoothing: RBF smoothing parameter
            
        Returns:
            New Geometry3D with deformed mesh
        """
        # Create RBF interpolator for smooth deformation
        rbf = RBFInterpolator(control_points, displacements, 
                            kernel='thin_plate_spline',
                            smoothing=smoothing)
        
        # Apply deformation to all vertices
        vertex_displacements = rbf(self.mesh.vertices)
        new_vertices = self.mesh.vertices + vertex_displacements
        
        # Create new mesh
        new_mesh = trimesh.Trimesh(vertices=new_vertices,
                                  faces=self.mesh.faces,
                                  process=True)
        
        return Geometry3D(new_mesh)
        
    def apply_vertex_displacement(self, displacements: np.ndarray) -> 'Geometry3D':
        """
        Apply direct vertex displacements.
        
        Args:
            displacements: Array of shape (n_vertices, 3)
            
        Returns:
            New Geometry3D object
        """
        new_vertices = self.mesh.vertices + displacements
        new_mesh = trimesh.Trimesh(vertices=new_vertices,
                                  faces=self.mesh.faces,
                                  process=True)
        return Geometry3D(new_mesh)
        
    def get_surface_points(self, n_points: int = 1000) -> np.ndarray:
        """
        Sample points uniformly on the surface.
        
        Args:
            n_points: Number of points to sample
            
        Returns:
            Array of shape (n_points, 3)
        """
        points, face_indices = trimesh.sample.sample_surface(self.mesh, n_points)
        return points
        
    def subdivide(self, iterations: int = 1) -> 'Geometry3D':
        """
        Subdivide mesh for finer control.
        
        Args:
            iterations: Number of subdivision iterations
            
        Returns:
            New Geometry3D with subdivided mesh
        """
        subdivided = self.mesh.subdivide(iterations)
        return Geometry3D(subdivided)
        
    def simplify(self, target_faces: int) -> 'Geometry3D':
        """
        Simplify mesh to target number of faces.
        
        Args:
            target_faces: Target number of faces
            
        Returns:
            Simplified Geometry3D
        """
        simplified = self.mesh.simplify_quadric_decimation(target_faces)
        return Geometry3D(simplified)
        
    def scale(self, factor: float) -> 'Geometry3D':
        """
        Scale the geometry by a factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New scaled Geometry3D object
        """
        scaled_vertices = self.mesh.vertices * factor
        scaled_mesh = trimesh.Trimesh(vertices=scaled_vertices,
                                     faces=self.mesh.faces,
                                     process=True)
        return Geometry3D(scaled_mesh)
        
    def rotate(self, rotation: np.ndarray) -> 'Geometry3D':
        """
        Rotate the geometry.
        
        Args:
            rotation: Rotation angles [rx, ry, rz] in radians
            
        Returns:
            New rotated Geometry3D object
        """
        from scipy.spatial.transform import Rotation
        
        # Create rotation matrix
        r = Rotation.from_euler('xyz', rotation)
        rotation_matrix = r.as_matrix()
        
        # Apply rotation
        rotated_vertices = self.mesh.vertices @ rotation_matrix.T
        rotated_mesh = trimesh.Trimesh(vertices=rotated_vertices,
                                      faces=self.mesh.faces,
                                      process=True)
        return Geometry3D(rotated_mesh)
        
    def translate(self, translation: np.ndarray) -> 'Geometry3D':
        """
        Translate the geometry.
        
        Args:
            translation: Translation vector [tx, ty, tz]
            
        Returns:
            New translated Geometry3D object
        """
        translated_vertices = self.mesh.vertices + translation
        translated_mesh = trimesh.Trimesh(vertices=translated_vertices,
                                         faces=self.mesh.faces,
                                         process=True)
        return Geometry3D(translated_mesh)
    
    def export(self, filename: str):
        """Export mesh to file."""
        self.mesh.export(filename)
        
    @property
    def volume(self) -> float:
        """Get mesh volume."""
        return self.mesh.volume
        
    @property
    def surface_area(self) -> float:
        """Get total surface area."""
        return self.mesh.area
        
    @property
    def scale_factor(self) -> float:
        """Get the scale factor (largest dimension)."""
        return np.max(self.bounds[1] - self.bounds[0])


def create_f117_inspired_3d() -> Geometry3D:
    """
    Create a 3D F-117 inspired geometry with faceted surfaces.
    
    Returns:
        Geometry3D object
    """
    # Define key vertices for F-117-like shape
    # Dimensions roughly based on F-117 proportions (scaled)
    
    # Top pyramid vertices
    vertices = [
        # Nose point
        [15.0, 0.0, 0.0],
        
        # Front top edge
        [10.0, -3.0, 2.0],
        [10.0, 3.0, 2.0],
        
        # Wing leading edges
        [0.0, -12.0, 0.5],
        [0.0, 12.0, 0.5],
        
        # Rear points
        [-10.0, -8.0, 0.0],
        [-10.0, 8.0, 0.0],
        [-12.0, 0.0, 1.0],
        
        # Bottom vertices (mirror with different z)
        [10.0, -3.0, -1.0],
        [10.0, 3.0, -1.0],
        [0.0, -12.0, -1.5],
        [0.0, 12.0, -1.5],
        [-10.0, -8.0, -1.0],
        [-10.0, 8.0, -1.0],
        [-12.0, 0.0, -0.5],
    ]
    
    vertices = np.array(vertices)
    
    # Define faces (triangular)
    faces = [
        # Top surfaces
        [0, 1, 2],  # Nose top
        [1, 3, 5],  # Left wing top
        [2, 4, 6],  # Right wing top
        [1, 5, 7],  # Left rear top
        [2, 6, 7],  # Right rear top
        [5, 6, 7],  # Rear top
        
        # Bottom surfaces
        [0, 8, 9],  # Nose bottom
        [8, 10, 12], # Left wing bottom
        [9, 11, 13], # Right wing bottom
        [8, 12, 14], # Left rear bottom
        [9, 13, 14], # Right rear bottom
        [12, 13, 14], # Rear bottom
        
        # Side surfaces
        [0, 1, 8],  # Left nose side
        [0, 2, 9],  # Right nose side
        [1, 3, 10], # Left front side
        [2, 4, 11], # Right front side
        [3, 5, 12], # Left wing side
        [4, 6, 13], # Right wing side
        [5, 7, 14], # Left rear side
        [6, 7, 14], # Right rear side
        
        # Additional faces to close the mesh
        [3, 10, 8],
        [4, 11, 9],
        [12, 14, 13],
    ]
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Fix normals and make watertight
    mesh.fix_normals()
    trimesh.repair.fill_holes(mesh)
    
    return Geometry3D(mesh)


def create_conventional_aircraft_3d() -> Geometry3D:
    """
    Create a conventional aircraft shape with smooth surfaces.
    
    Returns:
        Geometry3D object
    """
    # Use pygmsh to create a smooth fuselage
    with pygmsh.occ.Geometry() as geom:
        # Fuselage as ellipsoid
        fuselage = geom.add_ellipsoid([0.0, 0.0, 0.0], [15.0, 3.0, 2.5])
        
        # Wings as swept boxes
        wing_left = geom.add_box([-2.0, 3.0, -0.5], [10.0, 12.0, 1.0])
        wing_right = geom.add_box([-2.0, -15.0, -0.5], [10.0, 12.0, 1.0])
        
        # Vertical stabilizer
        v_stab = geom.add_box([-12.0, -1.0, 0.0], [4.0, 2.0, 5.0])
        
        # Horizontal stabilizers
        h_stab_left = geom.add_box([-12.0, 1.0, 1.0], [3.0, 4.0, 0.5])
        h_stab_right = geom.add_box([-12.0, -5.0, 1.0], [3.0, 4.0, 0.5])
        
        # Boolean union
        aircraft = geom.boolean_union([fuselage, wing_left, wing_right, 
                                     v_stab, h_stab_left, h_stab_right])
        
        # Generate mesh
        mesh = geom.generate_mesh(dim=3, max_element_size=2.0)
        
    # Convert to trimesh
    trimesh_mesh = trimesh.Trimesh(vertices=mesh.points,
                                  faces=mesh.cells_dict['triangle'])
    
    return Geometry3D(trimesh_mesh)


def create_simple_shape_3d(shape_type: str = 'sphere', 
                          size: float = 10.0) -> Geometry3D:
    """
    Create simple 3D shapes for testing.
    
    Args:
        shape_type: 'sphere', 'cube', 'cone', 'cylinder'
        size: Characteristic size
        
    Returns:
        Geometry3D object
    """
    if shape_type == 'sphere':
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=size)
    elif shape_type == 'cube':
        mesh = trimesh.creation.box(extents=[size, size, size])
    elif shape_type == 'cone':
        mesh = trimesh.creation.cone(radius=size, height=2*size)
    elif shape_type == 'cylinder':
        mesh = trimesh.creation.cylinder(radius=size, height=2*size)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
        
    return Geometry3D(mesh)


def load_from_file(filename: str) -> Geometry3D:
    """
    Load 3D geometry from file.
    
    Args:
        filename: Path to mesh file (STL, OBJ, PLY, etc.)
        
    Returns:
        Geometry3D object
    """
    mesh = trimesh.load(filename)
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, get the first mesh
        mesh = list(mesh.geometry.values())[0]
    return Geometry3D(mesh) 