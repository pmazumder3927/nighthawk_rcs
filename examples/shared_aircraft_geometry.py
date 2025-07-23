"""
Shared aircraft geometry for optimization demos.

This ensures both ECHO-1 and modern optimization start with the same base design.
"""

import numpy as np
import trimesh
from src.geometry_3d import Geometry3D


def create_base_aircraft_geometry(nose_angle=15.0, wing_sweep=35.0, tail_angle=20.0):
    """
    Create a simple parametric aircraft shape.
    
    This is the starting point for both manual and automated optimization.
    
    Args:
        nose_angle: Nose cone half-angle in degrees
        wing_sweep: Wing sweep angle in degrees  
        tail_angle: Tail cone angle in degrees
        
    Returns:
        Geometry3D object
    """
    # Basic dimensions (10m aircraft)
    length = 10.0
    wingspan = 6.0
    height = 2.0
    
    # Convert angles to radians
    nose_rad = np.deg2rad(nose_angle)
    sweep_rad = np.deg2rad(wing_sweep)
    tail_rad = np.deg2rad(tail_angle)
    
    # Define vertices
    vertices = []
    
    # Nose section (simple pyramid)
    nose_length = length * 0.2
    nose_width = nose_length * np.tan(nose_rad)
    
    vertices.extend([
        [0, 0, 0],  # 0: Nose tip
        [nose_length, 0, height/3],  # 1: Top
        [nose_length, nose_width, 0],  # 2: Right
        [nose_length, -nose_width, 0],  # 3: Left
        [nose_length, 0, -height/4],  # 4: Bottom
    ])
    
    # Fuselage mid-section
    mid_x = length * 0.5
    vertices.extend([
        [mid_x, 0, height/2],  # 5: Top
        [mid_x, wingspan/8, 0],  # 6: Right
        [mid_x, -wingspan/8, 0],  # 7: Left
        [mid_x, 0, -height/3],  # 8: Bottom
    ])
    
    # Wing attachment
    wing_x = length * 0.4
    wing_offset = (wingspan/2) * np.cos(sweep_rad)
    
    vertices.extend([
        [wing_x - wing_offset, wingspan/2, 0],  # 9: Right wing tip
        [wing_x - wing_offset, -wingspan/2, 0],  # 10: Left wing tip
        [wing_x + wingspan/4, wingspan/8, 0],  # 11: Right wing root rear
        [wing_x + wingspan/4, -wingspan/8, 0],  # 12: Left wing root rear
    ])
    
    # Tail section  
    tail_length = length * 0.2
    tail_width = tail_length * np.tan(tail_rad)
    
    vertices.extend([
        [length - tail_length, 0, height/3],  # 13: Tail top
        [length - tail_length, tail_width, 0],  # 14: Right
        [length - tail_length, -tail_width, 0],  # 15: Left
        [length - tail_length, 0, -height/4],  # 16: Bottom
        [length, 0, 0],  # 17: Tail tip
    ])
    
    # Vertical stabilizer
    vertices.extend([
        [length - tail_length/2, 0, height],  # 18: Vertical tail top
    ])
    
    # Define faces
    faces = [
        # Nose cone
        [0, 1, 2], [0, 2, 4], [0, 4, 3], [0, 3, 1],
        [1, 3, 4, 2],  # Nose base
        
        # Forward fuselage
        [1, 5, 2], [2, 5, 6], 
        [3, 7, 1], [1, 7, 5],
        [4, 8, 3], [3, 8, 7],
        [2, 6, 4], [4, 6, 8],
        
        # Wings
        [6, 9, 11], [7, 10, 12],
        
        # Mid to aft fuselage
        [5, 13, 6], [6, 13, 14],
        [7, 15, 5], [5, 15, 13],
        [8, 16, 7], [7, 16, 15],
        [6, 14, 8], [8, 14, 16],
        
        # Tail cone
        [13, 17, 14], [14, 17, 16], [16, 17, 15], [15, 17, 13],
        
        # Vertical stabilizer
        [13, 18, 15], [15, 18, 17], [17, 18, 13],
    ]
    
    # Convert quads to triangles
    triangular_faces = []
    for face in faces:
        if len(face) == 3:
            triangular_faces.append(face)
        elif len(face) == 4:
            triangular_faces.append([face[0], face[1], face[2]])
            triangular_faces.append([face[0], face[2], face[3]])
    
    # Create mesh
    vertices = np.array(vertices)
    faces = np.array(triangular_faces)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    
    # Count facets for ECHO-1 style reporting
    num_facets = len(mesh.faces)
    
    return Geometry3D(mesh), num_facets


def get_optimization_parameters():
    """
    Return the parameter ranges for optimization.
    
    Returns:
        Dictionary with parameter names and (min, max) ranges
    """
    return {
        'nose_angle': (10.0, 70.0),    # degrees
        'wing_sweep': (20.0, 70.0),    # degrees  
        'tail_angle': (10.0, 60.0),    # degrees
    }


def parameters_to_string(nose_angle, wing_sweep, tail_angle):
    """Convert parameters to readable string."""
    return f"Nose: {nose_angle:.0f}°, Wing: {wing_sweep:.0f}°, Tail: {tail_angle:.0f}°"