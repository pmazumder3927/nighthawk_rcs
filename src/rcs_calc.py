"""
RCS (Radar Cross Section) calculation module.

This module implements simplified physical optics (PO) methods for
calculating the radar cross-section of 2D and 3D geometries.
"""

import numpy as np
from typing import Union, Tuple, Optional
from .geometry import Geometry2D, Geometry3D


class RCSCalculator:
    """
    Calculate radar cross-section using physical optics approximation.
    
    The physical optics method assumes:
    - High frequency (object >> wavelength)
    - Surfaces are perfect electrical conductors
    - Single scattering (no multiple reflections)
    """
    
    def __init__(self, frequency: float = 10e9, polarization: str = 'vertical'):
        """
        Initialize RCS calculator.
        
        Args:
            frequency: Radar frequency in Hz (default 10 GHz, X-band)
            polarization: 'vertical' or 'horizontal' polarization
        """
        self.frequency = frequency
        self.wavelength = 3e8 / frequency  # c / f
        self.k = 2 * np.pi / self.wavelength  # Wave number
        self.polarization = polarization
        
    def calculate_2d_rcs(self, geometry: Geometry2D, 
                        theta_degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate 2D RCS for given incident angles.
        
        Args:
            geometry: 2D geometry object
            theta_degrees: Incident angle(s) in degrees (0° = x-axis)
            
        Returns:
            RCS in square meters (or dBsm if multiple angles)
        """
        theta_rad = np.deg2rad(theta_degrees)
        
        # Handle single angle or array of angles
        if isinstance(theta_rad, float):
            theta_rad = np.array([theta_rad])
            single_angle = True
        else:
            single_angle = False
            
        rcs_values = []
        
        for theta in theta_rad:
            # Incident direction (from angle theta)
            incident_dir = np.array([np.cos(theta), np.sin(theta)])
            
            # Calculate contribution from each edge
            total_field = 0 + 0j
            
            for i in range(len(geometry.edges)):
                edge_center = geometry.edge_centers[i]
                edge_normal = geometry.edge_normals[i]
                edge_length = geometry.edge_lengths[i]
                
                # Check if edge is illuminated (dot product > 0)
                cos_incident = np.dot(-incident_dir, edge_normal)
                if cos_incident > 0:
                    # Physical optics integral for 2D edge
                    # Simplified for perfect electrical conductor
                    
                    # Phase at edge center
                    phase = self.k * np.dot(incident_dir, edge_center)
                    
                    # Edge contribution (2D equivalent of surface integral)
                    # Using scalar approximation for simplicity
                    edge_contribution = (2 * cos_incident * edge_length * 
                                       np.exp(1j * 2 * phase))
                    
                    total_field += edge_contribution
            
            # RCS = |scattered field|² / (4π)
            # For 2D, we use per-unit-length RCS
            rcs = np.abs(total_field)**2 / (4 * np.pi)
            rcs_values.append(rcs)
            
        rcs_values = np.array(rcs_values)
        
        if single_angle:
            return rcs_values[0]
        else:
            return rcs_values
            
    def calculate_2d_rcs_pattern(self, geometry: Geometry2D, 
                                n_angles: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate full 360° RCS pattern for 2D geometry.
        
        Args:
            geometry: 2D geometry object
            n_angles: Number of angles to sample
            
        Returns:
            angles: Array of angles in degrees
            rcs_db: RCS values in dBsm
        """
        angles = np.linspace(0, 360, n_angles, endpoint=False)
        rcs_linear = self.calculate_2d_rcs(geometry, angles)
        
        # Convert to dBsm (decibel square meters)
        # Add small value to avoid log(0)
        rcs_db = 10 * np.log10(rcs_linear + 1e-10)
        
        return angles, rcs_db
        
    def calculate_3d_rcs(self, geometry: Geometry3D,
                        theta: float, phi: float) -> float:
        """
        Calculate 3D RCS for given spherical angles.
        
        Args:
            geometry: 3D geometry object
            theta: Elevation angle in degrees (0° = z-axis)
            phi: Azimuth angle in degrees (0° = x-axis)
            
        Returns:
            RCS in square meters
        """
        # Convert to radians
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        
        # Incident direction in Cartesian coordinates
        incident_dir = np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
        
        # Calculate contribution from each triangular face
        total_field = 0 + 0j
        
        for i in range(geometry.n_faces):
            face_center = geometry.face_centers[i]
            face_normal = geometry.face_normals[i]
            face_area = geometry.face_areas[i]
            
            # Check if face is illuminated
            cos_incident = np.dot(-incident_dir, face_normal)
            if cos_incident > 0:
                # Phase at face center
                phase = self.k * np.dot(incident_dir, face_center)
                
                # Physical optics surface integral (simplified)
                face_contribution = (2 * cos_incident * face_area * 
                                   np.exp(1j * 2 * phase))
                
                total_field += face_contribution
                
        # RCS = (4π/λ²) * |scattered field|²
        rcs = (4 * np.pi / self.wavelength**2) * np.abs(total_field)**2
        
        return rcs


def compute_rcs_statistics(rcs_pattern_db: np.ndarray) -> dict:
    """
    Compute statistics from RCS pattern.
    
    Args:
        rcs_pattern_db: RCS values in dBsm
        
    Returns:
        Dictionary with statistics
    """
    return {
        'mean_rcs_db': np.mean(rcs_pattern_db),
        'max_rcs_db': np.max(rcs_pattern_db),
        'min_rcs_db': np.min(rcs_pattern_db),
        'std_rcs_db': np.std(rcs_pattern_db),
        'dynamic_range_db': np.max(rcs_pattern_db) - np.min(rcs_pattern_db)
    } 