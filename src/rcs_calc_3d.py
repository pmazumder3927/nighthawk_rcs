"""
3D RCS calculation module using Physical Optics with JAX acceleration.

This module implements Physical Optics (PO) for RCS calculations on 3D geometries.
JAX is used for automatic CPU/GPU acceleration and batch processing.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple
import trimesh
from functools import partial


class RCS3DCalculator:
    """
    3D RCS calculator using Physical Optics with JAX acceleration.
    
    This implements the standard PO formulation:
    - Surface currents: J_s = 2n × H
    - Scattered field: E_s = (jkη/4π) ∫∫ J_s exp(jk·r') dS'
    - RCS: σ = 4π|E_s|²/|E_i|²
    """
    
    def __init__(self, frequency: float = 10e9):
        """
        Initialize 3D RCS calculator.
        
        Args:
            frequency: Radar frequency in Hz
        """
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self.k = 2 * np.pi / self.wavelength
        self.eta = 377.0  # Impedance of free space
        
        # Check if GPU is available
        self.device = jax.devices()[0]
        print(f"Using JAX on {self.device}")
        
        # Mesh caching
        self._cached_mesh_hash = None
        self._mesh_data = None
        self._rcs_func = None
    
    def _prepare_mesh_data(self, mesh: trimesh.Trimesh) -> dict:
        """
        Prepare mesh data for JAX computation.
        
        Args:
            mesh: 3D mesh object
            
        Returns:
            Dictionary with JAX arrays of mesh data
        """
        # Check if we need to update cache
        mesh_hash = hash((mesh.vertices.tobytes(), mesh.faces.tobytes()))
        
        if self._cached_mesh_hash != mesh_hash:
            # Compute mesh properties
            face_centers = mesh.vertices[mesh.faces].mean(axis=1)
            face_normals = mesh.face_normals
            
            # Face areas using cross product
            v0 = mesh.vertices[mesh.faces[:, 0]]
            v1 = mesh.vertices[mesh.faces[:, 1]]
            v2 = mesh.vertices[mesh.faces[:, 2]]
            face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            
            # Convert to JAX arrays
            self._mesh_data = {
                'centers': jnp.asarray(face_centers),
                'normals': jnp.asarray(face_normals),
                'areas': jnp.asarray(face_areas)
            }
            
            # Compile RCS functions
            self._compile_functions()
            
            # Update cache
            self._cached_mesh_hash = mesh_hash
            
        return self._mesh_data
    
    def _compile_functions(self):
        """Compile JAX functions for RCS calculation."""
        # Batch calculation - handles both single and multiple angles
        def batch_wrapper(ki_hat_batch, ks_hat_batch, Ei_hat_batch, Es_hat_batch, k, eta):
            return vmap(
                lambda ki, ks, Ei, Es: self._calculate_rcs_jax(
                    self._mesh_data['centers'],
                    self._mesh_data['normals'],
                    self._mesh_data['areas'],
                    ki, ks, Ei, Es, k, eta
                )
            )(ki_hat_batch, ks_hat_batch, Ei_hat_batch, Es_hat_batch)
        
        self._rcs_func = jit(batch_wrapper, static_argnames=('k', 'eta'))
    
    @staticmethod
    def _calculate_rcs_jax(face_centers, face_normals, face_areas,
                          ki_hat, ks_hat, Ei_hat, Es_hat, k, eta):
        """
        Pure JAX function for RCS calculation using Physical Optics.
        
        This is a pure function that can be JIT compiled and vectorized.
        """
        # Check illumination (visible faces)
        cos_theta_i = jnp.dot(face_normals, -ki_hat)
        illuminated = cos_theta_i > 0
        
        # Incident magnetic field: H = (k × E) / η
        Hi = jnp.cross(ki_hat, Ei_hat) / eta
        
        # Surface currents: J_s = 2n × H
        Js = 2 * jnp.cross(face_normals, Hi)
        
        # Apply illumination mask
        Js = jnp.where(illuminated[:, None], Js, 0.0)
        
        # Phase calculation
        phase = k * (jnp.dot(face_centers, ki_hat) - jnp.dot(face_centers, ks_hat))
        phase = jnp.where(illuminated, phase, 0.0)
        
        # Project surface currents onto receive polarization
        proj = jnp.sum(Js * Es_hat, axis=1)
        proj = jnp.where(illuminated, proj, 0.0)
        
        # Integrate with phase
        contributions = proj * face_areas * jnp.exp(1j * phase)
        scattered_field = jnp.sum(contributions)
        
        # Apply PO normalization: E_s = (jkη/4π) × integral
        scattered_field *= (k * eta) / (4 * jnp.pi)
        
        # Calculate RCS: σ = 4π|E_s|²
        rcs = 4 * jnp.pi * jnp.abs(scattered_field)**2
        
        return rcs
    
    def calculate_rcs(self, mesh: trimesh.Trimesh, 
                     theta, phi,
                     polarization: str = 'VV'):
        """
        Calculate monostatic RCS for single or multiple angles.
        
        Args:
            mesh: 3D mesh object
            theta: Elevation angle(s) in degrees (0° = z-axis)
                   Can be a scalar or array
            phi: Azimuth angle(s) in degrees (0° = x-axis)
                 Can be a scalar or array  
            polarization: 'VV', 'HH', 'VH', or 'HV'
            
        Returns:
            RCS in square meters (scalar if single angle, array if multiple)
        """
        # Prepare mesh data
        self._prepare_mesh_data(mesh)
        
        # Convert to arrays to handle both scalar and array inputs
        theta = np.atleast_1d(np.asarray(theta))
        phi = np.atleast_1d(np.asarray(phi))
        
        if theta.shape != phi.shape:
            raise ValueError("theta and phi must have the same shape")
        
        # Store if input was scalar
        was_scalar = theta.ndim == 0 or (theta.ndim == 1 and len(theta) == 1)
        
        # Convert angles to radians
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        
        # Calculate all directions
        ki_hat = -np.column_stack([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ]).reshape(-1, 3)
        ks_hat = -ki_hat
        
        # Polarization vectors
        mask_zero = theta_rad.flatten() < 1e-6
        n_angles = len(theta_rad.flatten())
        
        theta_hat = np.zeros((n_angles, 3))
        phi_hat = np.zeros((n_angles, 3))
        
        # At theta=0
        theta_hat[mask_zero] = [1, 0, 0]
        phi_hat[mask_zero] = [0, 1, 0]
        
        # Away from theta=0
        theta_flat = theta_rad.flatten()
        phi_flat = phi_rad.flatten()
        theta_hat[~mask_zero] = np.column_stack([
            np.cos(theta_flat[~mask_zero]) * np.cos(phi_flat[~mask_zero]),
            np.cos(theta_flat[~mask_zero]) * np.sin(phi_flat[~mask_zero]),
            -np.sin(theta_flat[~mask_zero])
        ])
        phi_hat[~mask_zero] = np.column_stack([
            -np.sin(phi_flat[~mask_zero]),
            np.cos(phi_flat[~mask_zero]),
            np.zeros(np.sum(~mask_zero))
        ])
        
        # Set polarizations
        Ei_hat = theta_hat if polarization[0] == 'V' else phi_hat
        Es_hat = theta_hat if polarization[1] == 'V' else phi_hat
        
        # Convert to JAX arrays and calculate
        ki_hat_jax = jnp.asarray(ki_hat)
        ks_hat_jax = jnp.asarray(ks_hat)
        Ei_hat_jax = jnp.asarray(Ei_hat)
        Es_hat_jax = jnp.asarray(Es_hat)
        
        # Use batch function
        rcs_values = self._rcs_func(ki_hat_jax, ks_hat_jax, Ei_hat_jax, Es_hat_jax, self.k, self.eta)
        rcs_values = np.array(rcs_values)
        
        # Return scalar if input was scalar
        if was_scalar:
            return float(rcs_values[0])
        else:
            return rcs_values.reshape(theta.shape)
    
    def calculate_rcs_pattern(self, mesh: trimesh.Trimesh,
                            theta_range: Tuple[float, float] = (0, 180),
                            phi_range: Tuple[float, float] = (0, 360),
                            n_theta: int = 37, n_phi: int = 73,
                            polarization: str = 'VV') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate full 3D RCS pattern.
        
        Args:
            mesh: 3D mesh object
            theta_range: Elevation angle range in degrees
            phi_range: Azimuth angle range in degrees
            n_theta: Number of theta samples
            n_phi: Number of phi samples
            polarization: Polarization configuration
            
        Returns:
            theta_grid, phi_grid, rcs_db: 2D arrays for plotting
        """
        # Create angle grids
        theta = np.linspace(theta_range[0], theta_range[1], n_theta)
        phi = np.linspace(phi_range[0], phi_range[1], n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        
        # Calculate RCS using unified function
        print(f"Calculating RCS pattern ({theta_grid.size} angles)...")
        rcs_grid = self.calculate_rcs(mesh, theta_grid, phi_grid, polarization)
        
        # Convert to dB
        rcs_db = 10 * np.log10(rcs_grid + 1e-10)
        
        return theta_grid, phi_grid, rcs_db
    
    
    def clear_cache(self):
        """Clear cached mesh data and compiled functions."""
        self._cached_mesh_hash = None
        self._mesh_data = None
        self._rcs_func = None