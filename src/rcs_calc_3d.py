"""
3D RCS calculation module using Physical Optics with JAX acceleration.

This module implements Physical Optics (PO) for RCS calculations on 3D geometries.
JAX is used for automatic CPU/GPU acceleration and batch processing.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, List, Optional
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
    
    def __init__(self, frequency: float = 10e9, check_mesh_quality: bool = True):
        """
        Initialize 3D RCS calculator.
        
        Args:
            frequency: Radar frequency in Hz
            check_mesh_quality: Whether to automatically check mesh resolution
        """
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self.k = 2 * np.pi / self.wavelength
        self.eta = 377.0  # Impedance of free space
        self.check_mesh_quality = check_mesh_quality
        
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
            # Check mesh quality if enabled
            if self.check_mesh_quality:
                from src.geometry_3d import Geometry3D
                geom = Geometry3D(mesh)
                quality = geom.check_mesh_resolution(self.wavelength)
                
                if 'warning' in quality:
                    print(f"\n⚠️  MESH QUALITY WARNING:")
                    print(f"   {quality['warning']}")
                    print(f"   Current: {quality['num_faces']} faces, max edge = {quality['max_edge_length']:.3f}m")
                    print(f"   Recommend: edge length < {quality['recommended_edge_length']:.3f}m")
                    print(f"   Consider using geometry.refine_mesh() or remesh_to_target_edge_length()\n")
            
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
        scattered_field *= 1j * k * eta / (4 * jnp.pi)
        
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
    
    
    def calculate_rcs_population(self, meshes: List[trimesh.Trimesh],
                                theta, phi,
                                polarization: str = 'VV') -> np.ndarray:
        """
        Calculate RCS for a population of meshes in batch.
        
        This method efficiently evaluates RCS for multiple meshes at the same angles,
        maximizing GPU utilization by batching all calculations together.
        
        Args:
            meshes: List of 3D mesh objects
            theta: Elevation angle(s) in degrees
            phi: Azimuth angle(s) in degrees  
            polarization: 'VV', 'HH', 'VH', or 'HV'
            
        Returns:
            Array of shape (n_meshes, n_angles) with RCS values
        """
        n_meshes = len(meshes)
        
        # Convert angles to arrays
        theta = np.atleast_1d(np.asarray(theta))
        phi = np.atleast_1d(np.asarray(phi))
        n_angles = len(theta.flatten())
        
        # Convert angles to radians
        theta_rad = np.deg2rad(theta).flatten()
        phi_rad = np.deg2rad(phi).flatten()
        
        # Calculate directions and polarizations (same for all meshes)
        ki_hat = -np.column_stack([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
        ks_hat = -ki_hat
        
        # Polarization vectors
        mask_zero = theta_rad < 1e-6
        theta_hat = np.zeros((n_angles, 3))
        phi_hat = np.zeros((n_angles, 3))
        
        theta_hat[mask_zero] = [1, 0, 0]
        phi_hat[mask_zero] = [0, 1, 0]
        
        theta_hat[~mask_zero] = np.column_stack([
            np.cos(theta_rad[~mask_zero]) * np.cos(phi_rad[~mask_zero]),
            np.cos(theta_rad[~mask_zero]) * np.sin(phi_rad[~mask_zero]),
            -np.sin(theta_rad[~mask_zero])
        ])
        phi_hat[~mask_zero] = np.column_stack([
            -np.sin(phi_rad[~mask_zero]),
            np.cos(phi_rad[~mask_zero]),
            np.zeros(np.sum(~mask_zero))
        ])
        
        # Set polarizations
        Ei_hat = theta_hat if polarization[0] == 'V' else phi_hat
        Es_hat = theta_hat if polarization[1] == 'V' else phi_hat
        
        # Prepare all mesh data
        all_centers = []
        all_normals = []
        all_areas = []
        face_counts = []
        
        for mesh in meshes:
            # Compute mesh properties
            face_centers = mesh.vertices[mesh.faces].mean(axis=1)
            face_normals = mesh.face_normals
            
            # Face areas using cross product
            v0 = mesh.vertices[mesh.faces[:, 0]]
            v1 = mesh.vertices[mesh.faces[:, 1]]
            v2 = mesh.vertices[mesh.faces[:, 2]]
            face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            
            all_centers.append(face_centers)
            all_normals.append(face_normals)
            all_areas.append(face_areas)
            face_counts.append(len(face_centers))
            
        # Stack mesh data with padding for uniform size
        max_faces = max(face_counts)
        
        # Pad mesh data to uniform size
        padded_centers = np.zeros((n_meshes, max_faces, 3))
        padded_normals = np.zeros((n_meshes, max_faces, 3))
        padded_areas = np.zeros((n_meshes, max_faces))
        valid_mask = np.zeros((n_meshes, max_faces), dtype=bool)
        
        for i, (centers, normals, areas, n_faces) in enumerate(
            zip(all_centers, all_normals, all_areas, face_counts)):
            padded_centers[i, :n_faces] = centers
            padded_normals[i, :n_faces] = normals
            padded_areas[i, :n_faces] = areas
            valid_mask[i, :n_faces] = True
            
        # Convert to JAX arrays
        centers_jax = jnp.asarray(padded_centers)
        normals_jax = jnp.asarray(padded_normals)
        areas_jax = jnp.asarray(padded_areas)
        valid_jax = jnp.asarray(valid_mask)
        
        ki_hat_jax = jnp.asarray(ki_hat)
        ks_hat_jax = jnp.asarray(ks_hat)
        Ei_hat_jax = jnp.asarray(Ei_hat)
        Es_hat_jax = jnp.asarray(Es_hat)
        
        # Define population RCS calculation
        @jit
        def calculate_population_rcs(centers, normals, areas, valid_mask,
                                   ki_hats, ks_hats, Ei_hats, Es_hats):
            """Calculate RCS for all meshes and angles in parallel."""
            n_meshes = centers.shape[0]
            n_angles = ki_hats.shape[0]
            
            # Vectorize over meshes and angles
            def single_mesh_all_angles(mesh_idx):
                mesh_centers = centers[mesh_idx]
                mesh_normals = normals[mesh_idx]
                mesh_areas = areas[mesh_idx]
                mesh_valid = valid_mask[mesh_idx]
                
                def single_angle(angle_idx):
                    ki_hat = ki_hats[angle_idx]
                    ks_hat = ks_hats[angle_idx]
                    Ei_hat = Ei_hats[angle_idx]
                    Es_hat = Es_hats[angle_idx]
                    
                    # Check illumination
                    cos_theta_i = jnp.dot(mesh_normals, -ki_hat)
                    illuminated = (cos_theta_i > 0) & mesh_valid
                    
                    # Incident magnetic field
                    Hi = jnp.cross(ki_hat, Ei_hat) / self.eta
                    
                    # Surface currents
                    Js = 2 * jnp.cross(mesh_normals, Hi)
                    Js = jnp.where(illuminated[:, None], Js, 0.0)
                    
                    # Phase calculation
                    phase = self.k * (jnp.dot(mesh_centers, ki_hat) - 
                                     jnp.dot(mesh_centers, ks_hat))
                    phase = jnp.where(illuminated, phase, 0.0)
                    
                    # Project and integrate
                    proj = jnp.sum(Js * Es_hat, axis=1)
                    proj = jnp.where(illuminated, proj, 0.0)
                    
                    contributions = proj * mesh_areas * jnp.exp(1j * phase)
                    scattered_field = jnp.sum(contributions)
                    
                    # Apply PO normalization
                    scattered_field *= 1j * self.k * self.eta / (4 * jnp.pi)
                    
                    # Calculate RCS
                    rcs = 4 * jnp.pi * jnp.abs(scattered_field)**2
                    
                    return rcs
                
                # Vectorize over angles
                return vmap(single_angle)(jnp.arange(n_angles))
            
            # Vectorize over meshes
            return vmap(single_mesh_all_angles)(jnp.arange(n_meshes))
        
        # Calculate RCS for all meshes and angles
        rcs_values = calculate_population_rcs(
            centers_jax, normals_jax, areas_jax, valid_jax,
            ki_hat_jax, ks_hat_jax, Ei_hat_jax, Es_hat_jax
        )
        
        # Convert back to numpy and reshape
        rcs_values = np.array(rcs_values)  # Shape: (n_meshes, n_angles)
        
        # Reshape to match input angle shape
        if theta.ndim > 1:
            rcs_values = rcs_values.reshape(n_meshes, *theta.shape)
            
        return rcs_values
    
    def clear_cache(self):
        """Clear cached mesh data and compiled functions."""
        self._cached_mesh_hash = None
        self._mesh_data = None
        self._rcs_func = None
    
    def calculate_rcs_stationary_phase(self, mesh: trimesh.Trimesh,
                                     theta, phi,
                                     polarization: str = 'VV') -> np.ndarray:
        """
        Calculate RCS using stationary phase approximation for very large objects.
        
        This method is more efficient for electrically very large objects (ka >> 100)
        where the standard PO calculation becomes expensive and the stationary phase
        points dominate the integral.
        
        The method identifies surface points where the phase is stationary
        (specular reflection points) and evaluates the integral asymptotically.
        
        Args:
            mesh: 3D mesh object
            theta: Elevation angle(s) in degrees
            phi: Azimuth angle(s) in degrees
            polarization: 'VV', 'HH', 'VH', or 'HV'
            
        Returns:
            RCS value(s) in m²
        """
        # Convert angles
        theta = np.atleast_1d(np.asarray(theta))
        phi = np.atleast_1d(np.asarray(phi))
        theta_rad = np.deg2rad(theta.flatten())
        phi_rad = np.deg2rad(phi.flatten())
        
        # Incident direction (monostatic case)
        ki_hat = -np.column_stack([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
        
        rcs_values = []
        
        for i in range(len(theta_rad)):
            ki = ki_hat[i]
            
            # Find specular points (where surface normal = -ki for monostatic)
            # These are the stationary phase points
            face_normals = mesh.face_normals
            face_centers = mesh.vertices[mesh.faces].mean(axis=1)
            
            # Dot product to find alignment with incident direction
            alignment = np.dot(face_normals, -ki)
            
            # Select faces that are nearly specular (within tolerance)
            # and illuminated (facing the radar)
            tolerance = 0.1  # Cosine tolerance for specular condition
            specular_mask = (alignment > (1 - tolerance)) & (alignment > 0)
            
            if not np.any(specular_mask):
                # No specular points - use closest faces
                n_closest = min(10, len(face_normals))
                closest_indices = np.argsort(alignment)[-n_closest:]
                specular_mask = np.zeros_like(alignment, dtype=bool)
                specular_mask[closest_indices] = alignment[closest_indices] > 0
            
            # Calculate RCS contribution from specular points
            if np.any(specular_mask):
                # Get specular faces
                spec_normals = face_normals[specular_mask]
                spec_centers = face_centers[specular_mask]
                spec_areas = mesh.area_faces[specular_mask]
                
                # For each specular face, calculate the stationary phase contribution
                # Using the asymptotic expansion of the PO integral
                total_rcs = 0.0
                
                for j in range(len(spec_normals)):
                    n = spec_normals[j]
                    area = spec_areas[j]
                    center = spec_centers[j]
                    
                    # Reflection coefficient (assuming PEC)
                    # For monostatic case, the phase gradient vanishes at specular point
                    cos_theta_i = np.dot(n, -ki)
                    
                    if cos_theta_i > 0:
                        # Gaussian curvature approximation
                        # For flat facets, we approximate local curvature
                        # This is a simplified version - full implementation would
                        # compute actual surface curvature
                        
                        # Effective area considering projection
                        effective_area = area * cos_theta_i
                        
                        # Stationary phase contribution
                        # For flat facets, this reduces to geometric optics
                        rcs_contribution = 4 * np.pi * (effective_area**2) / (self.wavelength**2)
                        
                        # Add phase if there are multiple specular points
                        if len(spec_normals) > 1:
                            phase = 2 * self.k * np.dot(center, ki)
                            rcs_contribution *= np.abs(np.exp(1j * phase))**2
                        
                        total_rcs += rcs_contribution
                
                rcs_values.append(total_rcs)
            else:
                # No illuminated faces
                rcs_values.append(1e-10)
        
        # Reshape to match input
        rcs_values = np.array(rcs_values)
        if theta.ndim > 1:
            rcs_values = rcs_values.reshape(theta.shape)
            
        return rcs_values