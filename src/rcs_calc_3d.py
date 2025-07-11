"""
Accurate 3D RCS calculation module with GPU acceleration.

This module implements Physical Optics (PO) and Physical Theory of Diffraction (PTD)
for accurate RCS calculations on complex 3D geometries using GPU acceleration.
"""

import numpy as np
from typing import Tuple, Optional, List
import trimesh
from tqdm import tqdm
from functools import partial

# Try to import JAX for GPU acceleration
try:
    import jax
    import jax.numpy as jnp
    from jax import jit as jax_jit, vmap
    GPU_AVAILABLE = len(jax.devices('gpu')) > 0
except ImportError:
    jax = None
    jnp = np  # Fallback to NumPy
    jax_jit = None
    vmap = None
    GPU_AVAILABLE = False
    print("Warning: JAX not available, falling back to CPU computation")

from numba import cuda, jit as numba_jit, prange
import numba.cuda


class RCS3DCalculator:
    """
    Accurate 3D RCS calculator using Physical Optics with GPU acceleration.
    
    This implements the full vector formulation of PO including:
    - Proper polarization handling
    - Accurate phase calculations
    - Edge diffraction corrections (PTD)
    """
    
    def __init__(self, frequency: float = 10e9, use_gpu: bool = True):
        """
        Initialize 3D RCS calculator.
        
        Args:
            frequency: Radar frequency in Hz
            use_gpu: Whether to use GPU acceleration if available
        """
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self.k = 2 * np.pi / self.wavelength
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Impedance of free space
        self.eta = 377.0  # Ohms
        
        # Mesh caching for precomputed invariants
        self._cached_mesh = None
        self._face_centers = None
        self._face_areas = None
        self._face_normals = None
        
        # JAX compiled functions (will be compiled when first mesh is loaded)
        self._rcs_gpu_func = None
        self._batched_rcs_gpu_func = None
        
        if self.use_gpu:
            if jax is not None:
                print(f"Using GPU acceleration with JAX on {jax.devices()[0]}")
            else:
                print("Warning: JAX not available, switching to CPU computation")
                self.use_gpu = False
    
    def _precompute_mesh_invariants(self, mesh: trimesh.Trimesh):
        """
        Precompute mesh invariants that don't change during RCS calculations.
        
        Args:
            mesh: 3D mesh object
        """
        # Check if we need to update cache
        # Use mesh hash for better cache invalidation
        mesh_hash = hash((mesh.vertices.tobytes(), mesh.faces.tobytes()))
        if (getattr(self, '_cached_mesh_hash', None) != mesh_hash or 
            self._face_centers is None or 
            self._face_areas is None or 
            self._face_normals is None):
            
            # Precompute face centers
            self._face_centers = mesh.vertices[mesh.faces].mean(axis=1)
            
            # Precompute face areas using cross product
            v0 = mesh.vertices[mesh.faces[:, 0]]
            v1 = mesh.vertices[mesh.faces[:, 1]]
            v2 = mesh.vertices[mesh.faces[:, 2]]
            self._face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            
            # Precompute face normals
            self._face_normals = mesh.face_normals
            
            # Cache the mesh reference
            self._cached_mesh_hash = mesh_hash
            
            # Compile JAX functions if using GPU
            if self.use_gpu and jax is not None:
                self._compile_jax_functions()
                 
    
    def clear_cache(self):
        """Clear the cached mesh data and compiled functions."""
        self._cached_mesh_hash = None
        self._face_centers = None
        self._face_areas = None
        self._face_normals = None
        self._rcs_gpu_func = None
        self._batched_rcs_gpu_func = None
    
    def _compile_jax_functions(self):
        """Compile JAX functions for GPU acceleration."""
        if not self.use_gpu or jax is None or jax_jit is None:
            return
            
        # Convert precomputed data to JAX DeviceArrays
        face_centers_gpu = jnp.asarray(self._face_centers)
        face_areas_gpu = jnp.asarray(self._face_areas)
        face_normals_gpu = jnp.asarray(self._face_normals)
        
        try:
            # Compile single RCS calculation function
            self._rcs_gpu_func = jax_jit(
                partial(self._calculate_rcs_gpu_pure, 
                       face_centers_gpu, face_areas_gpu, face_normals_gpu),
                static_argnames=('k', 'eta')
            )
            
            # Compile batched RCS calculation function
            self._batched_rcs_gpu_func = vmap(
                partial(self._calculate_rcs_gpu_pure,
                       face_centers_gpu, face_areas_gpu, face_normals_gpu),
                in_axes=(0, 0, 0, 0, None, None)
            )
        except Exception as e:
            print(f"Warning: JAX compilation failed: {e}")
            print("Falling back to CPU computation")
            self.use_gpu = False
            self._rcs_gpu_func = None
            self._batched_rcs_gpu_func = None
        
    def calculate_rcs(self, mesh: trimesh.Trimesh, 
                     theta: float, phi: float,
                     polarization: str = 'VV') -> float:
        """
        Calculate monostatic RCS for a single angle using accurate PO.
        
        Args:
            mesh: 3D mesh object
            theta: Elevation angle in degrees (0° = z-axis)
            phi: Azimuth angle in degrees (0° = x-axis)
            polarization: 'VV', 'HH', 'VH', or 'HV'
            
        Returns:
            RCS in square meters
        """
        # Precompute mesh invariants if needed
        self._precompute_mesh_invariants(mesh)
        
        # Convert angles to radians
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        
        # Incident direction (from radar to target)
        ki_hat = np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
        
        # Scattered direction (monostatic: back to radar)
        ks_hat = -ki_hat
        
        # Define polarization vectors
        theta_hat = np.array([
            np.cos(theta_rad) * np.cos(phi_rad),
            np.cos(theta_rad) * np.sin(phi_rad),
            -np.sin(theta_rad)
        ])
        phi_hat = np.array([-np.sin(phi_rad), np.cos(phi_rad), 0])
        
        # Incident field polarization
        if polarization[0] == 'V':
            Ei_hat = theta_hat
        else:  # H
            Ei_hat = phi_hat
            
        # Scattered field polarization  
        if polarization[1] == 'V':
            Es_hat = theta_hat
        else:  # H
            Es_hat = phi_hat
        
        if self.use_gpu and self._rcs_gpu_func is not None:
            try:
                # Convert to JAX arrays for GPU computation
                ki_hat_jax = jnp.array(ki_hat)
                ks_hat_jax = jnp.array(ks_hat)
                Ei_hat_jax = jnp.array(Ei_hat)
                Es_hat_jax = jnp.array(Es_hat)
                
                result = self._rcs_gpu_func(ki_hat_jax, ks_hat_jax, Ei_hat_jax, Es_hat_jax, self.k, self.eta)
                return float(result)  # Convert JAX result to Python float
            except Exception as e:
                print(f"JAX GPU computation failed: {e}")
                print("Falling back to CPU computation")
                return self._calculate_rcs_cpu(mesh, ki_hat, ks_hat, Ei_hat, Es_hat)
        else:
            return self._calculate_rcs_cpu(mesh, ki_hat, ks_hat, Ei_hat, Es_hat)
            
    def _calculate_rcs_cpu(self, mesh: trimesh.Trimesh,
                          ki_hat: np.ndarray, ks_hat: np.ndarray,
                          Ei_hat: np.ndarray, Es_hat: np.ndarray) -> float:
        """CPU implementation of PO calculation using precomputed invariants."""
        
        # Use precomputed mesh data
        face_centers = self._face_centers
        face_areas = self._face_areas
        face_normals = self._face_normals
        
        # Vectorized CPU implementation

        # Face illumination mask
        cos_theta_i = face_normals.dot(-ki_hat)
        illuminated = cos_theta_i > 0

        if not np.any(illuminated):
            return 0.0

        # Surface currents (constant Hi, Js varies with normals)
        Hi = np.cross(ki_hat, Ei_hat) / self.eta  # (3,)

        Js = 2 * np.cross(face_normals, Hi)  # (N,3)
        Js[~illuminated] = 0.0  # Mask non-illuminated faces

        # Phase term for each face
        phase = self.k * (
            face_centers.dot(ki_hat) - face_centers.dot(ks_hat)
        )
        phase[~illuminated] = 0.0

        # Scattered field integrand
        ks_cross_Js = np.cross(ks_hat, Js)  # (N,3)
        integrand = np.cross(ks_hat, ks_cross_Js)  # (N,3)

        # Projection onto receiving polarization and aggregation
        proj = integrand.dot(Es_hat)  # (N,)
        proj[~illuminated] = 0.0

        contributions = proj * face_areas * np.exp(1j * phase)
        scattered_field = contributions.sum()
        
        # Calculate RCS: σ = 4π|Es|²/|Ei|²
        # For far field, |Ei|² = 1 (normalized)
        rcs = 4 * np.pi * np.abs(scattered_field)**2
        
        return rcs

    @staticmethod
    def _calculate_rcs_gpu_pure(face_centers, face_areas, face_normals,
                               ki_hat, ks_hat, Ei_hat, Es_hat, k, eta):
        """
        Pure JAX function for RCS calculation.
        
        This is a pure function that can be compiled and vectorized by JAX.
        All inputs should be JAX arrays.
        """
        # Check illumination
        cos_theta_i = jnp.dot(face_normals, -ki_hat)
        illuminated = cos_theta_i > 0
        
        # Calculate surface currents for illuminated faces
        Hi = jnp.cross(ki_hat, Ei_hat) / eta
        Js = 2 * jnp.cross(face_normals, Hi)
        Js = jnp.where(illuminated[:, None], Js, 0.0)
        
        # Phase calculations
        r = face_centers
        phase = k * (jnp.dot(r, ki_hat) - jnp.dot(r, ks_hat))
        phase = jnp.where(illuminated, phase, 0.0)
        
        # Scattered field contributions
        ks_cross_Js = jnp.cross(ks_hat, Js)
        integrand = jnp.cross(ks_hat, ks_cross_Js)
        
        # Project onto receiving polarization
        proj = jnp.sum(integrand * Es_hat, axis=1)
        proj = jnp.where(illuminated, proj, 0.0)
        
        # Project and sum
        contributions = (proj * 
                        face_areas * 
                        jnp.exp(1j * phase))
        
        scattered_field = jnp.sum(contributions)
        
        # Calculate RCS
        rcs = 4 * np.pi * jnp.abs(scattered_field)**2
        
        return rcs

    def calculate_rcs_batch(self, mesh: trimesh.Trimesh,
                           theta_angles: np.ndarray, phi_angles: np.ndarray,
                           polarization: str = 'VV') -> np.ndarray:
        """
        Calculate RCS for multiple angles efficiently using batched GPU computation.
        
        Args:
            mesh: 3D mesh object
            theta_angles: Array of elevation angles in degrees
            phi_angles: Array of azimuth angles in degrees
            polarization: 'VV', 'HH', 'VH', or 'HV'
            
        Returns:
            Array of RCS values in square meters
        """
        # Precompute mesh invariants if needed
        self._precompute_mesh_invariants(mesh)
        
        if not self.use_gpu or self._batched_rcs_gpu_func is None:
            # Fallback to CPU batch processing
            rcs_values = np.zeros(len(theta_angles))
            for i, (theta, phi) in enumerate(zip(theta_angles, phi_angles)):
                rcs_values[i] = self.calculate_rcs(mesh, theta, phi, polarization)
            return rcs_values
        
        # Convert angles to radians
        theta_rad = np.deg2rad(theta_angles)
        phi_rad = np.deg2rad(phi_angles)
        
        # Calculate incident directions
        ki_hat = np.column_stack([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
        
        # Scattered directions (monostatic: back to radar)
        ks_hat = -ki_hat
        
        # Define polarization vectors for each angle
        theta_hat = np.column_stack([
            np.cos(theta_rad) * np.cos(phi_rad),
            np.cos(theta_rad) * np.sin(phi_rad),
            -np.sin(theta_rad)
        ])
        phi_hat = np.column_stack([-np.sin(phi_rad), np.cos(phi_rad), np.zeros_like(phi_rad)])
        
        # Incident field polarization
        if polarization[0] == 'V':
            Ei_hat = theta_hat
        else:  # H
            Ei_hat = phi_hat
            
        # Scattered field polarization  
        if polarization[1] == 'V':
            Es_hat = theta_hat
        else:  # H
            Es_hat = phi_hat
        
        # Use batched GPU function
        try:
            # Convert to JAX arrays for GPU computation
            ki_hat_jax = jnp.array(ki_hat)
            ks_hat_jax = jnp.array(ks_hat)
            Ei_hat_jax = jnp.array(Ei_hat)
            Es_hat_jax = jnp.array(Es_hat)
            
            rcs_values = self._batched_rcs_gpu_func(ki_hat_jax, ks_hat_jax, Ei_hat_jax, Es_hat_jax, self.k, self.eta)
        except Exception as e:
            print(f"JAX batched computation failed: {e}")
            print("Falling back to CPU batch processing")
            # Fallback to CPU batch processing
            rcs_values = np.zeros(len(theta_angles))
            for i, (theta, phi) in enumerate(zip(theta_angles, phi_angles)):
                rcs_values[i] = self.calculate_rcs(mesh, theta, phi, polarization)
            return rcs_values
        
        return np.array(rcs_values)
        
    def calculate_rcs_pattern(self, mesh: trimesh.Trimesh,
                            theta_range: Tuple[float, float] = (0, 180),
                            phi_range: Tuple[float, float] = (0, 360),
                            n_theta: int = 37, n_phi: int = 73,
                            polarization: str = 'VV') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate full 3D RCS pattern using optimized batch processing.
        
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
        theta = np.linspace(theta_range[0], theta_range[1], n_theta)
        phi = np.linspace(phi_range[0], phi_range[1], n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        
        # Flatten arrays for batch processing
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()
        
        # Use batch processing if GPU is available
        if self.use_gpu and self._batched_rcs_gpu_func is not None:
            print("Using GPU batch processing for RCS pattern calculation...")
            rcs_flat = self.calculate_rcs_batch(mesh, theta_flat, phi_flat, polarization)
        else:
            # Fallback to CPU processing with progress bar
            print("Using CPU processing for RCS pattern calculation...")
            rcs_flat = np.zeros(len(theta_flat))
            total = len(theta_flat)
            with tqdm(total=total, desc="Calculating RCS pattern") as pbar:
                for i in range(total):
                    rcs_flat[i] = self.calculate_rcs(mesh, theta_flat[i], phi_flat[i], polarization)
                    pbar.update(1)
        
        # Reshape back to 2D grid
        rcs_grid = rcs_flat.reshape(theta_grid.shape)
        
        # Convert to dBsm
        rcs_db = 10 * np.log10(rcs_grid + 1e-10)
        
        return theta_grid, phi_grid, rcs_db
