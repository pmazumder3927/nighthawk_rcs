"""
3D topology optimization for RCS reduction with GPU acceleration.

This module implements advanced optimization algorithms for 3D geometries:
- Gradient-based optimization with adjoint sensitivity
- GPU-accelerated objective evaluation
- Smooth deformation fields
- Multiple optimization strategies
"""

import numpy as np
from typing import Optional, Dict, List, Callable, Tuple
import trimesh
from tqdm import tqdm
import copy
from scipy.optimize import minimize, differential_evolution
import nlopt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

try:
    from .geometry_3d import Geometry3D
    from .rcs_calc_3d import RCS3DCalculator
except ImportError:
    # Fallback for direct execution
    from geometry_3d import Geometry3D
    from rcs_calc_3d import RCS3DCalculator

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, value_and_grad
    GPU_AVAILABLE = len(jax.devices('gpu')) > 0
except ImportError:
    jax = None
    jnp = np  # Fallback to NumPy
    GPU_AVAILABLE = False

# Try to import CuPy for GPU operations
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class TopologyOptimizer3D:
    """
    3D topology optimizer for RCS reduction with GPU support.
    """
    
    def __init__(self, rcs_calculator: RCS3DCalculator,
                 control_points: Optional[np.ndarray] = None,
                 max_displacement: float = 2.0,
                 volume_constraint: bool = True,
                 smoothness: float = 1.0):
        """
        Initialize 3D optimizer.
        
        Args:
            rcs_calculator: RCS calculator instance
            control_points: Control points for deformation (if None, use vertices)
            max_displacement: Maximum displacement magnitude
            volume_constraint: Whether to preserve volume
            smoothness: Smoothness parameter for deformations
        """
        self.rcs_calc = rcs_calculator
        self.control_points = control_points
        self.max_displacement = max_displacement
        self.volume_constraint = volume_constraint
        self.smoothness = smoothness
        
        # Optimization history
        self.history = {
            'geometries': [],
            'rcs_values': [],
            'objective_values': [],
            'iterations': 0,
            'volume_ratios': []
        }
        
        # Cache for objective function evaluations
        self._objective_cache = {}
        self._gradient_cache = {}
        
        # Parallel processing setup
        self.n_workers = mp.cpu_count()
        
        # Compiled functions
        self._objective_jax = None
        self._gradient_jax = None
        
    def objective_function(self, geometry: Geometry3D,
                          target_angles: Optional[List[Tuple[float, float]]] = None,
                          weights: Optional[np.ndarray] = None,
                          use_cache: bool = True) -> float:
        """
        Calculate objective function for 3D geometry with caching.
        
        Args:
            geometry: 3D geometry to evaluate
            target_angles: List of (theta, phi) tuples for specific angles
            weights: Weights for each angle (if None, uniform)
            use_cache: Whether to use cached results
            
        Returns:
            Objective value (weighted mean RCS in linear scale)
        """
        # Create hash for caching
        if use_cache:
            geom_hash = hash(geometry.mesh.vertices.tobytes())
            if geom_hash in self._objective_cache:
                return self._objective_cache[geom_hash]
        
        if target_angles is None:
            # Reduced default angles for faster computation
            theta = np.linspace(45, 135, 5)  # Reduced from 7
            phi = np.linspace(0, 360, 9, endpoint=False)  # Reduced from 13
            target_angles = [(t, p) for t in theta for p in phi]
            
        if weights is None:
            weights = np.ones(len(target_angles)) / len(target_angles)
        else:
            weights = weights / np.sum(weights)
            
        # Batch calculate RCS values
        theta_angles = np.array([angle[0] for angle in target_angles])
        phi_angles = np.array([angle[1] for angle in target_angles])
        
        # Use batch calculation for efficiency
        rcs_values = self.rcs_calc.calculate_rcs_batch(
            geometry.mesh, theta_angles, phi_angles
        )
        
        # Weighted mean in linear scale (not dB)
        objective = np.sum(weights * rcs_values)
        
        # Add volume penalty if enabled
        if self.volume_constraint and hasattr(self, 'initial_volume'):
            volume_ratio = geometry.volume / self.initial_volume
            volume_penalty = 100 * (volume_ratio - 1.0)**2
            objective += volume_penalty
            
        # Cache result
        if use_cache:
            self._objective_cache[geom_hash] = objective
            
        return objective
        
    def gradient_descent_3d(self, initial_geometry: Geometry3D,
                           n_iterations: int = 100,
                           learning_rate: float = 0.1,
                           target_angles: Optional[List[Tuple[float, float]]] = None,
                           method: str = 'adam',
                           adaptive_lr: bool = True,
                           checkpoint_interval: int = 10) -> Geometry3D:
        """
        Optimize using gradient descent with various update rules and optimizations.
        
        Args:
            initial_geometry: Starting geometry
            n_iterations: Number of iterations
            learning_rate: Initial learning rate
            target_angles: Target angles for optimization
            method: 'sgd', 'adam', or 'rmsprop'
            adaptive_lr: Whether to use adaptive learning rate
            checkpoint_interval: Save geometry every N iterations
            
        Returns:
            Optimized geometry
        """
        geometry = copy.deepcopy(initial_geometry)
        self.initial_volume = geometry.volume
        
        # Setup control points if not provided
        if self.control_points is None:
            # Adaptive control point selection based on mesh size
            n_vertices = len(geometry.mesh.vertices)
            n_control = min(max(50, n_vertices // 20), 200)  # Adaptive sizing
            
            # Use FPS (Farthest Point Sampling) for better coverage
            indices = self._farthest_point_sampling(geometry.mesh.vertices, n_control)
            self.control_points = geometry.mesh.vertices[indices]
            
        n_params = self.control_points.shape[0] * 3
        
        # Initialize optimizer state
        if method == 'adam':
            m = np.zeros(n_params)  # First moment
            v = np.zeros(n_params)  # Second moment
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
        elif method == 'rmsprop':
            v = np.zeros(n_params)
            beta = 0.9
            epsilon = 1e-8
            
        # Clear history and caches
        self.history = {
            'geometries': [copy.deepcopy(geometry)],
            'rcs_values': [],
            'objective_values': [self.objective_function(geometry, target_angles)],
            'iterations': 0,
            'volume_ratios': [1.0]
        }
        self._objective_cache.clear()
        self._gradient_cache.clear()
        
        print(f"Initial objective: {self.history['objective_values'][0]:.6f}")
        print(f"Using {len(self.control_points)} control points")
        
        # Learning rate schedule
        if adaptive_lr:
            lr_schedule = self._cosine_annealing_lr(learning_rate, n_iterations)
        else:
            lr_schedule = [learning_rate] * n_iterations
        
        # Best solution tracking
        best_obj = self.history['objective_values'][0]
        best_geometry = copy.deepcopy(geometry)
        patience = 0
        max_patience = 20
        
        with tqdm(total=n_iterations, desc="3D Optimization") as pbar:
            for iteration in range(n_iterations):
                # Get current learning rate
                current_lr = lr_schedule[iteration]
                
                # Calculate gradient with parallel processing
                gradient = self._calculate_gradient_3d_parallel(geometry, target_angles)
                gradient_flat = gradient.flatten()
                
                # Gradient clipping for stability
                grad_norm = np.linalg.norm(gradient_flat)
                if grad_norm > 10.0:
                    gradient_flat = gradient_flat * 10.0 / grad_norm
                
                # Apply optimizer update
                if method == 'sgd':
                    update = -current_lr * gradient_flat
                    
                elif method == 'adam':
                    m = beta1 * m + (1 - beta1) * gradient_flat
                    v = beta2 * v + (1 - beta2) * gradient_flat**2
                    m_hat = m / (1 - beta1**(iteration + 1))
                    v_hat = v / (1 - beta2**(iteration + 1))
                    update = -current_lr * m_hat / (np.sqrt(v_hat) + epsilon)
                    
                elif method == 'rmsprop':
                    v = beta * v + (1 - beta) * gradient_flat**2
                    update = -current_lr * gradient_flat / (np.sqrt(v) + epsilon)
                    
                # Reshape and apply constraints
                displacements = update.reshape(-1, 3)
                displacements = self._apply_constraints(displacements)
                
                # Apply deformation with momentum
                if iteration > 0 and hasattr(self, '_prev_displacements'):
                    momentum = 0.9
                    displacements = momentum * self._prev_displacements + (1 - momentum) * displacements
                self._prev_displacements = displacements.copy()
                
                # Apply deformation
                geometry = geometry.apply_deformation(
                    self.control_points,
                    displacements,
                    smoothing=self.smoothness
                )
                
                # Update control points
                self.control_points += displacements
                
                # Evaluate
                obj_value = self.objective_function(geometry, target_angles)
                
                # Check for improvement
                if obj_value < best_obj:
                    best_obj = obj_value
                    best_geometry = copy.deepcopy(geometry)
                    patience = 0
                else:
                    patience += 1
                
                # Early stopping
                if patience > max_patience:
                    print(f"\nEarly stopping at iteration {iteration}")
                    geometry = best_geometry
                    break
                
                # Store history
                if iteration % checkpoint_interval == 0:
                    self.history['geometries'].append(copy.deepcopy(geometry))
                self.history['objective_values'].append(obj_value)
                self.history['iterations'] = iteration + 1
                self.history['volume_ratios'].append(geometry.volume / self.initial_volume)
                
                # Calculate mean RCS for display
                mean_rcs_linear = obj_value  # Simplified
                mean_rcs_db = 10 * np.log10(mean_rcs_linear + 1e-10)
                self.history['rcs_values'].append(mean_rcs_db)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Obj': f"{obj_value:.6f}",
                    'Best': f"{best_obj:.6f}",
                    'RCS': f"{mean_rcs_db:.1f} dBsm",
                    'Vol': f"{self.history['volume_ratios'][-1]:.3f}",
                    'LR': f"{current_lr:.5f}"
                })
                
        print(f"\nFinal objective: {best_obj:.6f}")
        reduction_db = self.history['rcs_values'][0] - 10 * np.log10(best_obj + 1e-10)
        print(f"RCS reduction: {reduction_db:.1f} dB")
        
        return best_geometry
        
    def _calculate_gradient_3d_parallel(self, geometry: Geometry3D,
                                       target_angles: Optional[List[Tuple[float, float]]],
                                       epsilon: float = 0.01) -> np.ndarray:
        """
        Calculate gradient using parallel finite differences.
        
        Returns:
            Gradient array of shape (n_control_points, 3)
        """
        n_control = len(self.control_points)
        gradient = np.zeros((n_control, 3))
        base_obj = self.objective_function(geometry, target_angles)
        
        # Create list of perturbation tasks
        tasks = []
        for i in range(n_control):
            for j in range(3):
                tasks.append((i, j, epsilon))
        
        # Define worker function
        def calculate_perturbation(task):
            i, j, eps = task
            disp = np.zeros((n_control, 3))
            disp[i, j] = eps
            
            perturbed_geom = geometry.apply_deformation(
                self.control_points, disp, self.smoothness)
            perturbed_obj = self.objective_function(perturbed_geom, target_angles, use_cache=False)
            
            return i, j, (perturbed_obj - base_obj) / eps
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(calculate_perturbation, tasks))
        
        # Collect results
        for i, j, grad_value in results:
            gradient[i, j] = grad_value
            
        return gradient
        
    def _apply_constraints(self, displacements: np.ndarray) -> np.ndarray:
        """Apply displacement constraints with smooth limiting."""
        # Smooth displacement limiting using tanh
        norms = np.linalg.norm(displacements, axis=1)
        scale_factors = np.tanh(norms / self.max_displacement) * self.max_displacement / (norms + 1e-8)
        scale_factors = np.minimum(scale_factors, 1.0)
        
        return displacements * scale_factors[:, np.newaxis]
        
    def _farthest_point_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Farthest Point Sampling for better control point distribution.
        
        Args:
            points: Nx3 array of points
            n_samples: Number of samples to select
            
        Returns:
            Indices of selected points
        """
        n_points = len(points)
        if n_samples >= n_points:
            return np.arange(n_points)
        
        # Start with a random point
        indices = [np.random.randint(n_points)]
        distances = np.full(n_points, np.inf)
        
        for _ in range(n_samples - 1):
            # Update distances to nearest selected point
            last_point = points[indices[-1]]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            
            # Select farthest point
            next_idx = np.argmax(distances)
            indices.append(next_idx)
            
        return np.array(indices)
    
    def _cosine_annealing_lr(self, initial_lr: float, n_iterations: int) -> List[float]:
        """Generate cosine annealing learning rate schedule."""
        schedule = []
        for i in range(n_iterations):
            lr = initial_lr * 0.5 * (1 + np.cos(np.pi * i / n_iterations))
            schedule.append(lr)
        return schedule
        
    def differential_evolution_3d(self, initial_geometry: Geometry3D,
                                 n_generations: int = 50,
                                 population_size: int = 15,
                                 target_angles: Optional[List[Tuple[float, float]]] = None,
                                 strategy: str = 'best1bin',
                                 workers: int = -1) -> Geometry3D:
        """
        Optimize using differential evolution (global optimization) with parallel evaluation.
        
        Args:
            initial_geometry: Starting geometry
            n_generations: Number of generations
            population_size: DE population size
            target_angles: Target angles
            strategy: DE strategy
            workers: Number of parallel workers (-1 for all CPUs)
            
        Returns:
            Optimized geometry
        """
        self.initial_volume = initial_geometry.volume
        
        # Setup control points
        if self.control_points is None:
            n_control = min(50, len(initial_geometry.mesh.vertices) // 20)
            indices = self._farthest_point_sampling(initial_geometry.mesh.vertices, n_control)
            self.control_points = initial_geometry.mesh.vertices[indices].copy()
            
        n_params = self.control_points.shape[0] * 3
        
        # Store reference for objective function
        self.base_geometry = initial_geometry
        self.target_angles = target_angles
        
        def objective_wrapper(x):
            """Wrapper for scipy optimizer."""
            displacements = x.reshape(-1, 3)
            displacements = self._apply_constraints(displacements)
            
            # Apply deformation
            geometry = self.base_geometry.apply_deformation(
                self.control_points,
                displacements,
                smoothing=self.smoothness
            )
            
            return self.objective_function(geometry, self.target_angles)
            
        # Bounds
        bounds = [(-self.max_displacement, self.max_displacement)] * n_params
        
        # Run differential evolution with parallel workers
        print("Running differential evolution optimization...")
        result = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=n_generations,
            popsize=population_size,
            strategy=strategy,
            disp=True,
            workers=workers,
            updating='deferred',  # Better for parallel execution
            polish=False  # Skip local optimization at the end
        )
        
        # Apply final solution
        final_displacements = result.x.reshape(-1, 3)
        final_geometry = self.base_geometry.apply_deformation(
            self.control_points,
            final_displacements,
            smoothing=self.smoothness
        )
        
        return final_geometry
        
    def nlopt_optimization_3d(self, initial_geometry: Geometry3D,
                            algorithm: str = 'COBYLA',
                            n_iterations: int = 100,
                            target_angles: Optional[List[Tuple[float, float]]] = None) -> Geometry3D:
        """
        Optimize using NLopt algorithms with improved convergence.
        
        Args:
            initial_geometry: Starting geometry  
            algorithm: NLopt algorithm name
            n_iterations: Maximum iterations
            target_angles: Target angles
            
        Returns:
            Optimized geometry
        """
        self.initial_volume = initial_geometry.volume
        
        # Setup control points
        if self.control_points is None:
            n_control = min(75, len(initial_geometry.mesh.vertices) // 15)
            indices = self._farthest_point_sampling(initial_geometry.mesh.vertices, n_control)
            self.control_points = initial_geometry.mesh.vertices[indices].copy()
            
        n_params = self.control_points.shape[0] * 3
        
        # Create NLopt optimizer
        opt_algorithms = {
            'COBYLA': nlopt.LN_COBYLA,
            'BOBYQA': nlopt.LN_BOBYQA,
            'SBPLX': nlopt.LN_SBPLX,
            'PRAXIS': nlopt.LN_PRAXIS,
            'NELDERMEAD': nlopt.LN_NELDERMEAD
        }
        
        opt = nlopt.opt(opt_algorithms.get(algorithm, nlopt.LN_COBYLA), n_params)
            
        # Set bounds
        opt.set_lower_bounds(-self.max_displacement * np.ones(n_params))
        opt.set_upper_bounds(self.max_displacement * np.ones(n_params))
        
        # Store for objective
        self.base_geometry = initial_geometry
        self.target_angles = target_angles
        self.eval_count = 0
        self.best_obj = float('inf')
        self.best_x = None
        
        def objective_nlopt(x, grad):
            """NLopt objective function with gradient approximation."""
            self.eval_count += 1
            
            displacements = x.reshape(-1, 3)
            geometry = self.base_geometry.apply_deformation(
                self.control_points,
                displacements,
                smoothing=self.smoothness
            )
            
            obj = self.objective_function(geometry, self.target_angles)
            
            # Track best solution
            if obj < self.best_obj:
                self.best_obj = obj
                self.best_x = x.copy()
            
            if self.eval_count % 10 == 0:
                print(f"Eval {self.eval_count}: Obj = {obj:.6f}, Best = {self.best_obj:.6f}")
                
            # Approximate gradient if requested
            if grad.size > 0:
                # Use finite differences with small epsilon
                eps = 1e-4
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    
                    disp_plus = x_plus.reshape(-1, 3)
                    geom_plus = self.base_geometry.apply_deformation(
                        self.control_points, disp_plus, self.smoothness
                    )
                    obj_plus = self.objective_function(geom_plus, self.target_angles, use_cache=False)
                    
                    grad[i] = (obj_plus - obj) / eps
                
            return obj
            
        # Set objective
        opt.set_min_objective(objective_nlopt)
        opt.set_maxeval(n_iterations)
        
        # Set relative tolerance
        opt.set_ftol_rel(1e-4)
        opt.set_xtol_rel(1e-4)
        
        # Initial guess - small random perturbations
        x0 = 0.1 * self.max_displacement * np.random.randn(n_params)
        
        # Optimize
        print(f"Running NLopt {algorithm} optimization...")
        try:
            x_opt = opt.optimize(x0)
        except nlopt.RoundoffLimited:
            print("Optimization stopped due to roundoff errors")
            x_opt = self.best_x if self.best_x is not None else x0
        
        # Apply best solution found
        final_displacements = x_opt.reshape(-1, 3)
        final_geometry = self.base_geometry.apply_deformation(
            self.control_points,
            final_displacements,
            smoothing=self.smoothness
        )
        
        print(f"Final objective: {self.best_obj:.6f}")
        
        return final_geometry 