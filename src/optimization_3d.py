"""
3D topology optimization for RCS reduction with JAX acceleration.

This module implements optimization algorithms for 3D geometries:
- Gradient-based optimization with various update rules
- Differential evolution with JAX acceleration
- NLopt integration for derivative-free methods
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import List, Tuple, Optional
from tqdm import tqdm
import copy
import time
import nlopt

from .geometry_3d import Geometry3D
from .rcs_calc_3d import RCS3DCalculator


class TopologyOptimizer3D:
    """
    3D topology optimizer for RCS reduction using JAX acceleration.
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
        
    def objective_function(self, geometry: Geometry3D,
                          target_angles: Optional[List[Tuple[float, float]]] = None,
                          weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate objective function for 3D geometry.
        
        Args:
            geometry: 3D geometry to evaluate
            target_angles: List of (theta, phi) tuples for specific angles
            weights: Weights for each angle (if None, uniform)
            
        Returns:
            Objective value (weighted mean RCS in linear scale)
        """
        if target_angles is None:
            # Default: sample uniformly on hemisphere
            theta = np.linspace(30, 150, 7)  # Avoid grazing angles
            phi = np.linspace(0, 360, 13, endpoint=False)
            target_angles = [(t, p) for t in theta for p in phi]
            
        if weights is None:
            weights = np.ones(len(target_angles)) / len(target_angles)
        else:
            weights = weights / np.sum(weights)
            
        # Calculate RCS at all angles
        theta_angles = np.array([angle[0] for angle in target_angles])
        phi_angles = np.array([angle[1] for angle in target_angles])
        
        # Use unified calculation function
        rcs_values = self.rcs_calc.calculate_rcs(
            geometry.mesh,
            theta_angles,
            phi_angles
        )
        
        # Weighted mean in linear scale
        objective = np.sum(weights * rcs_values)
        
        # Add volume penalty if enabled
        if self.volume_constraint and hasattr(self, 'initial_volume'):
            volume_ratio = geometry.volume / self.initial_volume
            volume_penalty = 100 * (volume_ratio - 1.0)**2
            objective += volume_penalty
            
        return objective
        
    def gradient_descent_3d(self, initial_geometry: Geometry3D,
                           n_iterations: int = 100,
                           learning_rate: float = 0.1,
                           target_angles: Optional[List[Tuple[float, float]]] = None,
                           method: str = 'adam') -> Geometry3D:
        """
        Optimize using gradient descent with various update rules.
        
        Args:
            initial_geometry: Starting geometry
            n_iterations: Number of iterations
            learning_rate: Initial learning rate
            target_angles: Target angles for optimization
            method: 'sgd', 'adam', or 'rmsprop'
            
        Returns:
            Optimized geometry
        """
        geometry = copy.deepcopy(initial_geometry)
        self.initial_volume = geometry.volume
        
        # Setup control points if not provided
        if self.control_points is None:
            n_vertices = len(geometry.mesh.vertices)
            n_control = min(100, max(4, n_vertices // 10))
            n_control = min(n_control, n_vertices)
            indices = np.random.choice(n_vertices, n_control, replace=False)
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
            
        # Initialize history
        initial_objective = self.objective_function(geometry, target_angles)
        self._initialize_history(geometry, initial_objective)
        
        print(f"Initial objective: {initial_objective:.6f}")
        
        with tqdm(total=n_iterations, desc="3D Optimization") as pbar:
            for iteration in range(n_iterations):
                # Calculate gradient using finite differences
                gradient = self._calculate_gradient(geometry, target_angles)
                gradient_flat = gradient.flatten()
                
                # Apply optimizer update
                if method == 'sgd':
                    update = -learning_rate * gradient_flat
                    
                elif method == 'adam':
                    m = beta1 * m + (1 - beta1) * gradient_flat
                    v = beta2 * v + (1 - beta2) * gradient_flat**2
                    m_hat = m / (1 - beta1**(iteration + 1))
                    v_hat = v / (1 - beta2**(iteration + 1))
                    update = -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    
                elif method == 'rmsprop':
                    v = beta * v + (1 - beta) * gradient_flat**2
                    update = -learning_rate * gradient_flat / (np.sqrt(v) + epsilon)
                    
                # Reshape and apply constraints
                displacements = update.reshape(-1, 3)
                displacements = self._apply_constraints(displacements)
                
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
                
                # Update history
                self._update_history(geometry, obj_value, iteration + 1)
                
                # Get RCS in dB for display
                mean_rcs_db = self.history['rcs_values'][-1]
                
                # Adaptive learning rate
                if iteration > 0 and obj_value > self.history['objective_values'][-2]:
                    learning_rate *= 0.9
                    
                pbar.update(1)
                pbar.set_postfix({
                    'Obj': f"{obj_value:.6f}",
                    'RCS': f"{mean_rcs_db:.1f} dBsm",
                    'Vol': f"{self.history['volume_ratios'][-1]:.3f}"
                })
                
        print(f"Final objective: {self.history['objective_values'][-1]:.6f}")
        reduction_db = self.history['rcs_values'][0] - self.history['rcs_values'][-1]
        print(f"RCS reduction: {reduction_db:.1f} dB")
        
        return geometry
        
    def _calculate_gradient(self, geometry: Geometry3D,
                           target_angles: Optional[List[Tuple[float, float]]],
                           epsilon: float = 0.01) -> np.ndarray:
        """
        Calculate gradient using finite differences.
        
        Returns:
            Gradient array of shape (n_control_points, 3)
        """
        n_control = len(self.control_points)
        gradient = np.zeros((n_control, 3))
        base_obj = self.objective_function(geometry, target_angles)
        
        # Calculate gradient using finite differences
        for i in range(n_control):
            for j in range(3):
                # Create perturbation
                disp = np.zeros((n_control, 3))
                disp[i, j] = epsilon
                
                # Apply perturbation
                perturbed_geom = geometry.apply_deformation(
                    self.control_points, disp, self.smoothness)
                perturbed_obj = self.objective_function(perturbed_geom, target_angles)
                
                # Finite difference
                gradient[i, j] = (perturbed_obj - base_obj) / epsilon
                
        return gradient
        
    def _apply_constraints(self, displacements: np.ndarray) -> np.ndarray:
        """Apply displacement constraints."""
        # Limit maximum displacement
        norms = np.linalg.norm(displacements, axis=1)
        mask = norms > self.max_displacement
        if np.any(mask):
            displacements[mask] = (displacements[mask] / 
                                 norms[mask, np.newaxis] * 
                                 self.max_displacement)
        return displacements
        
    def _initialize_history(self, initial_geometry: Geometry3D, 
                           initial_objective: float) -> None:
        """Initialize optimization history tracking."""
        self.history = {
            'geometries': [copy.deepcopy(initial_geometry)],
            'rcs_values': [],
            'objective_values': [initial_objective],
            'iterations': 0,
            'volume_ratios': [1.0]
        }
        
        # Convert initial objective to RCS in dB for display
        mean_rcs_db = 10 * np.log10(initial_objective + 1e-10)
        self.history['rcs_values'].append(mean_rcs_db)
        
    def _update_history(self, geometry: Geometry3D, objective: float, 
                       iteration: int, store_geometry: bool = True) -> None:
        """Update optimization history with new results."""
        if store_geometry:
            self.history['geometries'].append(copy.deepcopy(geometry))
        
        self.history['objective_values'].append(objective)
        self.history['iterations'] = iteration
        self.history['volume_ratios'].append(geometry.volume / self.initial_volume)
        
        # Convert objective to RCS in dB for display
        mean_rcs_db = 10 * np.log10(objective + 1e-10)
        self.history['rcs_values'].append(mean_rcs_db)
        
    def differential_evolution_3d(self, initial_geometry: Geometry3D,
                                 n_generations: int = 50,
                                 population_size: int = 15,
                                 target_angles: Optional[List[Tuple[float, float]]] = None,
                                 F: float = 0.8,
                                 CR: float = 0.9) -> Geometry3D:
        """
        JAX-accelerated differential evolution optimization.
        
        Args:
            initial_geometry: Starting geometry
            n_generations: Number of generations
            population_size: Population size
            target_angles: Target angles
            F: Differential weight
            CR: Crossover probability
            
        Returns:
            Optimized geometry
        """
        self.initial_volume = initial_geometry.volume
        
        # Setup control points
        if self.control_points is None:
            n_vertices = len(initial_geometry.mesh.vertices)
            n_control = min(50, max(4, n_vertices // 20))
            n_control = min(n_control, n_vertices)
            indices = np.random.choice(n_vertices, n_control, replace=False)
            self.control_points = initial_geometry.mesh.vertices[indices].copy()
            
        n_params = self.control_points.shape[0] * 3
        
        # Store reference for objective function
        self.base_geometry = initial_geometry
        self.target_angles = target_angles
        
        # JIT compile DE step
        @jit
        def de_step(population, F, CR, key):
            """Single DE step compiled with JAX."""
            pop_size, n_dims = population.shape
            
            # Generate random indices for mutation
            keys = jax.random.split(key, pop_size)
            
            def mutate_individual(i, individual_key):
                # Select three random individuals
                indices = jax.random.choice(individual_key, pop_size, (3,), replace=False)
                indices = jnp.where(indices == i, (indices + 1) % pop_size, indices)
                
                # Mutation: vi = x[r1] + F * (x[r2] - x[r3])
                mutant = population[indices[0]] + F * (population[indices[1]] - population[indices[2]])
                
                # Crossover
                cross_key = jax.random.split(individual_key, 2)
                cross_mask = jax.random.uniform(cross_key[0], (n_dims,)) < CR
                
                # Ensure at least one dimension is crossed
                j_rand = jax.random.randint(cross_key[1], (), 0, n_dims)
                cross_mask = cross_mask.at[j_rand].set(True)
                
                trial = jnp.where(cross_mask, mutant, population[i])
                
                # Apply bounds
                trial = jnp.clip(trial, -self.max_displacement, self.max_displacement)
                
                return trial
                
            # Vectorized mutation and crossover
            trials = vmap(mutate_individual)(jnp.arange(pop_size), keys)
            
            return trials
            
        # Initialize population
        key = jax.random.PRNGKey(42)
        population = jax.random.uniform(
            key, 
            (population_size, n_params), 
            minval=-self.max_displacement, 
            maxval=self.max_displacement
        )
        
        # Convert to numpy for objective evaluation
        population = np.array(population)
        
        # Evaluate initial population
        objectives = np.array([
            self.objective_function(self._create_geometry_from_params(p), target_angles)
            for p in population
        ])
        
        best_idx = np.argmin(objectives)
        best_params = population[best_idx].copy()
        best_objective = objectives[best_idx]
        
        # Initialize history
        initial_geometry = self._create_geometry_from_params(best_params)
        self._initialize_history(initial_geometry, best_objective)
        
        print("="*60)
        print("DIFFERENTIAL EVOLUTION OPTIMIZATION")
        print("="*60)
        print(f"Population size: {population_size}")
        print(f"Max generations: {n_generations}")
        print(f"Control points: {len(self.control_points)}")
        print(f"Parameters: {n_params}")
        print(f"F: {F}, CR: {CR}")
        print(f"Initial best objective: {best_objective:.6f}")
        print("-"*60)
        
        # Main evolution loop
        start_time = time.time()
        for generation in range(n_generations):
            # Generate trial population using JAX
            key, subkey = jax.random.split(key)
            trials = de_step(jnp.array(population), F, CR, subkey)
            trials = np.array(trials)
            
            # Evaluate trial population
            trial_objectives = np.array([
                self.objective_function(self._create_geometry_from_params(p), target_angles)
                for p in trials
            ])
            
            # Selection
            improved = trial_objectives < objectives
            population = np.where(improved[:, np.newaxis], trials, population)
            objectives = np.where(improved, trial_objectives, objectives)
            
            # Update best
            current_best_idx = np.argmin(objectives)
            if objectives[current_best_idx] < best_objective:
                best_objective = objectives[current_best_idx]
                best_params = population[current_best_idx].copy()
                
            # Store history every generation
            best_geometry = self._create_geometry_from_params(best_params)
            self._update_history(best_geometry, best_objective, generation + 1)
                
            # Progress update every generation
            elapsed = time.time() - start_time
            print(f"Generation {generation+1:3d}/{n_generations}: "
                  f"Best = {best_objective:.6f}, "
                  f"Mean = {np.mean(objectives):.6f}, "
                  f"Time = {elapsed:.1f}s")
                      
        total_time = time.time() - start_time
        print("-"*60)
        print(f"Optimization complete in {total_time:.1f}s")
        print(f"Final best objective: {best_objective:.6f}")
        
        # Create final geometry
        final_geometry = self._create_geometry_from_params(best_params)
        return final_geometry
        
    def _create_geometry_from_params(self, params: np.ndarray) -> Geometry3D:
        """Create geometry from optimization parameters."""
        displacements = params.reshape(-1, 3)
        displacements = self._apply_constraints(displacements)
        
        return self.base_geometry.apply_deformation(
            self.control_points,
            displacements,
            smoothing=self.smoothness
        )
        
    def nlopt_optimization_3d(self, initial_geometry: Geometry3D,
                            algorithm: str = 'COBYLA',
                            n_iterations: int = 100,
                            target_angles: Optional[List[Tuple[float, float]]] = None) -> Geometry3D:
        """
        Optimize using NLopt algorithms.
        
        Args:
            initial_geometry: Starting geometry  
            algorithm: 'COBYLA', 'BOBYQA', or 'SBPLX'
            n_iterations: Maximum evaluations
            target_angles: Target angles
            
        Returns:
            Optimized geometry
        """
        self.initial_volume = initial_geometry.volume
        
        # Setup control points
        if self.control_points is None:
            n_vertices = len(initial_geometry.mesh.vertices)
            n_control = min(75, max(4, n_vertices // 15))
            n_control = min(n_control, n_vertices)
            indices = np.random.choice(n_vertices, n_control, replace=False)
            self.control_points = initial_geometry.mesh.vertices[indices].copy()
            
        n_params = self.control_points.shape[0] * 3
        
        # Create NLopt optimizer
        opt_map = {
            'COBYLA': nlopt.LN_COBYLA,
            'BOBYQA': nlopt.LN_BOBYQA,
            'SBPLX': nlopt.LN_SBPLX
        }
        opt = nlopt.opt(opt_map.get(algorithm, nlopt.LN_COBYLA), n_params)
        
        # Set bounds
        opt.set_lower_bounds(-self.max_displacement * np.ones(n_params))
        opt.set_upper_bounds(self.max_displacement * np.ones(n_params))
        
        # Store for objective
        self.base_geometry = initial_geometry
        self.target_angles = target_angles
        self.eval_count = 0
        
        # Initialize history
        initial_objective = self.objective_function(initial_geometry, target_angles)
        self._initialize_history(initial_geometry, initial_objective)
        
        def objective_nlopt(x, grad):
            """NLopt objective function."""
            self.eval_count += 1
            
            # Create geometry from parameters
            geometry = self._create_geometry_from_params(x)
            obj = self.objective_function(geometry, self.target_angles)
            
            # Store history periodically
            if self.eval_count % 10 == 0:
                print(f"Eval {self.eval_count}: Obj = {obj:.6f}")
                self._update_history(geometry, obj, self.eval_count)
                
            return obj
            
        # Set objective and run
        opt.set_min_objective(objective_nlopt)
        opt.set_maxeval(n_iterations)
        
        print(f"Running NLopt {algorithm} optimization...")
        x_opt = opt.optimize(np.zeros(n_params))
        
        # Create final geometry
        final_geometry = self._create_geometry_from_params(x_opt)
        
        # Store final result
        final_objective = self.objective_function(final_geometry, self.target_angles)
        self._update_history(final_geometry, final_objective, self.eval_count)
        
        print(f"Optimization complete after {self.eval_count} evaluations")
        print(f"Final objective: {final_objective:.6f}")
        
        return final_geometry