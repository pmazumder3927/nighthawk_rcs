"""
Topology optimization algorithms for RCS reduction.

This module implements various optimization strategies for
minimizing radar cross-section through geometry modification.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import copy

from .geometry import Geometry2D, Geometry3D
from .rcs_calc import RCSCalculator, compute_rcs_statistics


class TopologyOptimizer:
    """
    Topology optimization for RCS reduction.
    
    Implements both gradient-based and evolutionary optimization methods.
    """
    
    def __init__(self, rcs_calculator: RCSCalculator,
                 constraint_type: str = 'vertex_displacement',
                 max_displacement: float = 1.0):
        """
        Initialize optimizer.
        
        Args:
            rcs_calculator: RCS calculator instance
            constraint_type: Type of constraints ('vertex_displacement', 'volume')
            max_displacement: Maximum allowed vertex displacement in meters
        """
        self.rcs_calc = rcs_calculator
        self.constraint_type = constraint_type
        self.max_displacement = max_displacement
        
        # Optimization history
        self.history = {
            'geometries': [],
            'rcs_values': [],
            'max_rcs': [],
            'iterations': 0
        }
        
    def objective_function_2d(self, geometry: Geometry2D,
                             target_angles: Optional[np.ndarray] = None) -> float:
        """
        Calculate objective function for 2D geometry.
        
        Args:
            geometry: 2D geometry to evaluate
            target_angles: Specific angles to minimize RCS (None = all angles)
            
        Returns:
            Objective value (mean RCS in dBsm)
        """
        angles, rcs_db = self.rcs_calc.calculate_2d_rcs_pattern(geometry)
        
        if target_angles is not None:
            # Interpolate to get RCS at target angles
            target_rcs = np.interp(target_angles, angles, rcs_db, period=360)
            return np.mean(target_rcs)
        else:
            # Use statistics over all angles
            stats = compute_rcs_statistics(rcs_db)
            # Weighted objective: prioritize mean with some weight on max
            return 0.7 * stats['mean_rcs_db'] + 0.3 * stats['max_rcs_db']
            
    def gradient_descent_2d(self, initial_geometry: Geometry2D,
                           n_iterations: int = 50,
                           learning_rate: float = 0.1,
                           target_angles: Optional[np.ndarray] = None,
                           smoothing: float = 0.5) -> Geometry2D:
        """
        Optimize 2D geometry using gradient descent.
        
        Args:
            initial_geometry: Starting geometry
            n_iterations: Number of optimization iterations
            learning_rate: Step size for updates
            target_angles: Specific angles to optimize for
            smoothing: Gaussian smoothing parameter for gradients
            
        Returns:
            Optimized geometry
        """
        geometry = copy.deepcopy(initial_geometry)
        
        # Clear history
        self.history = {
            'geometries': [geometry],
            'rcs_values': [self.objective_function_2d(geometry, target_angles)],
            'max_rcs': [],
            'iterations': 0
        }
        
        print(f"Initial RCS: {self.history['rcs_values'][0]:.2f} dBsm")
        
        with tqdm(total=n_iterations, desc="Optimizing") as pbar:
            for iteration in range(n_iterations):
                # Calculate gradient using finite differences
                gradient = self._calculate_gradient_2d(geometry, target_angles)
                
                # Apply smoothing to gradient
                if smoothing > 0:
                    gradient = self._smooth_gradient(gradient, smoothing)
                
                # Normalize gradient
                grad_norm = np.linalg.norm(gradient)
                if grad_norm > 0:
                    gradient = gradient / grad_norm
                    
                # Update vertices
                displacement = -learning_rate * gradient
                
                # Apply constraints
                displacement = self._apply_constraints(displacement)
                
                # Create new geometry
                new_vertices = geometry.vertices + displacement
                geometry = Geometry2D(new_vertices)
                
                # Evaluate and store
                obj_value = self.objective_function_2d(geometry, target_angles)
                self.history['geometries'].append(copy.deepcopy(geometry))
                self.history['rcs_values'].append(obj_value)
                self.history['iterations'] = iteration + 1
                
                # Adaptive learning rate
                if iteration > 0 and obj_value > self.history['rcs_values'][-2]:
                    learning_rate *= 0.5  # Reduce if objective increased
                    
                pbar.update(1)
                pbar.set_postfix({'RCS': f"{obj_value:.2f} dBsm"})
                
        print(f"Final RCS: {self.history['rcs_values'][-1]:.2f} dBsm")
        print(f"Total reduction: {self.history['rcs_values'][0] - self.history['rcs_values'][-1]:.2f} dB")
        
        return geometry
        
    def _calculate_gradient_2d(self, geometry: Geometry2D,
                              target_angles: Optional[np.ndarray] = None,
                              epsilon: float = 0.01) -> np.ndarray:
        """
        Calculate gradient using finite differences.
        
        Args:
            geometry: Current geometry
            target_angles: Target angles for optimization
            epsilon: Finite difference step size
            
        Returns:
            Gradient array of shape (n_vertices, 2)
        """
        gradient = np.zeros_like(geometry.vertices)
        base_obj = self.objective_function_2d(geometry, target_angles)
        
        for i in range(geometry.n_vertices):
            for j in range(2):  # x, y coordinates
                # Perturb vertex
                perturb = np.zeros_like(geometry.vertices)
                perturb[i, j] = epsilon
                
                # Calculate objective with perturbation
                perturbed_geom = geometry.perturb_vertices(perturb)
                perturbed_obj = self.objective_function_2d(perturbed_geom, target_angles)
                
                # Finite difference
                gradient[i, j] = (perturbed_obj - base_obj) / epsilon
                
        return gradient
        
    def _smooth_gradient(self, gradient: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian smoothing to gradient."""
        smoothed = np.zeros_like(gradient)
        for i in range(gradient.shape[1]):
            # Create circular extension for periodic boundary
            extended = np.concatenate([gradient[-3:, i], gradient[:, i], gradient[:3, i]])
            smoothed_extended = gaussian_filter(extended, sigma=sigma)
            smoothed[:, i] = smoothed_extended[3:-3]
        return smoothed
        
    def _apply_constraints(self, displacement: np.ndarray) -> np.ndarray:
        """
        Apply constraints to vertex displacement.
        
        Args:
            displacement: Proposed vertex displacement
            
        Returns:
            Constrained displacement
        """
        if self.constraint_type == 'vertex_displacement':
            # Limit maximum displacement per vertex
            norms = np.linalg.norm(displacement, axis=1)
            mask = norms > self.max_displacement
            displacement[mask] = (displacement[mask] / norms[mask, np.newaxis] * 
                                self.max_displacement)
                                
        return displacement
        
    def evolutionary_optimization_2d(self, initial_geometry: Geometry2D,
                                   population_size: int = 20,
                                   n_generations: int = 50,
                                   mutation_rate: float = 0.1,
                                   crossover_rate: float = 0.7,
                                   target_angles: Optional[np.ndarray] = None) -> Geometry2D:
        """
        Optimize using evolutionary algorithm.
        
        Args:
            initial_geometry: Starting geometry
            population_size: Number of individuals in population
            n_generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            target_angles: Specific angles to optimize for
            
        Returns:
            Best geometry found
        """
        # Initialize population with variations of initial geometry
        population = []
        for _ in range(population_size):
            # Random perturbations
            perturbation = np.random.randn(*initial_geometry.vertices.shape) * 0.5
            perturbation = self._apply_constraints(perturbation)
            new_vertices = initial_geometry.vertices + perturbation
            population.append(Geometry2D(new_vertices))
            
        # Add original
        population[0] = copy.deepcopy(initial_geometry)
        
        # Clear history
        self.history = {
            'geometries': [],
            'rcs_values': [],
            'max_rcs': [],
            'iterations': 0
        }
        
        best_geometry = None
        best_fitness = float('inf')
        
        with tqdm(total=n_generations, desc="Evolution") as pbar:
            for generation in range(n_generations):
                # Evaluate fitness
                fitness = [self.objective_function_2d(geom, target_angles) 
                          for geom in population]
                
                # Track best
                min_idx = np.argmin(fitness)
                if fitness[min_idx] < best_fitness:
                    best_fitness = fitness[min_idx]
                    best_geometry = copy.deepcopy(population[min_idx])
                    
                self.history['geometries'].append(copy.deepcopy(best_geometry))
                self.history['rcs_values'].append(best_fitness)
                self.history['iterations'] = generation + 1
                
                # Selection (tournament)
                new_population = []
                for _ in range(population_size):
                    # Tournament selection
                    idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                    if fitness[idx1] < fitness[idx2]:
                        new_population.append(copy.deepcopy(population[idx1]))
                    else:
                        new_population.append(copy.deepcopy(population[idx2]))
                        
                # Crossover and mutation
                for i in range(0, population_size - 1, 2):
                    if np.random.rand() < crossover_rate:
                        # Crossover between pairs
                        child1, child2 = self._crossover_2d(new_population[i], 
                                                           new_population[i+1])
                        new_population[i] = child1
                        new_population[i+1] = child2
                        
                    # Mutation
                    if np.random.rand() < mutation_rate:
                        new_population[i] = self._mutate_2d(new_population[i])
                    if np.random.rand() < mutation_rate:
                        new_population[i+1] = self._mutate_2d(new_population[i+1])
                        
                # Elitism - keep best individual
                worst_idx = np.argmax(fitness)
                new_population[worst_idx] = copy.deepcopy(best_geometry)
                
                population = new_population
                
                pbar.update(1)
                pbar.set_postfix({'Best RCS': f"{best_fitness:.2f} dBsm"})
                
        print(f"Final RCS: {best_fitness:.2f} dBsm")
        print(f"Total reduction: {self.history['rcs_values'][0] - best_fitness:.2f} dB")
        
        return best_geometry
        
    def _crossover_2d(self, parent1: Geometry2D, parent2: Geometry2D) -> Tuple[Geometry2D, Geometry2D]:
        """Perform crossover between two geometries."""
        # Single-point crossover
        crossover_point = np.random.randint(1, parent1.n_vertices - 1)
        
        child1_vertices = np.vstack([parent1.vertices[:crossover_point],
                                   parent2.vertices[crossover_point:]])
        child2_vertices = np.vstack([parent2.vertices[:crossover_point],
                                   parent1.vertices[crossover_point:]])
                                   
        return Geometry2D(child1_vertices), Geometry2D(child2_vertices)
        
    def _mutate_2d(self, geometry: Geometry2D) -> Geometry2D:
        """Apply mutation to geometry."""
        # Random perturbation to random vertices
        n_mutate = max(1, int(0.2 * geometry.n_vertices))
        mutate_idx = np.random.choice(geometry.n_vertices, n_mutate, replace=False)
        
        perturbation = np.zeros_like(geometry.vertices)
        perturbation[mutate_idx] = np.random.randn(n_mutate, 2) * 0.3
        perturbation = self._apply_constraints(perturbation)
        
        new_vertices = geometry.vertices + perturbation
        return Geometry2D(new_vertices) 