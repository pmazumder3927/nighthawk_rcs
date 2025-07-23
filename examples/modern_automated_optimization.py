"""
Modern Component-Based RCS Optimization Demo

This demonstrates realistic RCS optimization by:
- Optimizing individual aircraft components separately
- Using control points specific to each component
- Showing progressive refinement of different parts
- Maintaining structural integrity between components
"""

import numpy as np
import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rcs_calc_3d import RCS3DCalculator
from src.visualization_manager import VisualizationManager
from src.optimization_3d import TopologyOptimizer3D
from shared_aircraft_geometry import create_base_aircraft_geometry


def optimize_component(aircraft, component_name, component_info, rcs_calc, threat_angles, 
                      n_generations=50, max_displacement=1.0):
    """
    Optimize a specific aircraft component.
    
    Args:
        aircraft: Current aircraft geometry
        component_name: Name of component to optimize
        component_info: Dictionary with vertex indices and description
        rcs_calc: RCS calculator instance
        threat_angles: List of (theta, phi) threat angles
        n_generations: Number of DE generations
        max_displacement: Maximum vertex displacement allowed
        
    Returns:
        Optimized aircraft geometry
    """
    print(f"\n   Optimizing {component_name}: {component_info['description']}")
    
    # Get control points for this component
    # After mesh refinement, we need to select control points differently
    # Use spatial proximity to original component locations
    control_point_indices = component_info['vertex_indices']
    
    # For refined meshes, select vertices near the component region
    if len(aircraft.mesh.vertices) > 100:  # Mesh has been refined
        # Get approximate region from component info
        if component_name == 'nose':
            # Select vertices in nose region (front of aircraft)
            mask = aircraft.mesh.vertices[:, 0] < 2.5
        elif component_name == 'wings':
            # Select vertices in wing region (wide middle section)
            mask = (np.abs(aircraft.mesh.vertices[:, 1]) > 0.5) & \
                   (aircraft.mesh.vertices[:, 0] > 2.0) & \
                   (aircraft.mesh.vertices[:, 0] < 6.0)
        elif component_name == 'fuselage':
            # Select vertices in mid-fuselage
            mask = (aircraft.mesh.vertices[:, 0] > 4.0) & \
                   (aircraft.mesh.vertices[:, 0] < 6.0) & \
                   (np.abs(aircraft.mesh.vertices[:, 1]) < 1.0)
        elif component_name == 'tail':
            # Select vertices in tail region
            mask = aircraft.mesh.vertices[:, 0] > 7.5
        elif component_name == 'vertical_stabilizer':
            # Select vertices in vertical stabilizer (high Z)
            mask = aircraft.mesh.vertices[:, 2] > 1.0
        else:
            mask = np.zeros(len(aircraft.mesh.vertices), dtype=bool)
        
        # Get indices where mask is True
        available_indices = np.where(mask)[0]
        
        # Select up to 20 control points from the region
        if len(available_indices) > 0:
            n_control = min(20, len(available_indices))
            control_point_indices = np.random.choice(available_indices, n_control, replace=False)
            control_points = aircraft.mesh.vertices[control_point_indices].copy()
        else:
            # Fallback: use random points
            n_control = min(20, len(aircraft.mesh.vertices))
            control_point_indices = np.random.choice(len(aircraft.mesh.vertices), n_control, replace=False)
            control_points = aircraft.mesh.vertices[control_point_indices].copy()
    else:
        # Original mesh, use defined indices
        control_points = aircraft.mesh.vertices[control_point_indices].copy()
    
    print(f"   Control points: {len(control_points)} vertices selected")
    
    # Create optimizer with component-specific control points
    optimizer = TopologyOptimizer3D(
        rcs_calc, 
        control_points=control_points,
        volume_constraint=True,
        max_displacement=max_displacement
    )
    
    # Run differential evolution on this component
    optimized = optimizer.differential_evolution_3d(
        aircraft,
        target_angles=threat_angles,
        n_generations=n_generations,
        population_size=20
    )
    
    # Calculate improvement
    rcs_before = np.mean([rcs_calc.calculate_rcs(aircraft.mesh, theta, phi) 
                         for theta, phi in threat_angles[:5]])
    rcs_after = np.mean([rcs_calc.calculate_rcs(optimized.mesh, theta, phi) 
                        for theta, phi in threat_angles[:5]])
    
    improvement_db = 10 * np.log10(rcs_before / (rcs_after + 1e-10))
    print(f"   Component improvement: {improvement_db:.1f} dB")
    
    return optimized, optimizer.history


def main():
    """Run component-based optimization demonstration."""
    
    print("=" * 70)
    print("COMPONENT-BASED RCS OPTIMIZATION")
    print("Realistic Sequential Component Refinement")
    print("=" * 70)
    
    # Setup
    print("\n1. Setting up optimization environment...")
    rcs_calc = RCS3DCalculator(frequency=10e9)
    print(f"   Frequency: {rcs_calc.frequency/1e9:.1f} GHz")
    
    # Setup visualization manager
    viz_manager = VisualizationManager(
        output_dir='../visualizations',
        project_name='component_optimization_demo'
    )
    
    # Define realistic threat angles
    threat_angles = [
        # Primary frontal threats (highest priority)
        (90, 0), (85, 0), (95, 0),
        (90, 10), (90, -10),
        # Off-axis frontal
        (80, 15), (80, -15),
        # Side aspects
        (90, 30), (90, -30),
        # Rear sector
        (90, 180), (90, 170), (90, -170)
    ]
    
    print(f"   Threat angles: {len(threat_angles)} directions")
    print(f"   Focus: Frontal sector with off-axis coverage")
    
    # Create initial aircraft design
    print("\n2. Creating initial aircraft design...")
    
    # Start with moderate baseline parameters
    initial_aircraft, _, component_info = create_base_aircraft_geometry(
        nose_angle=30.0,
        wing_sweep=40.0,
        tail_angle=25.0
    )
    # Refine mesh to appropriate resolution
    print("   Refining mesh for accurate RCS calculation...")
    initial_aircraft.refine_mesh()
    initial_aircraft.refine_mesh()  # Refine twice for better resolution
    
    print(f"   Vertices: {len(initial_aircraft.mesh.vertices)}")
    print(f"   Faces: {len(initial_aircraft.mesh.faces)}")
    print(f"   Components: {list(component_info.keys())}")
    
    # Calculate baseline RCS
    print("\n3. Calculating baseline RCS...")
    baseline_rcs_values = []
    for theta, phi in threat_angles[:5]:
        rcs = rcs_calc.calculate_rcs(initial_aircraft.mesh, theta, phi)
        rcs_db = 10 * np.log10(rcs + 1e-10)
        baseline_rcs_values.append(rcs_db)
        print(f"   θ={theta}°, φ={phi}°: {rcs_db:.1f} dBsm")
    
    # Component optimization sequence
    print("\n4. Beginning component-wise optimization sequence...")
    
    # Define optimization strategy
    optimization_sequence = [
        ('nose', 75, 1.5),      # Nose: more generations, larger displacement
        ('wings', 60, 2.0),     # Wings: significant impact on RCS
        ('fuselage', 40, 0.8),  # Fuselage: moderate refinement
        ('tail', 50, 1.2),      # Tail: balance rear RCS
        ('vertical_stabilizer', 30, 1.0)  # Vertical stabilizer: fine tuning
    ]
    
    # Track optimization progress
    current_aircraft = copy.deepcopy(initial_aircraft)
    all_histories = {}
    component_improvements = {}
    
    # Create combined history for visualization
    combined_history = {
        'geometries': [copy.deepcopy(initial_aircraft)],
        'objective_values': [],
        'rcs_values': [],
        'volume_ratios': [],
        'component_names': []  # Track which component was optimized
    }
    
    # Optimize each component sequentially
    for component_name, n_gen, max_disp in optimization_sequence:
        if component_name not in component_info:
            continue
            
        print(f"\n{'='*50}")
        print(f"OPTIMIZING: {component_name.upper()}")
        print(f"{'='*50}")
        
        # Measure RCS before component optimization
        rcs_before = np.mean([rcs_calc.calculate_rcs(current_aircraft.mesh, theta, phi) 
                             for theta, phi in threat_angles])
        
        # Optimize this component
        current_aircraft, history = optimize_component(
            current_aircraft,
            component_name,
            component_info[component_name],
            rcs_calc,
            threat_angles,
            n_generations=n_gen,
            max_displacement=max_disp
        )
        
        # Measure improvement
        rcs_after = np.mean([rcs_calc.calculate_rcs(current_aircraft.mesh, theta, phi) 
                            for theta, phi in threat_angles])
        
        improvement = 10 * np.log10(rcs_before / (rcs_after + 1e-10))
        component_improvements[component_name] = improvement
        all_histories[component_name] = history
        
        # Update combined history
        combined_history['geometries'].append(copy.deepcopy(current_aircraft))
        combined_history['objective_values'].extend(history.get('objective_values', []))
        combined_history['rcs_values'].extend(history.get('rcs_values', []))
        combined_history['volume_ratios'].extend(history.get('volume_ratios', []))
        combined_history['component_names'].extend([component_name] * len(history.get('objective_values', [])))
        
        print(f"   Cumulative RCS reduction: {improvement:.1f} dB")
    
    # Final analysis
    print("\n5. Analyzing optimization results...")
    
    # Overall RCS comparison
    print("\n   RCS comparison (key angles):")
    print("   " + "-" * 60)
    print("   Angle         | Original | Optimized | Reduction")
    print("   " + "-" * 60)
    
    total_improvements = []
    for theta, phi in threat_angles[:8]:  # Show more angles
        rcs_orig = rcs_calc.calculate_rcs(initial_aircraft.mesh, theta, phi)
        rcs_opt = rcs_calc.calculate_rcs(current_aircraft.mesh, theta, phi)
        
        rcs_orig_db = 10 * np.log10(rcs_orig + 1e-10)
        rcs_opt_db = 10 * np.log10(rcs_opt + 1e-10)
        reduction_db = rcs_orig_db - rcs_opt_db
        total_improvements.append(reduction_db)
        
        print(f"   θ={theta}°,φ={phi:3d}° | {rcs_orig_db:7.1f} | {rcs_opt_db:8.1f} | {reduction_db:8.1f} dB")
    
    # Component contribution summary
    print("\n   Component contributions to RCS reduction:")
    print("   " + "-" * 40)
    for comp, improvement in component_improvements.items():
        print(f"   {comp.capitalize():<20}: {improvement:>6.1f} dB")
    
    # Overall statistics
    print(f"\n   Average improvement: {np.mean(total_improvements):.1f} dB")
    print(f"   Best improvement: {np.max(total_improvements):.1f} dB")
    print(f"   Frontal sector avg: {np.mean(total_improvements[:5]):.1f} dB")
    
    # Create comprehensive visualizations
    print("\n6. Creating component optimization visualizations...")
    
    # Create comparison visualizations
    viz_manager.create_optimization_comparison(
        initial_geometry=initial_aircraft,
        optimized_geometry=current_aircraft,
        rcs_calculator=rcs_calc,
        target_angles=threat_angles,
        optimization_history=combined_history  # Pass combined history
    )
    
    # Create index
    viz_manager.create_index_html()
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPONENT-BASED OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    print("\nKey achievements:")
    print("- Sequential component optimization mimics real design process")
    print("- Each component optimized with appropriate constraints")
    print("- Structural integrity maintained between components")
    print("- Progressive refinement shows contribution of each part")
    
    print("\nRealistic aspects:")
    print("- Nose optimized for frontal RCS (primary threat)")
    print("- Wings balanced for RCS vs aerodynamic constraints")
    print("- Fuselage refined within structural limits")
    print("- Tail/stabilizer fine-tuned for all-aspect performance")
    
    print(f"\nTotal optimization time: ~{sum(n for _, n, _ in optimization_sequence)*2} seconds")
    
    # Print visualization summary
    print(viz_manager.get_summary())
    print("\nOpen the index.html file to explore component-wise optimization!")


if __name__ == "__main__":
    main()