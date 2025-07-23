"""
ECHO-1 Style Manual Optimization Demo

This demonstrates how F-117 engineers optimized aircraft in the 1970s:
- Manual parameter selection
- Limited iterations (time/cost constraints)
- Engineer intuition guiding changes
- Physical understanding of scattering
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rcs_calc_3d import RCS3DCalculator
from src.visualization_3d import RCSVisualizer3D
from shared_aircraft_geometry import (
    create_base_aircraft_geometry, 
    get_optimization_parameters,
    parameters_to_string
)


def echo1_style_optimization():
    """Demonstrate manual iterative optimization as done in ECHO-1 program."""
    
    print("\n" + "="*70)
    print(" "*20 + "ECHO-1 MANUAL OPTIMIZATION")
    print(" "*15 + "Simulating 1970s Design Process")
    print("="*70)
    
    # Setup - 200 MHz like original ECHO-1
    frequency = 200e6
    rcs_calc = RCS3DCalculator(frequency=frequency, check_mesh_quality=False)
    
    print(f"\nRadar: {frequency/1e6:.0f} MHz (VHF)")
    print(f"Wavelength: {rcs_calc.wavelength:.1f} m")
    
    # Target angles (frontal threats)
    threat_angles = [
        (90, 0),    # Head-on
        (90, 30),   # 30° off nose
        (90, -30),
        (90, 60),   # Broadside
        (90, -60),
    ]
    
    print(f"\nThreat angles: {len(threat_angles)} directions")
    print("Optimization goal: Minimize frontal RCS\n")
    
    # Manual iterations - engineer makes educated guesses
    iterations = [
        # Iteration 1: Conservative baseline
        {"nose": 15.0, "sweep": 35.0, "tail": 20.0,
         "rationale": "Conservative design - traditional aircraft angles"},
        
        # Iteration 2: Increase sweep based on theory
        {"nose": 15.0, "sweep": 55.0, "tail": 20.0,
         "rationale": "Increased wing sweep to deflect energy sideways"},
        
        # Iteration 3: Sharp nose for forward scattering
        {"nose": 45.0, "sweep": 55.0, "tail": 30.0,
         "rationale": "Sharper nose and tail to reduce forward return"},
        
        # Iteration 4: Extreme angles like F-117
        {"nose": 60.0, "sweep": 67.5, "tail": 40.0,
         "rationale": "Extreme angles - energy deflection priority"},
    ]
    
    results = []
    
    print("="*70)
    print("MANUAL ITERATION PROCESS")
    print("="*70)
    
    for i, params in enumerate(iterations):
        print(f"\n--- Iteration {i+1} ---")
        print(f"Engineer decision: {params['rationale']}")
        print(f"Parameters: {parameters_to_string(params['nose'], params['sweep'], params['tail'])}")
        
        # Create aircraft
        aircraft, num_facets = create_base_aircraft_geometry(
            params['nose'], params['sweep'], params['tail']
        )
        
        print(f"Geometry: {num_facets} facets")
        
        # Calculate RCS
        start_time = time.time()
        rcs_values = []
        for theta, phi in threat_angles:
            rcs = rcs_calc.calculate_rcs(aircraft.mesh, theta, phi)
            rcs_values.append(10 * np.log10(rcs))
        calc_time = time.time() - start_time
        
        mean_rcs = np.mean(rcs_values)
        min_rcs = np.min(rcs_values)
        max_rcs = np.max(rcs_values)
        
        print(f"RCS Results:")
        print(f"  Mean: {mean_rcs:.1f} dBsm")
        print(f"  Range: {min_rcs:.1f} to {max_rcs:.1f} dBsm")
        print(f"  Calculation time: {calc_time:.2f}s")
        
        # Engineer analysis
        if i == 0:
            print("Analysis: Baseline too reflective, need more aggressive angles")
        elif i == 1:
            print("Analysis: Better but nose still too blunt")
        elif i == 2:
            print("Analysis: Improvement! Sharp features helping")
        elif i == 3:
            print("Analysis: Diminishing returns, may be near optimum")
        
        results.append({
            'iteration': i + 1,
            'params': params,
            'mean_rcs': mean_rcs,
            'rcs_values': rcs_values,
            'aircraft': aircraft,
            'num_facets': num_facets
        })
    
    # Summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Evolution of parameters
    iterations_x = [r['iteration'] for r in results]
    nose_angles = [r['params']['nose'] for r in results]
    sweep_angles = [r['params']['sweep'] for r in results]
    
    ax1.plot(iterations_x, nose_angles, 'o-', label='Nose angle', linewidth=2, markersize=8)
    ax1.plot(iterations_x, sweep_angles, 's-', label='Wing sweep', linewidth=2, markersize=8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Manual Parameter Selection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(iterations_x)
    
    # RCS evolution
    mean_rcs_values = [r['mean_rcs'] for r in results]
    ax2.plot(iterations_x, mean_rcs_values, 'ro-', linewidth=2, markersize=10)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean RCS (dBsm)')
    ax2.set_title('RCS Improvement')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(iterations_x)
    
    for i, (x, y) in enumerate(zip(iterations_x, mean_rcs_values)):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # Facet count
    facet_counts = [r['num_facets'] for r in results]
    ax3.bar(iterations_x, facet_counts, alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Number of Facets')
    ax3.set_title('Geometric Complexity')
    ax3.set_xticks(iterations_x)
    
    for i, (x, y) in enumerate(zip(iterations_x, facet_counts)):
        ax3.text(x, y + 0.5, str(y), ha='center')
    
    # Summary text
    ax4.axis('off')
    ax4.text(0.5, 0.95, 'ECHO-1 Manual Optimization Summary', 
             ha='center', fontsize=14, weight='bold', transform=ax4.transAxes)
    
    summary_text = f"""
Process:
• 4 iterations over ~1 hour
• Engineer-guided parameter selection
• Physical intuition drives changes

Results:
• Initial RCS: {results[0]['mean_rcs']:.1f} dBsm
• Final RCS: {results[-1]['mean_rcs']:.1f} dBsm
• Improvement: {results[0]['mean_rcs'] - results[-1]['mean_rcs']:.1f} dB

Key Insights:
• Sharp angles deflect energy
• Extreme sweep reduces frontal RCS
• Faceted design enables analysis
• Limited iterations due to time/cost

Final Design:
{parameters_to_string(
    results[-1]['params']['nose'],
    results[-1]['params']['sweep'], 
    results[-1]['params']['tail']
)}
"""
    
    ax4.text(0.05, 0.85, summary_text, fontsize=10, 
             verticalalignment='top', transform=ax4.transAxes,
             family='monospace')
    
    plt.suptitle('ECHO-1 Style Manual Optimization Process', fontsize=16)
    plt.tight_layout()
    plt.savefig('../visualizations/echo1_manual_process.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create 3D visualization of evolution
    visualizer = RCSVisualizer3D(backend='plotly')
    
    # Show initial vs final
    fig = visualizer.plot_optimization_comparison_3d(
        results[0]['aircraft'],
        results[-1]['aircraft'],
        rcs_calc
    )
    fig.update_layout(title="ECHO-1 Manual Optimization: Initial vs Final Design")
    fig.write_html('../visualizations/echo1_manual_comparison.html')
    
    print("\n" + "="*70)
    print("MANUAL OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nTotal iterations: {len(iterations)}")
    print(f"Time simulated: ~1 hour of engineer time")
    print(f"Designs tested: {len(iterations)}")
    print(f"Final improvement: {results[0]['mean_rcs'] - results[-1]['mean_rcs']:.1f} dB")
    
    print("\nLimitations of manual approach:")
    print("- Limited exploration of design space")
    print("- Dependent on engineer intuition")
    print("- Time/cost constraints limit iterations")
    print("- May miss non-intuitive optimal solutions")
    
    print("\nFiles generated:")
    print("  - echo1_manual_process.png")
    print("  - echo1_manual_comparison.html")
    
    return results


if __name__ == "__main__":
    os.makedirs('../visualizations', exist_ok=True)
    results = echo1_style_optimization()