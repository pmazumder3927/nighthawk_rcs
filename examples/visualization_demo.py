"""
Simple demonstration of the new modular visualization system.

This script shows how to use the VisualizationManager for various
visualization tasks without hardcoded file paths.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry_3d import create_simple_shape_3d, create_f117_inspired_3d
from src.rcs_calc_3d import RCS3DCalculator
from src.visualization_manager import VisualizationManager


def demo_single_geometry():
    """Demo: Analyze a single geometry."""
    print("\n" + "=" * 60)
    print("DEMO 1: Single Geometry Analysis")
    print("=" * 60)
    
    # Create geometry and calculator
    geometry = create_simple_shape_3d('cylinder', size=5.0, subdivisions=2)
    rcs_calc = RCS3DCalculator(frequency=10e9)
    
    # Setup visualization manager
    viz_manager = VisualizationManager(
        output_dir='../visualizations',
        project_name='cylinder_analysis'
    )
    
    # Define radar angles
    radar_angles = [
        (90, 0),    # Front
        (90, 90),   # Side
        (45, 0),    # Front elevated
        (135, 180)  # Back depressed
    ]
    
    # Create all visualizations
    files = viz_manager.create_single_geometry_analysis(
        geometry=geometry,
        rcs_calculator=rcs_calc,
        name="cylinder",
        target_angles=radar_angles
    )
    
    # Create index
    viz_manager.create_index_html()
    
    print(viz_manager.get_summary())


def demo_geometry_comparison():
    """Demo: Compare multiple geometries."""
    print("\n" + "=" * 60)
    print("DEMO 2: Geometry Comparison")
    print("=" * 60)
    
    # Create geometries
    sphere = create_simple_shape_3d('sphere', size=5.0, subdivisions=3)
    cube = create_simple_shape_3d('cube', size=5.0, subdivisions=2)
    cylinder = create_simple_shape_3d('cylinder', size=5.0, subdivisions=2)
    
    rcs_calc = RCS3DCalculator(frequency=10e9)
    
    # Setup visualization manager
    viz_manager = VisualizationManager(
        output_dir='../visualizations',
        project_name='shape_comparison'
    )
    
    # Common radar angles
    radar_angles = [(90, 0), (90, 90), (90, 180), (90, 270)]
    
    # Analyze each shape
    for name, geom in [('sphere', sphere), ('cube', cube), ('cylinder', cylinder)]:
        print(f"\nAnalyzing {name}...")
        viz_manager.create_single_geometry_analysis(
            geometry=geom,
            rcs_calculator=rcs_calc,
            name=name,
            target_angles=radar_angles
        )
    
    # Create index
    viz_manager.create_index_html()
    
    print(viz_manager.get_summary())


def demo_quick_visualization():
    """Demo: Quick visualization of specific aspects."""
    print("\n" + "=" * 60)
    print("DEMO 3: Quick Specific Visualizations")
    print("=" * 60)
    
    # Create F-117 geometry
    f117 = create_f117_inspired_3d()
    rcs_calc = RCS3DCalculator(frequency=10e9)
    
    # Setup visualization manager
    viz_manager = VisualizationManager(
        output_dir='../visualizations',
        project_name='f117_quick_viz'
    )
    
    # Just create specific visualizations
    print("\nCreating geometry visualization...")
    viz_manager.create_geometry_visualization(f117, "f117", show_normals=True)
    
    print("\nCreating radar beam visualization...")
    tactical_angles = [(90, 0), (90, 45), (90, -45), (75, 0)]
    viz_manager.create_radar_illumination_visualization(
        f117, tactical_angles, "f117_tactical", beam_width=10.0
    )
    
    print("\nCreating RCS pattern...")
    viz_manager.create_rcs_pattern_visualization(
        f117, rcs_calc, "f117_frontal",
        target_angles=tactical_angles,
        theta_range=(30, 150),  # Focus on tactical angles
        phi_range=(-60, 60),    # Frontal sector only
        n_samples=25
    )
    
    # Create index
    viz_manager.create_index_html()
    
    print(viz_manager.get_summary())


def main():
    """Run all demos."""
    print("VISUALIZATION SYSTEM DEMONSTRATIONS")
    print("==================================")
    
    # Demo 1: Single geometry
    demo_single_geometry()
    
    # Demo 2: Compare geometries
    demo_geometry_comparison()
    
    # Demo 3: Quick specific visualizations
    demo_quick_visualization()
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("Check the visualizations folder for organized results.")
    print("Each project has its own folder with an index.html file.")


if __name__ == "__main__":
    main()