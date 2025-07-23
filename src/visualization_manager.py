"""
Modular visualization manager for RCS optimization results.

This module provides a high-level interface for creating standardized
visualization sets for pre/post optimization comparisons.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import json
import numpy as np

from .geometry_3d import Geometry3D
from .rcs_calc_3d import RCS3DCalculator
from .visualization_3d import RCSVisualizer3D


class VisualizationManager:
    """Manages creation and organization of RCS visualization sets."""
    
    def __init__(self, output_dir: str = "visualizations", 
                 project_name: Optional[str] = None):
        """
        Initialize visualization manager.
        
        Args:
            output_dir: Base directory for visualization outputs
            project_name: Optional project name for organization
        """
        self.output_dir = Path(output_dir)
        self.project_name = project_name or f"rcs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.project_dir = self.output_dir / self.project_name
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = RCSVisualizer3D(backend='plotly')
        self.files_created = []
        self.metadata = {
            'project_name': self.project_name,
            'created': datetime.now().isoformat(),
            'visualizations': {}
        }
        
    def save_file(self, fig, filename: str, category: str, description: str):
        """Save figure and track in metadata."""
        filepath = self.project_dir / filename
        fig.write_html(str(filepath))
        self.files_created.append(filepath)
        
        self.metadata['visualizations'][filename] = {
            'category': category,
            'description': description,
            'path': str(filepath.relative_to(self.output_dir))
        }
        
        return filepath
        
    def create_geometry_visualization(self, geometry: Geometry3D, 
                                    name: str = "geometry",
                                    show_normals: bool = False) -> Path:
        """Create basic geometry visualization."""
        fig = self.visualizer.plot_geometry_3d(
            geometry, 
            show_normals=show_normals,
            title=f"{name.replace('_', ' ').title()} Geometry"
        )
        
        filename = f"{name}_geometry.html"
        return self.save_file(fig, filename, "geometry", 
                            f"3D visualization of {name} geometry")
        
    def create_rcs_pattern_visualization(self, geometry: Geometry3D,
                                       rcs_calculator: RCS3DCalculator,
                                       name: str = "geometry",
                                       target_angles: Optional[List[Tuple[float, float]]] = None,
                                       theta_range: Tuple[float, float] = (0, 180),
                                       phi_range: Tuple[float, float] = (0, 360),
                                       n_samples: int = 30) -> Path:
        """Create RCS pattern visualization."""
        print(f"Calculating RCS pattern for {name}...")
        theta_grid, phi_grid, rcs_db = rcs_calculator.calculate_rcs_pattern(
            geometry.mesh, theta_range, phi_range, n_samples, n_samples
        )
        
        fig = self.visualizer.plot_rcs_sphere(
            theta_grid, phi_grid, rcs_db,
            title=f"{name.replace('_', ' ').title()} RCS Pattern",
            target_angles=target_angles,
            show_axes=True
        )
        
        filename = f"{name}_rcs_pattern.html"
        return self.save_file(fig, filename, "rcs_pattern",
                            f"3D RCS pattern for {name}")
        
    def create_radar_illumination_visualization(self, geometry: Geometry3D,
                                              target_angles: List[Tuple[float, float]],
                                              name: str = "geometry",
                                              beam_width: float = 15.0) -> Path:
        """Create geometry with radar beam visualization."""
        fig = self.visualizer.plot_geometry_with_radar_beams(
            geometry, target_angles, beam_width,
            title=f"{name.replace('_', ' ').title()} with Radar Illumination"
        )
        
        filename = f"{name}_radar_beams.html"
        return self.save_file(fig, filename, "radar_beams",
                            f"Radar illumination visualization for {name}")
        
    def create_combined_visualization(self, geometry: Geometry3D,
                                    rcs_calculator: RCS3DCalculator,
                                    target_angles: List[Tuple[float, float]],
                                    name: str = "geometry") -> Path:
        """Create combined geometry and RCS pattern view."""
        fig = self.visualizer.plot_combined_view(
            geometry, rcs_calculator, target_angles,
            title=f"{name.replace('_', ' ').title()} - Combined View"
        )
        
        filename = f"{name}_combined.html"
        return self.save_file(fig, filename, "combined",
                            f"Combined geometry and RCS view for {name}")
        
    def create_optimization_comparison(self, 
                                     initial_geometry: Geometry3D,
                                     optimized_geometry: Geometry3D,
                                     rcs_calculator: RCS3DCalculator,
                                     target_angles: Optional[List[Tuple[float, float]]] = None,
                                     optimization_history: Optional[Dict] = None) -> Dict[str, Path]:
        """
        Create full visualization set for optimization comparison.
        
        Args:
            initial_geometry: Pre-optimization geometry
            optimized_geometry: Post-optimization geometry
            rcs_calculator: RCS calculator instance
            target_angles: Radar angles for visualization
            optimization_history: Optional optimization history data
            
        Returns:
            Dictionary of created file paths
        """
        if target_angles is None:
            # Default radar angles
            target_angles = [
                (90, 0),    # Front
                (90, 90),   # Right
                (90, 180),  # Back
                (90, 270),  # Left
                (60, 0),    # Front elevated
            ]
        
        files = {}
        
        # Initial geometry visualizations
        print("\nCreating initial geometry visualizations...")
        files['initial_geometry'] = self.create_geometry_visualization(
            initial_geometry, "initial"
        )
        files['initial_rcs'] = self.create_rcs_pattern_visualization(
            initial_geometry, rcs_calculator, "initial", target_angles
        )
        files['initial_radar'] = self.create_radar_illumination_visualization(
            initial_geometry, target_angles, "initial"
        )
        
        # Optimized geometry visualizations
        print("\nCreating optimized geometry visualizations...")
        files['optimized_geometry'] = self.create_geometry_visualization(
            optimized_geometry, "optimized"
        )
        files['optimized_rcs'] = self.create_rcs_pattern_visualization(
            optimized_geometry, rcs_calculator, "optimized", target_angles
        )
        files['optimized_radar'] = self.create_radar_illumination_visualization(
            optimized_geometry, target_angles, "optimized"
        )
        
        # Comparison visualization
        print("\nCreating comparison visualization...")
        fig_compare = self.visualizer.plot_optimization_comparison_3d(
            initial_geometry, optimized_geometry, rcs_calculator
        )
        files['comparison'] = self.save_file(
            fig_compare, "optimization_comparison.html", "comparison",
            "Side-by-side comparison of initial and optimized geometries"
        )
        
        # Evolution animations if history provided
        if optimization_history:
            print("\nCreating optimization evolution animations...")
            
            # Simple surface evolution
            fig_evolution = self.visualizer.create_surface_evolution_animation(
                optimization_history, skip_frames=max(1, len(optimization_history['geometries']) // 50)
            )
            files['surface_evolution'] = self.save_file(
                fig_evolution, "surface_evolution.html", "animation",
                "Animation of surface evolution"
            )
            
            # Comprehensive evolution with RCS
            print("Creating comprehensive evolution with RCS patterns...")
            fig_comprehensive = self.visualizer.create_comprehensive_evolution(
                optimization_history, rcs_calculator, target_angles,  # Use ALL target angles
                skip_frames=max(1, len(optimization_history['geometries']) // 20),
                n_rcs_samples=30  # Same resolution as standalone RCS patterns
            )
            files['comprehensive_evolution'] = self.save_file(
                fig_comprehensive, "comprehensive_evolution.html", "animation", 
                "Comprehensive evolution showing geometry, RCS pattern, and metrics"
            )
        
        # Save metadata
        self.save_metadata()
        
        return files
        
    def create_single_geometry_analysis(self, geometry: Geometry3D,
                                      rcs_calculator: RCS3DCalculator,
                                      name: str = "geometry",
                                      target_angles: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Path]:
        """
        Create visualization set for a single geometry.
        
        Args:
            geometry: Geometry to analyze
            rcs_calculator: RCS calculator instance
            name: Name for the geometry
            target_angles: Radar angles for visualization
            
        Returns:
            Dictionary of created file paths
        """
        if target_angles is None:
            target_angles = [(90, 0), (90, 90), (90, 180), (90, 270)]
        
        files = {}
        
        print(f"\nCreating visualizations for {name}...")
        files['geometry'] = self.create_geometry_visualization(geometry, name)
        files['rcs_pattern'] = self.create_rcs_pattern_visualization(
            geometry, rcs_calculator, name, target_angles
        )
        files['radar_beams'] = self.create_radar_illumination_visualization(
            geometry, target_angles, name
        )
        files['combined'] = self.create_combined_visualization(
            geometry, rcs_calculator, target_angles, name
        )
        
        # Save metadata
        self.save_metadata()
        
        return files
        
    def save_metadata(self):
        """Save project metadata to JSON file."""
        metadata_file = self.project_dir / "visualization_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        self.files_created.append(metadata_file)
        
    def create_index_html(self):
        """Create an index.html file with links to all visualizations."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RCS Visualizations - {self.project_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .category {{
            background-color: white;
            padding: 20px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .category h2 {{
            color: #555;
            margin-top: 0;
        }}
        .viz-link {{
            display: block;
            padding: 10px;
            margin: 5px 0;
            background-color: #f0f0f0;
            text-decoration: none;
            color: #333;
            border-radius: 4px;
            transition: background-color 0.3s;
        }}
        .viz-link:hover {{
            background-color: #e0e0e0;
        }}
        .description {{
            color: #666;
            font-size: 0.9em;
            margin-left: 20px;
        }}
        .metadata {{
            background-color: #e9e9e9;
            padding: 10px;
            margin-top: 20px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <h1>RCS Visualization Results - {self.project_name}</h1>
    <p>Created: {self.metadata['created']}</p>
"""
        
        # Group visualizations by category
        categories = {}
        for filename, info in self.metadata['visualizations'].items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((filename, info))
        
        # Add links by category
        for category, items in categories.items():
            html_content += f"""
    <div class="category">
        <h2>{category.replace('_', ' ').title()}</h2>
"""
            for filename, info in items:
                html_content += f"""
        <a href="{filename}" class="viz-link">
            {filename.replace('.html', '').replace('_', ' ').title()}
            <div class="description">{info['description']}</div>
        </a>
"""
            html_content += "    </div>\n"
        
        html_content += """
    <div class="metadata">
        <strong>Metadata:</strong> <a href="visualization_metadata.json">visualization_metadata.json</a>
    </div>
</body>
</html>
"""
        
        index_file = self.project_dir / "index.html"
        with open(index_file, 'w') as f:
            f.write(html_content)
        self.files_created.append(index_file)
        
        return index_file
        
    def get_summary(self) -> str:
        """Get summary of created visualizations."""
        summary = f"\nVisualization Summary for '{self.project_name}':\n"
        summary += "=" * 60 + "\n"
        summary += f"Output directory: {self.project_dir}\n"
        summary += f"Total files created: {len(self.files_created)}\n\n"
        
        if self.project_dir / "index.html" in self.files_created:
            summary += f"View all visualizations: {self.project_dir / 'index.html'}\n\n"
        
        summary += "Files created:\n"
        for file in self.files_created:
            summary += f"  - {file.relative_to(self.output_dir)}\n"
            
        return summary