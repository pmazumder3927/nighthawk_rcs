
# Stealth Aircraft Mathematics Visualizations

## Overview
This visualization suite provides comprehensive illustrations of the mathematical concepts behind stealth aircraft technology, as described in the article "Black Diamonds and Bouncing Photons."

## Visualizations Created

### 1. RCS Concept Visualization
- **File**: 1_rcs_concept.png, 1_rcs_concept_interactive.html
- **Concepts**: Basic radar cross section, wave scattering, effective area
- **Key Insights**: How RCS relates to detection and why it's measured in effective area

### 2. RCS Equation Breakdown
- **File**: 2_rcs_equation_breakdown.png
- **Concepts**: Distance scaling, field ratios, effective area interpretation
- **Key Insights**: Mathematical components of the RCS formula σ = lim(r→∞) 4πr²|Es/Ei|²

### 3. Detection Distance Analysis
- **File**: 3_detection_distance.png
- **Concepts**: Fourth root relationship, RCS reduction impact
- **Key Insights**: Why cutting RCS by 100× only reduces detection distance to 56%

### 4. Physical Optics Demonstration
- **File**: 4_physical_optics.png, 4_po_integral_interactive.html
- **Concepts**: PO assumptions, surface currents, computational complexity
- **Key Insights**: How PO simplifies the problem while maintaining accuracy

### 5. Surface Current Visualization
- **File**: 5_surface_currents.png
- **Concepts**: Boundary conditions, current distribution, scattered fields
- **Key Insights**: How surface currents generate scattered electromagnetic fields

### 6. Faceted vs Curved Comparison
- **File**: 6_faceted_vs_curved.png, 6_surface_comparison_3d.html
- **Concepts**: Disco ball effect, controlled reflection, F-117 design
- **Key Insights**: Why faceted surfaces are superior for stealth applications

### 7. Stealth Evolution Timeline
- **File**: 7_stealth_evolution.png, 7_stealth_evolution_interactive.html
- **Concepts**: Technology progression, RCS improvements over time
- **Key Insights**: Evolution from speed-based to shape-based stealth

## Mathematical Concepts Covered

1. **Radar Cross Section (RCS)**: σ = lim(r→∞) 4πr²|Es/Ei|²
2. **Physical Optics Surface Current**: J̃s = 2n̂ × H̃i
3. **PO Scattered Field**: Ẽs = (jkη/4π) ∫S_lit [k̂s × (k̂s × J̃s)] e^(jk(k̂i-k̂s)·r̃) dS
4. **Detection Distance**: R_detection ∝ σ^(1/4)

## Usage Notes

- All static images are saved as high-resolution PNG files (300 DPI)
- Interactive visualizations are saved as HTML files for web viewing
- Color scheme follows a 3Blue1Brown-inspired palette for clarity
- Mathematical formulas are rendered using LaTeX notation

## Dependencies

- numpy, matplotlib, plotly, seaborn
- For interactive features: plotly, ipywidgets
- For 3D visualization: pyvista, vedo (optional)

Generated on: 2025-07-12T09:07:08
