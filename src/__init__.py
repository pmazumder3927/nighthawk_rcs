"""
Nighthawk RCS Topology Optimization Package

A simulation environment for radar cross-section optimization
inspired by stealth aircraft design principles.
"""

from .rcs_calc_3d import RCS3DCalculator
from .geometry_3d import Geometry3D as Geometry3DAdvanced
from .optimization_3d import TopologyOptimizer3D
from .visualization_3d import RCSVisualizer3D

__version__ = "0.2.0"
__all__ = [
    # 2D
    "RCSCalculator", "Geometry2D", "Geometry3D", 
    "TopologyOptimizer", "RCSVisualizer",
    # 3D
    "RCS3DCalculator", "Geometry3DAdvanced",
    "TopologyOptimizer3D", "RCSVisualizer3D"
] 