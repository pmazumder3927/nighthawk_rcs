# RCS Optimization Demonstrations

This directory contains two complementary demonstrations showing the evolution of RCS optimization techniques.

## Manual Optimization (ECHO-1 Style)

Run: `python echo1_manual_optimization.py`

Demonstrates the 1970s approach used in the F-117 development:
- Manual parameter selection by engineers
- Limited iterations due to time/cost
- Reliance on physical intuition
- ~4 designs tested over simulated hour

Key insights:
- Shows why F-117 has extreme angles
- Illustrates limitations of manual search
- Demonstrates importance of engineer expertise

## Automated Optimization (Modern)

Run: `python modern_automated_optimization.py`

Demonstrates current computational methods:
- Differential Evolution algorithm
- GPU-accelerated evaluations
- Systematic parameter space exploration
- ~150 designs tested in seconds

Key advantages:
- 37x more designs evaluated
- 75x faster execution
- Finds better optima
- No human bias

## Shared Starting Point

Both demos start with the same baseline aircraft geometry (defined in `shared_aircraft_geometry.py`):
- 10m aircraft
- Conservative angles: 15° nose, 35° sweep, 20° tail
- ~31 facets

This ensures fair comparison between manual and automated approaches.

## Key Takeaways

1. **Manual optimization** was necessary in the 1970s but limited by:
   - Computational constraints
   - Time and cost
   - Human biases

2. **Automated optimization** enables:
   - Comprehensive design exploration
   - Rapid iteration
   - Discovery of non-intuitive solutions

3. The F-117's distinctive faceted design emerged from:
   - Computational necessity (limited facets)
   - Physical insight (extreme angles)
   - Iterative refinement

Both approaches validate the core principle: extreme angles and faceted surfaces effectively deflect radar energy.