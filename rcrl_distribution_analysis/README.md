# Rc/RL Distribution Analysis

## Overview

This analysis examines the distribution of Rc/RL (radius of curvature to Larmor radius ratio) values in the magnetosphere during equinox conditions. The critical threshold Rc/RL = 8 marks the boundary where pitch angle scattering becomes significant.

## Key Findings

### XZ Plane Analysis (Y=0, Noon-Midnight Meridian)
- **Result**: Minimal scattering (<1%) even with 30 keV electrons
- **Reason**: The noon-midnight meridian plane intersects regions with strong magnetic fields and gentle curvature
- **Implication**: Pitch angle scattering is rare in this plane

### XY Plane Analysis (Z=0, Equatorial Plane) 
- **10 keV electrons**: 1.22% of region has Rc/RL < 8
- **30 keV electrons**: 2.89% of region has Rc/RL < 8  
- **100 keV electrons**: 9.15% of region has Rc/RL < 8
- **Key observation**: Scattering increases dramatically with energy in the equatorial plane

## Figures

### Figure 1: XZ Plane Distribution (fig01_rcrl_distribution_equinox.png)
Shows the distribution in the noon-midnight meridian plane:
- 6-panel comprehensive analysis
- Spatial distribution with Rc/RL = 8 contour
- Histogram and cumulative distribution
- Radial and Z-profiles
- Scattering intensity regions

Key features:
- Very limited scattering regions
- High Rc/RL values throughout most of the plane
- Median Rc/RL > 100 for 30 keV electrons

### Figure 2: XY Plane Distribution (fig02_rcrl_distribution_xy_equinox.png)
Shows the distribution in the equatorial plane for multiple energies:
- 3x3 grid showing 10, 30, and 100 keV electrons
- Spatial distributions with Rc/RL = 8 contours
- Histograms showing value distributions
- Radial profiles of scattering percentage

Key features:
- Scattering concentrated in tail current sheet
- Strong energy dependence
- Dawn-dusk asymmetry evident
- Peak scattering around 8-12 Re from Earth

## Physical Interpretation

### Why XY Plane Shows More Scattering

1. **Current Sheet Presence**: The equatorial plane contains the magnetotail current sheet where field lines have high curvature
2. **Weaker Fields**: Magnetic field strength is generally lower in the equatorial plane, especially in the tail
3. **Field Line Stretching**: Tail field lines are highly stretched, creating regions of strong curvature

### Energy Dependence

- **Lower energy (10 keV)**: Smaller Larmor radius, requires extreme curvature for scattering
- **Higher energy (100 keV)**: Larger Larmor radius, more easily meets Rc/RL < 8 criterion
- **Relativistic effects**: Become important above ~100 keV, affecting Larmor radius calculation

### Spatial Distribution

- **Near Earth (< 6 Re)**: Strong dipole field, low curvature → minimal scattering
- **Mid-tail (8-12 Re)**: Transition region with optimal conditions for scattering
- **Distant tail (> 15 Re)**: Current sheet thinning increases scattering probability

## Model Parameters

- **Model**: T96 (Tsyganenko 1996)
- **Solar Wind**: Pdyn = 3 nPa (moderate pressure)
- **Storm Conditions**: Dst = -30 nT (moderate storm)
- **IMF**: By = 1 nT, Bz = -3 nT (southward)
- **Dipole Tilt**: -2.56° (equinox conditions)

## Implications for Magnetospheric Physics

1. **Particle Dynamics**: 
   - Electrons in the equatorial plane are more susceptible to pitch angle scattering
   - Creates preferential loss cone filling in certain regions

2. **Wave-Particle Interactions**:
   - Regions with Rc/RL < 8 are prime locations for wave generation
   - May contribute to chorus wave excitation

3. **Radiation Belt Dynamics**:
   - Identifies regions where adiabatic motion breaks down
   - Important for understanding particle precipitation

## Usage

To regenerate the analysis:
```bash
# XZ plane analysis
python rcrl_distribution_analysis.py

# XY plane analysis  
python rcrl_distribution_xy_plane.py
```

## References

- Tsyganenko, N. A. (1996), "Effects of the solar wind conditions on the global magnetospheric configuration"
- Büchner, J., and L. M. Zelenyi (1989), "Regular and chaotic charged particle motion in magnetotaillike field reversals"
- Young et al. (2008), "Magnetic field line curvature scattering of energetic electrons"

---
*Analysis created using geopack-vectorize Python implementation*