# Release Notes - Version 1.1.3

## 🚀 New Features

### Field Line Directional Derivatives
- **New vectorized implementation** of the 9 directional derivative formulas for field line analysis
  - Located in `geopack/vectorized/field_line_directional_derivatives_new.py`
  - Implements all 9 formulas: (∂T/∂T)·n, (∂T/∂T)·b, (∂n/∂T)·b, etc.
  - Includes antisymmetry validation functions
  - 10-50x speedup compared to scalar implementations
  - Full compatibility with existing field line tracing functions

### Enhanced Analysis Tools
- **Comprehensive magnetospheric analysis updates**
  - Extended X range analysis for T01 model (-15 Re)
  - Solar Magnetic (SM) coordinate system analysis
  - Seasonal dipole tilt analysis for T96 model
  - Improved parameter effects analysis with 0.2 Re Z increments

### New Documentation
- **Tutorial notebook for directional derivatives** (`examples/notebooks/12_field_line_directional_derivatives_guide.ipynb`)
  - Comprehensive guide to using the 9 directional derivative formulas
  - Examples with both scalar and vectorized implementations
  - Visualization of results along field lines
- **Dipole field analysis notebook** (`examples/notebooks/13_dipole_field_directional_derivatives.ipynb`)
  - Analysis of all 9 formulas for dipole field
  - Validation of antisymmetry relations
  - Color-coded visualizations showing spatial variations

## 🔧 Improvements & Fixes

### Bug Fixes
- Fixed colorbar scale for (∂n/∂b)·b plot in dipole field analysis
- Corrected figure titles to show accurate parameter values
- Restored accidentally removed figures in analysis outputs

### Code Cleanup
- Removed temporary files and obsolete implementations
- Converted curvature scattering analysis from Jupyter notebooks to Python package
- Improved error handling in analysis notebooks

## 📚 Documentation Updates
- Updated CLAUDE.md to reflect new field line directional derivatives implementation
- Added comprehensive documentation for the 9 directional derivative formulas
- Enhanced accuracy reports and implementation guides

## 🛠️ Installation

Install from PyPI:
```bash
pip install geopack-vectorized==1.1.3
```

Or download from GitHub release:
```bash
wget https://github.com/Butadiene/geopack-vectorize/releases/download/v1.1.3/geopack_vectorized-1.1.3.tar.gz
tar -xzf geopack_vectorized-1.1.3.tar.gz
cd geopack_vectorized-1.1.3
python setup.py install
```

## 🔄 Compatibility
- Fully backward compatible with v1.1.x
- Python 3.6+ supported
- NumPy and SciPy dependencies remain unchanged

## 📈 Performance
- Field line directional derivatives: 10-50x speedup
- All existing vectorized functions maintain their performance characteristics
- Memory usage remains linear with input size