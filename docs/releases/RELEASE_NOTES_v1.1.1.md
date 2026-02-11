# Release Notes - Version 1.1.1

## 📝 Documentation Update

### Changes
- **Removed PyPI references** from all documentation
- **Updated installation instructions** to use GitHub releases and source installation
- **Cleaned up release guide** to focus on GitHub releases only

### Installation
Download from GitHub release and install:
```bash
wget https://github.com/Butadiene/geopack-vectorize/releases/download/v1.1.1/geopack_vectorized-1.1.1.tar.gz
tar -xzf geopack_vectorized-1.1.1.tar.gz
cd geopack_vectorized-1.1.1
python setup.py install
```

### What's Included
All features from v1.1.0 remain unchanged:
- Complete vectorization of all Tsyganenko models (T89, T96, T01, T04)
- Vectorized IGRF and coordinate transformations
- Enhanced field line tracing with boundary interpolation
- 11 comprehensive tutorial notebooks
- Full test suite and benchmarks

No code changes - this is a documentation-only update.