# Release Notes - Version 1.1.2

## 📝 Documentation Update - Fork Attribution

### Changes
- **Fixed GitHub repository URLs** throughout documentation
  - Corrected from `tsssss/geopack` to `Butadiene/geopack-vectorize`
- **Added proper attribution** to clarify project origins
  - Added Attribution section explaining this is a fork of the original geopack
  - Added Author section crediting both original and fork contributors
- **Updated package metadata**
  - Updated author field to reflect community maintenance
  - Added fork reference in package description

### Installation
Download from GitHub release and install:
```bash
wget https://github.com/Butadiene/geopack-vectorize/releases/download/v1.1.2/geopack_vectorized-1.1.2.tar.gz
tar -xzf geopack_vectorized-1.1.2.tar.gz
cd geopack_vectorized-1.1.2
python setup.py install
```

### What's Included
All features from v1.1.0 and v1.1.1 remain unchanged:
- Complete vectorization of all Tsyganenko models (T89, T96, T01, T04)
- Vectorized IGRF and coordinate transformations
- Enhanced field line tracing with boundary interpolation
- 11 comprehensive tutorial notebooks
- Full test suite and benchmarks

No code changes - this is a documentation-only update.