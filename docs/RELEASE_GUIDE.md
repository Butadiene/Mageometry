# Release Guide for geopack-vectorized

This guide outlines the steps to release a new version of the geopack-vectorized package.

## Pre-Release Checklist

### 1. Code Preparation
- [ ] Ensure all tests pass: `python -m unittest discover tests/`
- [ ] Run benchmarks to verify performance: `python tests/benchmark_models.py`
- [ ] Check all notebooks execute without errors
- [ ] Remove any temporary or debug files
- [ ] Ensure no sensitive information in code

### 2. Version Updates
Update version numbers in these files:
- [ ] `setup.py` - version field
- [ ] `pyproject.toml` - version field
- [ ] `geopack/__init__.py` - __version__ variable
- [ ] `README.md` - version badges and download links

### 3. Documentation Updates
- [ ] Update `RELEASE_NOTES_vX.X.X.md` with changes
- [ ] Update `README.md` with new features
- [ ] Update `CLAUDE.md` if project structure changed
- [ ] Check all import examples are current
- [ ] Update performance/accuracy numbers if improved

### 4. Final Testing
```bash
# Clean install test
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate
python setup.py develop
python tests/test_geopack1.py
python tests/test_vectorized_models.py
deactivate
rm -rf test_env
```

## Release Process

### 1. Git Tasks
```bash
# Ensure you're on master/main branch
git checkout master

# Ensure branch is up to date
git pull origin master

# Create and push version tag
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin v1.1.0

# Push all commits
git push origin master
```

### 2. Build Distribution
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution
python setup.py sdist

# Check the built package
ls -la dist/
# Should see:
# - geopack_vectorized-1.1.0.tar.gz
```

### 3. Test Distribution
```bash
# Test installation from built package
tar -xzf dist/geopack_vectorized-1.1.0.tar.gz
cd geopack_vectorized-1.1.0
python setup.py install

# Run basic import test
python -c "import geopack; print(geopack.__version__)"
```

### 4. GitHub Release
1. Go to https://github.com/Butadiene/geopack-vectorize/releases
2. Click "Create a new release"
3. Choose the tag you created (v1.1.0)
4. Title: "geopack-vectorized v1.1.0"
5. Copy content from `RELEASE_NOTES_v1.1.0.md`
6. Attach the built files from `dist/`
7. Click "Publish release"

### 5. Post-Release Verification
```bash
# Download from GitHub release and test
wget https://github.com/Butadiene/geopack-vectorize/releases/download/v1.1.0/geopack_vectorized-1.1.0.tar.gz
tar -xzf geopack_vectorized-1.1.0.tar.gz
cd geopack_vectorized-1.1.0
python setup.py install
python -c "import geopack; print(geopack.__version__)"
```

## Version Numbering

Follow semantic versioning (https://semver.org/):
- **Major** (X.0.0): Breaking API changes
- **Minor** (1.X.0): New features, backwards compatible
- **Patch** (1.1.X): Bug fixes, backwards compatible

Examples:
- 1.0.0 → 1.1.0: Added vectorized field line tracing (new feature)
- 1.1.0 → 1.1.1: Fixed bug in T96 calculation
- 1.1.0 → 2.0.0: Changed function signatures (breaking change)

## Troubleshooting

### Build Errors
```bash
# Ensure latest setuptools
python -m pip install --upgrade setuptools

# Clear caches
rm -rf build/ dist/ *.egg-info __pycache__
find . -type d -name __pycache__ -exec rm -rf {} +
```

### Installation Issues
```bash
# If installation fails, try:
python setup.py install --user

# Or install dependencies first:
python -m pip install numpy scipy
python setup.py install
```

## Maintenance Tasks

### Regular Updates
- Keep dependencies updated in setup.py
- Update Python version compatibility
- Review and update documentation
- Check for security vulnerabilities

### Deprecation Process
1. Add deprecation warnings in code
2. Document in release notes
3. Provide migration guide
4. Remove in next major version

## Quick Release Command Summary

```bash
# 1. Update version in all files
# 2. Commit changes
git add -A
git commit -m "chore: Bump version to 1.1.0"

# 3. Tag and push
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin master --tags

# 4. Build
python setup.py sdist

# 5. Create GitHub release and upload dist/*.tar.gz
```

## Contact

For package maintenance questions:
- Repository: https://github.com/Butadiene/geopack-vectorize
- Issues: https://github.com/Butadiene/geopack-vectorize/issues