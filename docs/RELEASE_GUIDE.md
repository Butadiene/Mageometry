# Release Guide for geopack-vectorize

This guide outlines the steps to release a new version of the geopack-vectorize package.

## Pre-Release Checklist

### 1. Code Preparation
- [ ] Ensure all tests pass: `python -m unittest discover tests/`
- [ ] Check all notebooks execute without errors
- [ ] Remove any temporary or debug files
- [ ] Ensure no sensitive information in code

### 2. Version Updates
Update version numbers in these files:
- [ ] `setup.py` - version field
- [ ] `pyproject.toml` - version field
- [ ] `geopack/__init__.py` - __version__ variable

### 3. Documentation Updates
- [ ] Update `docs/releases/RELEASE_NOTES_vX.X.X.md` with changes
- [ ] Update `CHANGELOG.md` with changes
- [ ] Update `README.md` with new features
- [ ] Update `CLAUDE.md` if project structure changed
- [ ] Check all import examples are current
- [ ] Update performance/accuracy numbers if improved

### 4. Final Testing
```bash
# Clean install test
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate
pip install -e .
python -m unittest discover tests/
deactivate
rm -rf test_env
```

## Release Process

### 1. Git Tasks
```bash
# Ensure you're on master branch
git checkout master

# Ensure branch is up to date
git pull origin master

# Create and push version tag
git tag -a v2.0.0 -m "Release version 2.0.0"
git push origin v2.0.0

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
# - geopack_vectorize-2.0.0.tar.gz
```

### 3. Test Distribution
```bash
# Test installation from built package
tar -xzf dist/geopack_vectorize-2.0.0.tar.gz
cd geopack_vectorize-2.0.0
pip install .

# Run basic import test
python -c "import geopack; print(geopack.__version__)"
```

### 4. GitHub Release
1. Go to https://github.com/Butadiene/geopack-vectorize/releases
2. Click "Create a new release"
3. Choose the tag you created (v2.0.0)
4. Title: "geopack-vectorize v2.0.0"
5. Copy content from `docs/releases/RELEASE_NOTES_v2.0.0.md`
6. Attach the built files from `dist/`
7. Click "Publish release"

### 5. Post-Release Verification
```bash
# Install from PyPI and test
pip install geopack-vectorize
python -c "import geopack; print(geopack.__version__)"
```

## Version Numbering

Follow semantic versioning (https://semver.org/):
- **Major** (X.0.0): Breaking API changes
- **Minor** (2.X.0): New features, backwards compatible
- **Patch** (2.0.X): Bug fixes, backwards compatible

Examples:
- 1.1.4 → 2.0.0: Package renamed, module structure reorganized (breaking change)
- 2.0.0 → 2.1.0: Added new vectorized functions (new feature)
- 2.0.0 → 2.0.1: Fixed bug in field calculation

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
pip install --user .

# Or install dependencies first:
python -m pip install numpy scipy
pip install .
```

## Maintenance Tasks

### Regular Updates
- Keep dependencies updated in setup.py and pyproject.toml
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
# 1. Update version in setup.py, pyproject.toml, geopack/__init__.py
# 2. Update CHANGELOG.md and docs/releases/RELEASE_NOTES_vX.X.X.md
# 3. Commit changes
git add -A
git commit -m "chore: Bump version to 2.0.0"

# 4. Tag and push
git tag -a v2.0.0 -m "Release version 2.0.0"
git push origin master --tags

# 5. Build
python setup.py sdist

# 6. Create GitHub release and upload dist/*.tar.gz
```

## Contact

For package maintenance questions:
- Repository: https://github.com/Butadiene/geopack-vectorize
- Issues: https://github.com/Butadiene/geopack-vectorize/issues
