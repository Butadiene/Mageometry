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
pip install -e .
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

# Install build tools
pip install --upgrade build twine

# Build source distribution and wheel
python -m build

# Check the built packages
ls -la dist/
# Should see:
# - geopack_vectorized-1.1.0.tar.gz
# - geopack_vectorized-1.1.0-py3-none-any.whl
```

### 3. Test Distribution
```bash
# Test installation from built wheel
pip install dist/geopack_vectorized-1.1.0-py3-none-any.whl

# Run basic import test
python -c "import geopack; print(geopack.__version__)"

# Uninstall test version
pip uninstall geopack-vectorized
```

### 4. Upload to PyPI

#### Test PyPI First (Recommended)
```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ geopack-vectorized

# Verify it works
python -c "import geopack; print(geopack.__version__)"
```

#### Production PyPI
```bash
# Upload to PyPI (requires PyPI account and API token)
twine upload dist/*

# Will prompt for:
# Username: __token__
# Password: <your-pypi-api-token>
```

### 5. GitHub Release
1. Go to https://github.com/Butadiene/geopack-vectorize/releases
2. Click "Create a new release"
3. Choose the tag you created (v1.1.0)
4. Title: "geopack-vectorized v1.1.0"
5. Copy content from `RELEASE_NOTES_v1.1.0.md`
6. Attach the built files from `dist/`
7. Click "Publish release"

### 6. Post-Release Verification
```bash
# Wait 5-10 minutes for PyPI to update, then test
pip install --upgrade geopack-vectorized
python -c "import geopack; print(geopack.__version__)"
```

## PyPI Setup (First Time Only)

### 1. Create PyPI Account
- Register at https://pypi.org/account/register/
- Verify email address
- Enable 2FA (highly recommended)

### 2. Create API Token
- Go to https://pypi.org/manage/account/
- Scroll to "API tokens"
- Click "Add API token"
- Name: "geopack-vectorized-upload"
- Scope: "Project: geopack-vectorized" (or "Entire account" for first upload)
- Copy the token (starts with `pypi-`)

### 3. Configure Token
```bash
# Option 1: Use keyring (recommended)
pip install keyring
keyring set https://upload.pypi.org/legacy/ __token__
# Paste your token when prompted

# Option 2: Create .pypirc file (less secure)
# Create ~/.pypirc with:
[pypi]
username = __token__
password = pypi-xxxxx-your-token-here
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
# Ensure latest tools
pip install --upgrade pip setuptools wheel build

# Clear caches
rm -rf build/ dist/ *.egg-info __pycache__
find . -type d -name __pycache__ -exec rm -rf {} +
```

### Upload Errors
- **401 Unauthorized**: Check API token is correct
- **400 Bad Request**: Package name might already exist
- **File already exists**: Version already uploaded (bump version number)

### Installation Issues
```bash
# Force reinstall
pip install --force-reinstall --no-cache-dir geopack-vectorized

# Check installed version
pip show geopack-vectorized
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
python -m build

# 5. Upload
twine upload dist/*

# 6. Create GitHub release
```

## Contact

For package maintenance questions:
- Repository: https://github.com/Butadiene/geopack-vectorize
- Issues: https://github.com/Butadiene/geopack-vectorize/issues