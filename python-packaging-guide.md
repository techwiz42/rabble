# Python Package Distribution Guide

## 1. Project Structure

Create a standard Python project structure:

```
my_package/
│
├── my_package/
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
│
├── tests/
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_module2.py
│
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.py
└── setup.cfg
```

## 2. Essential Files

### `pyproject.toml` (Modern Packaging)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
version = "0.1.0"
description = "A fantastic Python package"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {file = "LICENSE"}
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
keywords = ["awesome", "package", "example"]
requires-python = ">=3.8"
dependencies = [
    "requests>=2.25.1",
    "numpy>=1.20.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov",
]

[project.urls]
Homepage = "https://github.com/yourusername/my_package"
Repository = "https://github.com/yourusername/my_package"

[tool.setuptools]
packages = ["my_package"]
```

### `README.md`
```markdown
# My Package

## Installation

```bash
pip install my_package
```

## Usage

```python
import my_package

# Example usage
result = my_package.some_function()
```

## Development

To set up development environment:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
```

### `LICENSE` (MIT License Example)
```
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full MIT License Text]
```

## 3. Package Code

### `my_package/__init__.py`
```python
"""
My Package - A fantastic Python package.
"""

__version__ = "0.1.0"

from .module1 import some_function
from .module2 import another_function

__all__ = ['some_function', 'another_function']
```

### `my_package/module1.py`
```python
def some_function():
    """
    An example function in the package.
    
    Returns:
        str: A greeting message
    """
    return "Hello from my_package!"
```

### `my_package/module2.py`
```python
def another_function(x):
    """
    Another example function.
    
    Args:
        x (int): Input number
    
    Returns:
        int: Squared input
    """
    return x ** 2
```

## 4. Testing

### `tests/test_module1.py`
```python
import my_package

def test_some_function():
    result = my_package.some_function()
    assert result == "Hello from my_package!"

def test_another_function():
    result = my_package.another_function(5)
    assert result == 25
```

## 5. Building and Publishing

### Prerequisites
```bash
# Install build tools
pip install build twine

# Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel twine
```

### Build the Package
```bash
# Build distribution files
python -m build

# This creates two files in the `dist/` directory:
# - my_package-0.1.0-py3-none-any.whl (wheel)
# - my_package-0.1.0.tar.gz (source distribution)
```

### Test Publishing (TestPyPI)
```bash
# Upload to TestPyPI (recommended before real PyPI)
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ my_package
```

### Publish to PyPI
```bash
# Upload to real PyPI
python -m twine upload dist/*
```

## 6. Version Management

### Semantic Versioning
- `0.1.0`: Initial release
- `0.1.1`: Patch (bug fixes)
- `0.2.0`: Minor version (backwards-compatible features)
- `1.0.0`: Major version (breaking changes)

### Updating Version
1. Update `__version__` in `__init__.py`
2. Update version in `pyproject.toml`
3. Commit changes
4. Create git tag: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`

## 7. Continuous Integration

Consider adding a `.github/workflows/python-package.yml` for GitHub Actions:

```yaml
name: Python Package

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install .[dev]
    - name: Test with pytest
      run: |
        pytest
```

## Common Pitfalls to Avoid
1. Don't include sensitive information in package
2. Keep dependencies minimal
3. Write clear documentation
4. Include type hints
5. Add meaningful tests
6. Keep package focused on a single responsibility

## Resources
- [Python Packaging User Guide](https://packaging.python.org/)
- [setuptools Documentation](https://setuptools.pypa.io/)
- [Twine Documentation](https://twine.readthedocs.io/)

## Final Checklist
- [ ] Create project structure
- [ ] Write package code
- [ ] Add tests
- [ ] Write README
- [ ] Choose and add LICENSE
- [ ] Create `pyproject.toml`
- [ ] Build distribution
- [ ] Test package
- [ ] Publish to PyPI
```
