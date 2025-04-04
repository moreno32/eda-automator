#!/usr/bin/env python3
"""
Verify that the version numbers in different files are consistent.

This script checks that the version number in eda_automator/__init__.py
matches the version in pyproject.toml to ensure consistency.
"""

import os
import re
import sys
import tomli

def get_version_from_init():
    """Extract version from __init__.py file."""
    init_path = os.path.join("eda_automator", "__init__.py")
    
    if not os.path.exists(init_path):
        print(f"Error: Could not find {init_path}")
        return None
    
    with open(init_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Look for __version__ = "x.y.z"
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print(f"Error: Could not find __version__ in {init_path}")
        return None
    
    return match.group(1)

def get_version_from_pyproject():
    """Extract version from pyproject.toml file."""
    pyproject_path = "pyproject.toml"
    
    if not os.path.exists(pyproject_path):
        print(f"Error: Could not find {pyproject_path}")
        return None
    
    try:
        with open(pyproject_path, "rb") as file:
            pyproject_data = tomli.load(file)
        
        # Try to find version in different possible locations
        if "project" in pyproject_data and "version" in pyproject_data["project"]:
            return pyproject_data["project"]["version"]
        
        # Old format using [tool.poetry]
        if "tool" in pyproject_data and "poetry" in pyproject_data["tool"] and "version" in pyproject_data["tool"]["poetry"]:
            return pyproject_data["tool"]["poetry"]["version"]
        
        print("Error: Could not find version in pyproject.toml")
        return None
    
    except Exception as e:
        print(f"Error parsing pyproject.toml: {e}")
        return None

def is_valid_semver(version):
    """Check if a version string follows semantic versioning."""
    if not version:
        return False
    
    # Simple regex for MAJOR.MINOR.PATCH format
    semver_pattern = r'^[0-9]+\.[0-9]+\.[0-9]+$'
    return re.match(semver_pattern, version) is not None

def main():
    """Main function to verify version consistency."""
    init_version = get_version_from_init()
    pyproject_version = get_version_from_pyproject()
    
    if init_version is None or pyproject_version is None:
        sys.exit(1)
    
    # Verify that both versions follow semantic versioning
    if not is_valid_semver(init_version):
        print(f"Error: Version in __init__.py ({init_version}) does not follow semantic versioning (MAJOR.MINOR.PATCH)")
        sys.exit(1)
    
    if not is_valid_semver(pyproject_version):
        print(f"Error: Version in pyproject.toml ({pyproject_version}) does not follow semantic versioning (MAJOR.MINOR.PATCH)")
        sys.exit(1)
    
    # Verify that versions match
    if init_version != pyproject_version:
        print(f"Error: Version mismatch between __init__.py ({init_version}) and pyproject.toml ({pyproject_version})")
        sys.exit(1)
    
    print(f"Success: Version {init_version} is consistent across files and follows semantic versioning")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 