from setuptools import setup, find_packages
import os
import re

# Read version from package __init__.py
with open(os.path.join('eda_automator', '__init__.py'), 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    version = version_match.group(1) if version_match else '0.1.0'

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="eda_automator",
    version=version,
    author="Daniel Moreno",
    author_email="danielmoreno3291@gmail.com",
    description="Automated exploratory data analysis with visualization and reporting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moreno32/eda-automator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "pyyaml>=6.0",
        "jinja2>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "interactive": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eda-automator=eda_automator.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 