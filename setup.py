""
Setup script for the ncrsh package.
"""
from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt."""
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ncrsh",
    version="0.1.0",
    author="Dineth Nethsara",
    author_email="dineth@example.com",
    description="The Ultimate Deep Learning Stack — Transformers, Tokenizers, and Torch in One",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dinethnethsara/ncrsh",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.900",
            "pytest-cov>=3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="deep-learning transformers tokenizers pytorch machine-learning",
)
