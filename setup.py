from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semantic-geometry-toolkit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for analyzing the geometry of high-dimensional semantic embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semantic-geometry-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/semantic-geometry-toolkit/issues",
        "Documentation": "https://github.com/yourusername/semantic-geometry-toolkit#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "umap-learn>=0.5.3",
        "hdbscan>=0.8.28",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "topology": [
            "ripser>=0.6.0",
            "persim>=0.3.0",
        ],
        "openai": [
            "openai>=1.0.0",
        ],
        "all": [
            "ripser>=0.6.0",
            "persim>=0.3.0",
            "openai>=1.0.0",
            "cohere>=4.0.0",
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
    },
)
