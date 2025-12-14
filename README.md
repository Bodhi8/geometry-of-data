# Semantic Geometry Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python toolkit for understanding and analyzing the geometry of high-dimensional semantic embeddings. Explore how meaning is encoded as geometry in modern AI systems.

## 🎯 Overview

Modern language models encode words, sentences, and documents as points in high-dimensional vector spaces. This toolkit provides practical methods to:

- **Analyze** the geometric structure of embedding spaces
- **Visualize** high-dimensional semantic relationships
- **Discover** natural clusters and semantic directions
- **Measure** quality metrics like isotropy and intrinsic dimensionality
- **Probe** for specific semantic properties encoded in geometry

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic-geometry-toolkit.git
cd semantic-geometry-toolkit

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```python
from semantic_geometry import SemanticGeometryAnalyzer

# Initialize with your preferred embedding model
analyzer = SemanticGeometryAnalyzer(model_name='all-MiniLM-L6-v2')

# Encode your texts
texts = [
    "The cat sat on the mat",
    "Dogs are loyal companions", 
    "Machine learning transforms data",
    "Neural networks learn patterns",
    # ... more texts
]
analyzer.encode(texts)

# Get basic geometric statistics
stats = analyzer.basic_stats()
print(f"Mean pairwise similarity: {stats['mean_similarity']:.3f}")

# Discover clusters
clusters = analyzer.find_clusters(min_cluster_size=10)
print(f"Found {clusters['n_clusters']} natural clusters")

# Visualize the space
analyzer.visualize(color_by='cluster')
```

## 📦 Features

### Core Analysis

| Feature | Description |
|---------|-------------|
| `basic_stats()` | Compute norms, pairwise similarities, and distribution statistics |
| `find_clusters()` | Discover natural clusters using HDBSCAN |
| `analyze_isotropy()` | Measure how uniformly distributed embeddings are |
| `estimate_intrinsic_dim()` | Estimate the true dimensionality of your data manifold |

### Dimensionality Reduction

| Method | Best For |
|--------|----------|
| PCA | Understanding variance structure, fast projection |
| UMAP | Visualization preserving local + global structure |
| t-SNE | Visualization emphasizing local neighborhoods |

### Semantic Probing

| Feature | Description |
|---------|-------------|
| `find_semantic_direction()` | Find geometric directions corresponding to concepts |
| `project_onto_direction()` | Measure where points fall along a semantic axis |
| `measure_projection_quality()` | Assess information loss in dimensionality reduction |

### AI-Enhanced Analysis

| Feature | Description |
|---------|-------------|
| `describe_cluster()` | Use LLMs to semantically label discovered clusters |
| `interpret_direction()` | Use LLMs to explain what a geometric direction encodes |

## 📂 Project Structure

```
semantic-geometry-toolkit/
├── src/
│   └── semantic_geometry/
│       ├── __init__.py
│       ├── analyzer.py        # Main SemanticGeometryAnalyzer class
│       ├── distances.py       # Distance metrics and computations
│       ├── reduction.py       # Dimensionality reduction methods
│       ├── clustering.py      # Clustering analysis
│       ├── probing.py         # Semantic direction probing
│       ├── topology.py        # Topological data analysis
│       └── visualization.py   # Plotting utilities
├── examples/
│   ├── quickstart.py          # Basic usage example
│   ├── cluster_analysis.py    # In-depth clustering workflow
│   ├── semantic_directions.py # Finding and interpreting directions
│   └── compare_models.py      # Compare different embedding models
├── tests/
│   ├── test_analyzer.py
│   ├── test_distances.py
│   └── test_clustering.py
├── docs/
│   └── article.md             # Comprehensive guide article
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🔬 Example: Finding Semantic Directions

```python
from semantic_geometry import SemanticGeometryAnalyzer
from semantic_geometry.probing import find_semantic_direction, project_onto_direction

analyzer = SemanticGeometryAnalyzer()

# Encode examples with known properties
positive_texts = ["I love this!", "Amazing product", "Best ever"]
negative_texts = ["Terrible experience", "Waste of money", "Disappointing"]

all_texts = positive_texts + negative_texts
labels = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts)

analyzer.encode(all_texts)

# Find the sentiment direction
sentiment_direction = find_semantic_direction(
    analyzer.embeddings, 
    labels, 
    positive_class='positive',
    negative_class='negative'
)

# Project new texts onto this direction
new_texts = ["Not bad", "Could be better", "Exceeded expectations"]
new_embeddings = analyzer.encoder.encode(new_texts)
sentiment_scores = project_onto_direction(new_embeddings, sentiment_direction)

for text, score in zip(new_texts, sentiment_scores):
    print(f"{score:+.3f}: {text}")
```

## 📊 Example: Analyzing Cluster Structure

```python
from semantic_geometry import SemanticGeometryAnalyzer

analyzer = SemanticGeometryAnalyzer(model_name='all-mpnet-base-v2')
analyzer.encode(your_documents)

# Discover clusters
cluster_results = analyzer.find_clusters(min_cluster_size=20)

print(f"Found {cluster_results['n_clusters']} clusters")
print(f"Noise points: {cluster_results['n_noise']}")

# Examine each cluster
for label, info in cluster_results['clusters'].items():
    print(f"\nCluster {label} ({info['size']} documents):")
    for sample in info['sample_texts'][:3]:
        print(f"  - {sample[:80]}...")

# Visualize with cluster coloring
fig = analyzer.visualize(color_by='cluster')
fig.savefig('cluster_visualization.png', dpi=150)
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=semantic_geometry --cov-report=html
```

## 📈 Supported Embedding Models

The toolkit works with any embedding model that produces fixed-size vectors:

| Source | Models |
|--------|--------|
| Sentence Transformers | `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1` |
| OpenAI | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| Cohere | `embed-english-v3.0`, `embed-multilingual-v3.0` |
| HuggingFace | Any model via `transformers` library |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Further Reading

- [Understanding the Geometry of High-Dimensional Semantic Data](docs/article.md) - Comprehensive guide article
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Understanding HDBSCAN](https://hdbscan.readthedocs.io/)

## 🙏 Acknowledgments

- The Sentence Transformers team for excellent embedding models
- The UMAP authors for powerful dimensionality reduction
- The scientific community working on representation learning

---

**Note**: This toolkit is for educational and research purposes. Always verify results and consider the limitations of geometric analysis in high dimensions.
