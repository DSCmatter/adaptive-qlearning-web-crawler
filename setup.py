from setuptools import setup, find_packages

setup(
    name="adaptive-qlearning-crawler",
    version="0.1.0",
    description="Adaptive Q-Learning Web Crawler with Contextual Bandits and GNNs",
    author="DSCmatter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torch-geometric",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "requests",
        "beautifulsoup4",
        "lxml",
        "networkx",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "viz": ["matplotlib", "seaborn"],
    },
)
