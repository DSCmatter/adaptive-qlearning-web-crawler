"""Phase 2.1: Bootstrap initial graph from seed URLs"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graph.graph_builder import bootstrap_initial_graph
from graph.web_graph import WebGraph


def load_seeds(topic: str) -> list:
    """Load seed URLs for a topic"""
    seeds_file = Path(__file__).parent.parent / 'data' / 'seeds' / f'{topic}_seeds.json'
    with open(seeds_file, 'r') as f:
        data = json.load(f)
    return data['seeds']


def main():
    print("=" * 60)
    print("Phase 2.1: Bootstrap Initial Web Graph")
    print("=" * 60)
    
    # Load seeds from all topics
    topics = ['ml', 'climate', 'blockchain']
    all_seeds = []
    
    for topic in topics:
        try:
            seeds = load_seeds(topic)
            print(f"\n{topic.upper()}: Loaded {len(seeds)} seed URLs")
            all_seeds.extend(seeds)
        except FileNotFoundError:
            print(f"Warning: {topic}_seeds.json not found, skipping")
    
    print(f"\nTotal seed URLs: {len(all_seeds)}")
    print(f"Target pages to crawl: 500")
    print(f"Estimated time: 10-15 minutes")
    print(f"\nStarting bootstrap crawl...")
    print("-" * 60)
    
    # Bootstrap the graph
    graph = bootstrap_initial_graph(all_seeds, max_pages=500)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Bootstrap Complete!")
    print("=" * 60)
    print(f"Total pages crawled: {graph.num_nodes()}")
    print(f"Total links discovered: {graph.num_edges()}")
    print(f"Average out-degree: {graph.num_edges() / max(graph.num_nodes(), 1):.2f}")
    
    # Save the graph
    output_dir = Path(__file__).parent.parent / 'data' / 'graphs'
    output_dir.mkdir(exist_ok=True)
    graph_file = output_dir / 'bootstrap_graph.pkl'
    
    graph.save(graph_file)
    print(f"\nGraph saved to: {graph_file}")
    
    print("\n✓ Phase 2.1 complete!")
    print("\nNext steps:")
    print("  1. Label URLs as relevant/irrelevant (Phase 2.2)")
    print("  2. Extract features for all nodes (Phase 2.3)")
    print("  3. Pre-train GNN on bootstrap graph (Phase 3.1)")


if __name__ == '__main__':
    main()
