"""Phase 2.2: Create labeled training data from bootstrap graph"""

import json
import csv
import random
from pathlib import Path
from typing import List, Dict


def extract_urls_from_graph(graph_file: Path) -> List[str]:
    """Extract all URLs from bootstrap graph"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from graph.web_graph import WebGraph
    
    graph = WebGraph.load(graph_file)
    urls = list(graph.graph.nodes())
    return urls


def create_labeling_template(urls: List[str], output_file: Path, sample_size: int = 500):
    """
    Create CSV template for manual labeling
    
    Args:
        urls: List of URLs to label
        output_file: Output CSV file path
        sample_size: Number of URLs to sample for labeling
    """
    # Sample URLs if we have more than needed
    if len(urls) > sample_size:
        urls = random.sample(urls, sample_size)
    
    # Load keywords from seed files to help with labeling
    seeds_dir = Path(__file__).parent.parent / 'data' / 'seeds'
    keywords = {}
    
    for seed_file in seeds_dir.glob('*_seeds.json'):
        with open(seed_file, 'r') as f:
            data = json.load(f)
            keywords[data['topic']] = data['keywords']
    
    # Create CSV with columns
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'label', 'topic', 'confidence', 'notes'])
        
        for url in urls:
            # Auto-suggest topic based on keywords in URL
            suggested_topic = ''
            for topic, kws in keywords.items():
                if any(kw.replace(' ', '_') in url.lower() for kw in kws):
                    suggested_topic = topic
                    break
            
            writer.writerow([url, '', suggested_topic, '', ''])
    
    print(f"Created labeling template with {len(urls)} URLs")
    print(f"Saved to: {output_file}")
    print("\nInstructions:")
    print("  1. Open the CSV file in a spreadsheet editor")
    print("  2. For each URL, fill in:")
    print("     - label: 1 (relevant) or 0 (irrelevant)")
    print("     - confidence: high/medium/low")
    print("     - notes: optional comments")
    print("  3. Save the file when done")
    print("\nTip: Visit the URL to determine relevance")


def split_labeled_data(labeled_file: Path):
    """
    Split labeled data into train/val/test sets (70/15/15)
    """
    # Read labeled data
    with open(labeled_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader if row['label']]  # Only labeled rows
    
    # Shuffle and split
    random.shuffle(data)
    n = len(data)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Save splits
    output_dir = labeled_file.parent
    
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_file = output_dir / f'{split_name}_labeled.csv'
        with open(split_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(split_data)
        print(f"Saved {len(split_data)} examples to {split_file}")


def main():
    print("=" * 60)
    print("Phase 2.2: Create Labeled Training Data")
    print("=" * 60)
    
    graph_file = Path(__file__).parent.parent / 'data' / 'graphs' / 'bootstrap_graph.pkl'
    
    if not graph_file.exists():
        print(f"Error: Bootstrap graph not found at {graph_file}")
        print("Run experiments/bootstrap_graph.py first")
        return
    
    # Extract URLs from graph
    print("\nExtracting URLs from bootstrap graph...")
    urls = extract_urls_from_graph(graph_file)
    print(f"Found {len(urls)} total URLs")
    
    # Create labeling template
    output_dir = Path(__file__).parent.parent / 'data' / 'target_domains'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    template_file = output_dir / 'urls_to_label.csv'
    create_labeling_template(urls, template_file, sample_size=500)
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Open urls_to_label.csv in Excel/Google Sheets")
    print("2. Label at least 500 URLs as relevant (1) or irrelevant (0)")
    print("3. Save as labeled_urls.csv in the same directory")
    print("4. Run this script again with --split flag to create train/val/test splits")
    print("\nExample labeling criteria:")
    print("  ML topic: Contains machine learning concepts, algorithms, models")
    print("  Climate topic: Contains climate science, environmental policy")
    print("  Blockchain topic: Contains crypto, distributed ledger tech")


if __name__ == '__main__':
    import sys
    
    if '--split' in sys.argv:
        # Split already labeled data
        labeled_file = Path(__file__).parent.parent / 'data' / 'target_domains' / 'labeled_urls.csv'
        if labeled_file.exists():
            split_labeled_data(labeled_file)
        else:
            print(f"Error: {labeled_file} not found")
    else:
        # Create labeling template
        main()
