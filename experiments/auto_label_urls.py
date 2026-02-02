"""Auto-label URLs based on keywords (for research/testing)"""

import csv
from pathlib import Path
import re


def auto_label_url(url: str) -> tuple:
    """
    Auto-label URL based on keyword presence
    
    Returns: (label, topic, confidence)
    """
    url_lower = url.lower()
    
    # Define keywords for each topic
    ml_keywords = [
        'machine_learning', 'artificial_intelligence', 'deep_learning',
        'neural_network', 'supervised_learning', 'reinforcement_learning',
        'algorithm', 'classification', 'regression', 'clustering',
        'data_science', 'model', 'training', 'inference'
    ]
    
    climate_keywords = [
        'climate', 'warming', 'greenhouse', 'carbon', 'emission',
        'sustainability', 'renewable', 'environment', 'paris_agreement',
        'fossil_fuel', 'clean_energy', 'ipcc'
    ]
    
    blockchain_keywords = [
        'blockchain', 'bitcoin', 'ethereum', 'cryptocurrency',
        'smart_contract', 'defi', 'proof_of_work', 'proof_of_stake',
        'distributed_ledger', 'consensus', 'crypto', 'web3'
    ]
    
    # Count keyword matches
    ml_count = sum(1 for kw in ml_keywords if kw in url_lower)
    climate_count = sum(1 for kw in climate_keywords if kw in url_lower)
    blockchain_count = sum(1 for kw in blockchain_keywords if kw in url_lower)
    
    # Determine label
    total_matches = ml_count + climate_count + blockchain_count
    
    if total_matches == 0:
        # Check if it's a generic navigation page
        if any(nav in url_lower for nav in ['main_page', 'contents', 'help:', 'special:', 'portal:', 'category:']):
            return 0, '', 'high'  # Irrelevant navigation page
        else:
            return 0, '', 'low'  # Unknown, default to irrelevant
    
    # Determine topic
    max_count = max(ml_count, climate_count, blockchain_count)
    if ml_count == max_count:
        topic = 'machine_learning'
    elif climate_count == max_count:
        topic = 'climate_science'
    else:
        topic = 'blockchain'
    
    # Confidence based on number of matches
    if max_count >= 3:
        confidence = 'high'
    elif max_count >= 2:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return 1, topic, confidence


def main():
    print("=" * 60)
    print("Auto-Label URLs (Research Helper)")
    print("=" * 60)
    
    # Load template
    template_file = Path(__file__).parent.parent / 'data' / 'target_domains' / 'urls_to_label.csv'
    
    if not template_file.exists():
        print(f"Error: Template not found at {template_file}")
        print("Run experiments/create_labeled_data.py first")
        return
    
    # Read and auto-label
    labeled_rows = []
    stats = {'relevant': 0, 'irrelevant': 0, 'high': 0, 'medium': 0, 'low': 0}
    
    with open(template_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row['url']
            label, topic, confidence = auto_label_url(url)
            
            row['label'] = label
            row['topic'] = topic if topic else row['topic']
            row['confidence'] = confidence
            row['notes'] = 'Auto-labeled'
            
            labeled_rows.append(row)
            
            # Update stats
            if label == 1:
                stats['relevant'] += 1
            else:
                stats['irrelevant'] += 1
            stats[confidence] += 1
    
    # Save labeled data
    output_file = template_file.parent / 'labeled_urls.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'label', 'topic', 'confidence', 'notes'])
        writer.writeheader()
        writer.writerows(labeled_rows)
    
    print(f"\nAuto-labeled {len(labeled_rows)} URLs")
    print(f"Saved to: {output_file}")
    print("\nStatistics:")
    print(f"  Relevant: {stats['relevant']} ({stats['relevant']/len(labeled_rows)*100:.1f}%)")
    print(f"  Irrelevant: {stats['irrelevant']} ({stats['irrelevant']/len(labeled_rows)*100:.1f}%)")
    print(f"\nConfidence:")
    print(f"  High: {stats['high']}")
    print(f"  Medium: {stats['medium']}")
    print(f"  Low: {stats['low']}")
    
    print("\n" + "=" * 60)
    print("✓ Auto-labeling complete!")
    print("=" * 60)
    print("\nNote: These are automated labels for research/testing.")
    print("For production, manually review and correct labels.")
    print("\nNext: Run with --split flag to create train/val/test splits")


if __name__ == '__main__':
    main()
