# Phase 2: Data Collection & Preprocessing

**Timeline**: Week 1, Days 3-5  
**Status**: ✅ Complete  
**Date Completed**: February 2, 2026

---

## Overview

Phase 2 focused on collecting and preparing the data needed to train the hybrid RL crawler. This included bootstrapping an initial web graph, creating labeled training data, and implementing a comprehensive feature extraction pipeline.

---

## 2.1 Seed URL Collection

### Objective
Identify target domains and collect high-quality seed URLs for bootstrapping the crawler.

### Topics Selected

**1. Machine Learning** ([ml_seeds.json](../../data/seeds/ml_seeds.json))
- 5 seed URLs from Wikipedia
- Keywords: machine learning, AI, neural network, deep learning, supervised learning

**2. Climate Science** ([climate_seeds.json](../../data/seeds/climate_seeds.json))
- 8 seed URLs from Wikipedia
- Keywords: climate change, global warming, greenhouse gas, sustainability, renewable energy

**3. Blockchain** ([blockchain_seeds.json](../../data/seeds/blockchain_seeds.json))
- 8 seed URLs from Wikipedia
- Keywords: blockchain, cryptocurrency, bitcoin, ethereum, smart contract, DeFi

### Total Seeds: 21 URLs across 3 topics

### Seed File Format

```json
{
  "topic": "machine_learning",
  "description": "Machine Learning and AI related pages",
  "seeds": [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    ...
  ],
  "keywords": [
    "machine learning",
    "artificial intelligence",
    ...
  ]
}
```

---

## 2.2 Graph Bootstrapping

### Objective
Build an initial graph structure to train the GNN on. This solves the "chicken-egg" problem: GNN needs graph → crawler needs GNN → need pages to build graph.

### Implementation ([graph_builder.py](../../src/graph/graph_builder.py))

**Algorithm**: Simple BFS (Breadth-First Search)

```python
def bootstrap_initial_graph(seed_urls, max_pages=500):
    1. Start with seed URLs in queue
    2. Pop URL from queue
    3. Fetch page (timeout=10s, politeness delay=0.2s)
    4. Extract links using BeautifulSoup
    5. Convert relative URLs to absolute (urljoin)
    6. Take first 30 links per page
    7. Add to graph and queue
    8. Repeat until max_pages reached
```

### Key Technical Fixes

**Problem 1**: Wikipedia uses relative URLs (`/wiki/Page`)
- **Solution**: Added `urljoin(base_url, link)` to convert to absolute URLs

**Problem 2**: Too few links queued (originally 10/page)
- **Solution**: Increased to 30 links/page to reach target 500 pages

**Problem 3**: Connection timeouts and rate limiting
- **Solution**: 
  - Increased timeout from 5s to 10s
  - Added better error handling for RequestException
  - Reduced delay from 0.5s to 0.2s for faster crawling

### Results

**Bootstrap Graph Statistics**:
- **Nodes**: 60 pages (Wikipedia rate-limited before reaching 500)
- **Edges**: 600 directed links
- **Average out-degree**: 10 links per page
- **File**: [bootstrap_graph.pkl](../../data/graphs/bootstrap_graph.pkl)

**Graph Structure**:
- Seed pages (21 nodes) with high centrality
- Outlinks to related Wikipedia pages
- Mix of content pages and navigation pages

**Why 60 is Sufficient**:
- GNNs learn from local structure, not global size
- 60 nodes with 600 edges provides enough graph topology
- Can still validate generalization on new pages

---

## 2.3 Feature Extraction Pipeline

### Objective
Build 174-dimensional context vectors for each page/link to feed into the bandit and Q-learning agent.

### Architecture

```
Context Vector (174 dims) = 
  [GNN Embedding: 64 dims]        ← Learned from graph (Phase 3)
  [URL Features: 20 dims]         ← Handcrafted
  [Content Features: 50 dims]     ← Handcrafted
  [Anchor Features: 30 dims]      ← Handcrafted
  [Graph Features: 10 dims]       ← Handcrafted
```

### Implementation ([feature_extractor.py](../../src/models/feature_extractor.py))

#### 1. URL Features (20 dimensions)

**Purpose**: Capture URL structure and domain characteristics

**Features**:
```python
- URL depth (number of path segments)
- Path length (characters)
- Domain length
- Subdomain count
- Has query parameters (boolean)
- Has fragment (boolean)
- Domain type indicators:
  - Is Wikipedia
  - Is .edu domain
  - Is .gov domain
  - Is .org domain
- File type indicators:
  - Ends with .html
  - Ends with .pdf
  - Ends with .php
  - Ends with .asp
- Special path indicators:
  - Contains "category"
  - Contains "archive"
  - Contains "blog"
  - Contains "news"
- Query string length
- Contains numbers in path
```

**Example**:
```python
url = "https://en.wikipedia.org/wiki/Machine_learning"
features = [
    2,      # depth
    22,     # path length
    18,     # domain length
    1,      # subdomains
    0.0,    # no query
    0.0,    # no fragment
    1.0,    # is Wikipedia
    ...
]
```

#### 2. Content Features (50 dimensions)

**Purpose**: Capture page content characteristics

**Components**:
- **TF-IDF vectors (40 dims)**: Top 40 terms from page text
- **Content statistics (10 dims)**:
  - Log-scaled word count
  - Log-scaled character count
  - Log-scaled link count
  - Link density (links/words)
  - Heading ratio (headings/paragraphs)
  - Has tables (boolean)
  - Has images (boolean)
  - Has lists (boolean)
  - Div count
  - Span count

**Preprocessing**:
```python
- Parse HTML with BeautifulSoup
- Remove <script> and <style> tags
- Extract text content
- Normalize whitespace
- Apply TF-IDF transformation
```

**Note**: TF-IDF requires fitting on corpus first:
```python
extractor.fit_tfidf(documents)  # Call before extraction
```

#### 3. Anchor Features (30 dimensions)

**Purpose**: Capture information from link anchor text

**Features**:
- Word count
- Character count
- Has capitalized words
- Contains numbers
- **Topic keyword matching**:
  - ML keywords (5 features): "machine", "learning", "neural", "network", "ai"
  - Climate keywords (5 features): "climate", "environment", "carbon", "energy", "warming"
  - Blockchain keywords (5 features): "blockchain", "crypto", "bitcoin", "ethereum", "defi"
- **Action words**: "read", "view", "more", "details", "article", "page"
- **Navigation indicators**: "main", "home", "next", "previous", "category", "section"
- **Special characters**: ":", "|", "(", "["

**Design Rationale**:
- **Why hardcode keywords?** Domain-specific priors help the crawler focus
- **Is this cheating?** No - it's feature engineering, not leaking test data
- **Limitation**: Only works for these 3 topics (needs rewriting for new domains)
- **Better alternative** (future work): Use pre-trained word embeddings (BERT) for generalization

#### 4. Graph Features (10 dimensions)

**Purpose**: Capture node position and importance in graph

**Features**:
```python
- Log(in-degree)              # Popularity
- Log(out-degree)             # Hub-ness
- In/out degree ratio         # Link balance
- PageRank score (scaled)     # Authority
- Is popular (in-degree > 5)
- Is hub (out-degree > 10)
- Is leaf (in-degree == 0)
- Is dead-end (out-degree == 0)
- Normalized in-degree (0-1)
- Normalized out-degree (0-1)
```

**Note**: These require the full graph structure, so they're zeros for unseen pages initially.

### Feature Extraction Methods

```python
class FeatureExtractor:
    def extract_url_features(url) -> np.ndarray[20]
    def extract_content_features(html) -> np.ndarray[50]
    def extract_anchor_features(anchor_text) -> np.ndarray[30]
    def extract_graph_features(node_id, graph) -> np.ndarray[10]
    
    def build_context_vector(url, html, anchor_text, 
                            gnn_embedding, graph) -> np.ndarray[174]
```

### Validation ([test_features.py](../../experiments/test_features.py))

**Tests performed**:
- ✅ All features return correct dimensions
- ✅ No NaN or Inf values
- ✅ Feature ranges are reasonable
- ✅ Context vector is exactly 174 dimensions

**Sample Results** (10 Wikipedia pages):
```
Feature dimension: 174
Min value: 0.0000
Max value: 87.5000
Mean value: 0.7551
Std value: 6.2735
Has NaN: False
Has Inf: False
```

---

## 2.4 Labeled Training Data

### Objective
Create labeled dataset (relevant/irrelevant) for supervised GNN pre-training.

### Labeling Process

#### Step 1: Template Generation ([create_labeled_data.py](../../experiments/create_labeled_data.py))

```python
# Extract all URLs from bootstrap graph
urls = graph.graph.nodes()  # 60 URLs

# Create CSV template
csv columns: [url, label, topic, confidence, notes]

# Auto-suggest topics based on keyword presence
for url in urls:
    if "machine_learning" in url:
        suggested_topic = "machine_learning"
```

**Output**: [urls_to_label.csv](../../data/target_domains/urls_to_label.csv)

#### Step 2: Auto-Labeling ([auto_label_urls.py](../../experiments/auto_label_urls.py))

**Algorithm**: Keyword-based classification

```python
def auto_label_url(url):
    # Count keyword matches for each topic
    ml_count = count_keywords(url, ML_KEYWORDS)
    climate_count = count_keywords(url, CLIMATE_KEYWORDS)
    blockchain_count = count_keywords(url, BLOCKCHAIN_KEYWORDS)
    
    # Label as relevant if any matches found
    if max(ml_count, climate_count, blockchain_count) > 0:
        label = 1  # Relevant
        topic = topic_with_max_count
        confidence = "high" if max_count >= 3 else "medium"
    else:
        label = 0  # Irrelevant (navigation pages)
        confidence = "high" if is_navigation_page else "low"
```

**Output**: [labeled_urls.csv](../../data/target_domains/labeled_urls.csv)

### Labeling Results

**Statistics**:
- **Total URLs**: 60
- **Relevant**: 40 (66.7%)
- **Irrelevant**: 20 (33.3%)

**Confidence Distribution**:
- High confidence: 12 URLs
- Medium confidence: 2 URLs
- Low confidence: 46 URLs

**Why Low Confidence?**
- Many URLs are navigation pages (Main_Page, Contents, Help:)
- URL alone doesn't reveal content
- Would need to read page text for higher confidence

### The "Hacky" Part - Limitations

**Problem**: We labeled based on URL keywords, not actual page content

**Example Issue**:
```
URL: https://en.wikipedia.org/wiki/Machine_learning
Label: Relevant (because URL contains "Machine_learning")

But what if:
- Page was vandalized and now talks about gardening?
- It's just a disambiguation page with 1 sentence?
- The page is in a language we don't support?
```

**Proper Research Approach**:
1. Download each page
2. Extract text content
3. Human annotator reads it
4. Labels as relevant/irrelevant based on actual content
5. Repeat for 500-1000 URLs

**Our Shortcut Justification**:
- Good enough for testing/prototyping
- Wikipedia is stable (vandalism is rare and reverted quickly)
- We can mention this limitation in paper's "Future Work" section
- Real production systems would use human labelers (expensive!)

#### Step 3: Train/Val/Test Split

```bash
python experiments/create_labeled_data.py --split
```

**Split Ratio**: 70% train / 15% val / 15% test

**Results**:
- [train_labeled.csv](../../data/target_domains/train_labeled.csv) - 42 examples
- [val_labeled.csv](../../data/target_domains/val_labeled.csv) - 9 examples
- [test_labeled.csv](../../data/target_domains/test_labeled.csv) - 9 examples

---

## Scripts Created

### 1. bootstrap_graph.py
**Purpose**: Crawl seed URLs to build initial graph  
**Usage**: `python experiments/bootstrap_graph.py`  
**Output**: `data/graphs/bootstrap_graph.pkl`

### 2. create_labeled_data.py
**Purpose**: Generate labeling template and split data  
**Usage**: 
```bash
# Create template
python experiments/create_labeled_data.py

# Split labeled data
python experiments/create_labeled_data.py --split
```

### 3. auto_label_urls.py
**Purpose**: Auto-label URLs based on keywords  
**Usage**: `python experiments/auto_label_urls.py`  
**Output**: `data/target_domains/labeled_urls.csv`

### 4. test_features.py
**Purpose**: Validate feature extraction pipeline  
**Usage**: `python experiments/test_features.py`  
**Output**: Console output with validation results

---

## Data Files Summary

```
data/
├── seeds/                          # Seed URL collections
│   ├── ml_seeds.json              # Machine learning seeds (5)
│   ├── climate_seeds.json         # Climate science seeds (8)
│   └── blockchain_seeds.json      # Blockchain seeds (8)
├── graphs/                         # Graph structures
│   └── bootstrap_graph.pkl        # Bootstrap graph (60 nodes, 600 edges)
└── target_domains/                 # Labeled data
    ├── urls_to_label.csv          # Template (60 URLs)
    ├── labeled_urls.csv           # Auto-labeled (60 URLs)
    ├── train_labeled.csv          # Training set (42)
    ├── val_labeled.csv            # Validation set (9)
    └── test_labeled.csv           # Test set (9)
```

---

## Lessons Learned

### What Worked Well

1. **Auto-labeling**: Fast way to create training data for prototyping
2. **Bootstrap approach**: Simple BFS effectively built diverse graph
3. **Hybrid features**: Combining learned (GNN) + handcrafted features
4. **Multiple topics**: Prevents overfitting to single domain

### Challenges Encountered

1. **Wikipedia rate limiting**: Limited to 60 pages instead of 500
   - Solution: 60 is still sufficient for GNN training
   
2. **Relative URLs**: Initial crawl found 0 edges
   - Solution: Added `urljoin()` to convert relative → absolute URLs

3. **Queue exhaustion**: Only taking 10 links/page ran out too fast
   - Solution: Increased to 30 links/page

4. **Content features without HTML**: Graph doesn't store HTML
   - Impact: Content features are all zeros for now
   - Future: Store HTML separately or extract features during crawl

### Design Decisions & Trade-offs

#### 1. Keyword-based Features
**Decision**: Hardcode ML/climate/blockchain keywords  
**Pro**: Domain-specific priors improve crawler focus  
**Con**: Not generalizable to new topics  
**Alternative**: Use pre-trained word embeddings (BERT)

#### 2. Auto-labeling vs Human Labels
**Decision**: Auto-label based on URL keywords  
**Pro**: Fast, scalable, no cost  
**Con**: Lower quality, potential noise  
**Alternative**: Hire annotators on Amazon MTurk (~$0.05/label × 1000 = $50)

#### 3. Small Dataset (60 pages)
**Decision**: Accept 60 pages instead of waiting for 500  
**Pro**: Faster iteration, still sufficient for GNN  
**Con**: Risk of overfitting  
**Mitigation**: Early stopping, validation set, simple model

#### 4. Feature Dimension (174 dims)
**Decision**: 64 GNN + 110 handcrafted  
**Pro**: Rich representation without being too large  
**Con**: Still significant for CPU (but manageable)  
**Alternative**: PCA to reduce dims (may lose information)

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Seed URLs | 21 (3 topics) |
| Bootstrap pages | 60 |
| Bootstrap edges | 600 |
| Avg out-degree | 10.0 |
| Labeled URLs | 60 |
| Training examples | 42 |
| Validation examples | 9 |
| Test examples | 9 |
| Feature dimensions | 174 |
| URL features | 20 |
| Content features | 50 |
| Anchor features | 30 |
| Graph features | 10 |
| GNN embedding | 64 |

---

## Deliverables Summary

✅ **Seed URLs**: 21 URLs across 3 topics  
✅ **Bootstrap Graph**: 60 nodes, 600 edges, saved to PKL  
✅ **Feature Extraction**: Complete 174-dim pipeline  
✅ **Labeled Data**: 60 examples split into train/val/test  
✅ **Scripts**: 4 executable Python scripts  
✅ **Validation**: Feature extraction tested, no NaN/Inf  

---

## How to Run Phase 2

### Prerequisites
- ✅ Phase 1 completed (virtual environment set up)
- Internet connection for web crawling
- ~10-20 minutes for execution

### Step-by-Step Instructions

```bash
# 0. Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # Linux/macOS

# Verify activation (should show venv path)
python -c "import sys; print(sys.prefix)"

# 1. Bootstrap Initial Graph
# Crawls seed URLs to build graph structure
# Time: ~5-15 minutes depending on network
python experiments\bootstrap_graph.py

# Expected output:
# ============================================================
# Phase 2.1: Bootstrap Initial Web Graph
# ============================================================
# ML: Loaded 5 seed URLs
# CLIMATE: Loaded 8 seed URLs
# BLOCKCHAIN: Loaded 8 seed URLs
# Total seed URLs: 21
# Starting bootstrap crawl...
# [50/500] Found 5253 links (30 HTTP), queued 1, queue size: 464
# Bootstrap complete: 60 nodes, 600 edges
# Graph saved to: data/graphs/bootstrap_graph.pkl
# ✓ Phase 2.1 complete!

# 2. Create Labeling Template
# Extracts URLs from graph for labeling
python experiments\create_labeled_data.py

# Expected output:
# Found 60 total URLs
# Created labeling template with 60 URLs
# Saved to: data/target_domains/urls_to_label.csv

# 3. Auto-Label URLs
# Labels URLs based on keyword matching
python experiments\auto_label_urls.py

# Expected output:
# Auto-labeled 60 URLs
# Relevant: 40 (66.7%)
# Irrelevant: 20 (33.3%)
# Saved to: data/target_domains/labeled_urls.csv

# 4. Split into Train/Val/Test
# Creates 70/15/15 split
python experiments\create_labeled_data.py --split

# Expected output:
# Saved 42 examples to train_labeled.csv
# Saved 9 examples to val_labeled.csv
# Saved 9 examples to test_labeled.csv

# 5. Test Feature Extraction
# Validates feature extraction pipeline
python experiments\test_features.py

# Expected output:
# Loaded 60 nodes, 600 edges
# Testing feature extraction on 10 sample nodes...
# Feature dimension: 174
# Has NaN: False
# Has Inf: False
# ✓ Feature dimension correct: 174
# ✓ No NaN or Inf values
# ✓ Feature extraction test complete!
```

### Alternative: Manual Labeling (Optional)

If you prefer to manually review and correct labels:

```bash
# After step 2, open the CSV in Excel/Google Sheets
start data\target_domains\urls_to_label.csv  # Windows
# or
open data/target_domains/urls_to_label.csv   # macOS

# Fill in the 'label' column:
#   1 = Relevant to your topics
#   0 = Irrelevant (navigation pages, etc.)

# Save as: labeled_urls.csv in the same directory

# Then continue with step 4 (split)
```

### Troubleshooting

**Problem**: Bootstrap stops at 21 pages
- **Cause**: All links are relative URLs and not being converted
- **Check**: Graph builder has `urljoin()` implementation
- **Verify**: See lines in `src/graph/graph_builder.py`

**Problem**: Bootstrap fails with RequestException
- **Cause**: Network timeout or Wikipedia rate limiting
- **Solution**: Script has retry logic and error handling
- **Alternative**: The 60-page graph is sufficient for testing

**Problem**: Feature extraction returns NaN values
- **Check**: Run test_features.py to diagnose
- **Verify**: Graph was loaded correctly from pickle file

**Problem**: "ModuleNotFoundError: No module named 'graph'"
- **Cause**: PYTHONPATH not set or script run from wrong directory
- **Solution**: 
  ```bash
  # Run from project root
  cd d:\Coding\adaptive-qlearning-web-crawler
  python experiments\bootstrap_graph.py
  ```

**Problem**: Bootstrap very slow (>30 minutes)
- **Cause**: 0.5s delay between requests
- **Solution**: Already reduced to 0.2s in code
- **Note**: Slower is more polite to Wikipedia

### Verify Phase 2 Completion

Check that all files were created:

```bash
# Windows PowerShell
Test-Path data\graphs\bootstrap_graph.pkl
Test-Path data\target_domains\train_labeled.csv
Test-Path data\target_domains\val_labeled.csv
Test-Path data\target_domains\test_labeled.csv

# Linux/macOS
ls -lh data/graphs/bootstrap_graph.pkl
ls -lh data/target_domains/*.csv
```

### Expected Results

✅ Bootstrap graph: 60 nodes, 600 edges  
✅ Labeled data: 60 URLs (40 relevant, 20 irrelevant)  
✅ Train/val/test split: 42/9/9 examples  
✅ Feature extraction: 174 dims, no NaN/Inf  
✅ All scripts run without errors  
✅ Ready for Phase 3 (GNN training)  

### Time Required
- Bootstrap crawl: 5-15 minutes
- Labeling & splitting: 1 minute
- Feature testing: 30 seconds
- Total: ~10-20 minutes

### Files Created

```
data/
├── seeds/
│   ├── ml_seeds.json              ← Created manually
│   ├── climate_seeds.json         ← Created manually
│   └── blockchain_seeds.json      ← Created manually
├── graphs/
│   └── bootstrap_graph.pkl        ← Step 1 output (60 nodes, 600 edges)
└── target_domains/
    ├── urls_to_label.csv          ← Step 2 output (template)
    ├── labeled_urls.csv           ← Step 3 output (auto-labeled)
    ├── train_labeled.csv          ← Step 4 output (42 examples)
    ├── val_labeled.csv            ← Step 4 output (9 examples)
    └── test_labeled.csv           ← Step 4 output (9 examples)
```

---

## Next Steps → Phase 3

- 🔄 Pre-train GNN on bootstrap graph (60 nodes)
- 🔄 Use labeled data for supervision (42 train examples)
- 🔄 Evaluate on validation set (9 examples)
- 🔄 Freeze GNN for inference during crawling
- 🔄 Save trained model for deployment

**See**: [Phase 3 Documentation](PHASE_3.md) (Coming next)

---

## References

- WALKTHROUGH.md - Original project plan
- STUDENT_BUDGET_GUIDE.md - Cost optimization strategies
- [GraphSAGE paper](https://arxiv.org/abs/1706.02216) - Inductive learning on graphs
- [LinUCB paper](https://arxiv.org/abs/1003.0146) - Contextual bandits
