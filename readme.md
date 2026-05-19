# DASS Question Reduction using Network Analysis and Machine Learning

**Optimal Questionnaire Shortening via Graph Centrality and Ensemble Learning**

**Psychometrics & ML Project**  
**May 2026**

---

## Project Concept

This project explores **questionnaire optimization** through **network psychometrics** and **machine learning** — a data-driven approach to reduce the DASS-42 (Depression Anxiety Stress Scales) while maintaining predictive accuracy.

The core challenge is: _How many questions can we remove from a 42-item psychological scale while still accurately predicting the total score?_

We implement and compare **four analytical techniques** for question reduction:

- **Variance Filtering** — Remove low-information items
- **Correlation Network** — Model question interdependencies
- **PageRank Centrality** — Identify structurally important questions
- **Random Forest Regression** — Validate predictive power incrementally

All methods are integrated into a **sequential reduction pipeline** that automatically finds the minimal viable question subset.

---

## Technical Approach

### Core Idea

1. Load and preprocess DASS questionnaire data (42 items).
2. Remove questions with near-zero variance (redundant items).
3. Build a **correlation network** where:
   - Nodes = questions
   - Edges = significant correlations (threshold = 0.15)
4. Apply **PageRank** to compute question importance.
5. Train **Random Forest models** incrementally:
   - Start with top-ranked question
   - Add next most important question
   - Evaluate R² score after each addition
6. Stop when **target accuracy (R² ≥ 0.95)** is reached.

### Key Innovations

- Automatic DASS question detection (any `Q* A` format)
- NetworkX-based correlation graph construction
- PageRank centrality as feature importance metric
- Incremental RF training with early stopping
- Dual visualization: accuracy curve + network graph

### Algorithms Summary

| Technique            | Purpose                        | Output                      |
| -------------------- | ------------------------------ | --------------------------- |
| VarianceThreshold    | Remove constant/low-info items | Filtered question set       |
| NetworkX correlation | Model question relationships   | Graph (nodes = questions)   |
| PageRank             | Identify influential questions | Centrality scores           |
| Random Forest        | Validate predictive power      | R² score vs. # of questions |

### Parameters

| Parameter             | Value | Purpose                             |
| --------------------- | ----- | ----------------------------------- |
| Correlation threshold | 0.15  | Edge inclusion threshold in network |
| PageRank alpha        | 0.85  | Teleportation probability           |
| Variance threshold    | 0.01  | Remove near-constant columns        |
| Target R²             | 0.95  | Stopping criterion for reduction    |

---

## Results & Key Findings

### From the actual run:

**Network Statistics:**

- Total DASS Questions: 42
- Remaining after variance filtering: 42
- Nodes: 42 | Edges: 861

**Top Important Questions (by PageRank Centrality):**
| Rank | Question | Centrality |
|------|----------|------------|
| 1 | Q13A | 0.0269 |
| 2 | Q34A | 0.0260 |
| 3 | Q21A | 0.0259 |
| 4 | Q26A | 0.0259 |
| 5 | Q17A | 0.0258 |

**Performance Progression:**
| Questions | R² Score | MAE |
|-----------|----------|-----|
| 5 | 0.7408 | 12.13 |
| 10 | 0.8795 | 8.17 |
| 15 | 0.9304 | 6.18 |
| 19 | 0.9451 | 5.48 |
| 21 | **0.9542** | 4.99 |

**Target R² = 0.95 reached!**

- Minimum questions needed: **21**
- Final R² score: **0.9542**
- Final MAE: **4.99**

The analysis shows that PageRank successfully identifies central questions in the DASS network, and just 21 of the original 42 questions are sufficient to achieve excellent predictive accuracy (R² > 0.95).

---

## Visualizations

The notebook generates two key visualizations:

1. **Accuracy vs. Number of Questions** — Shows how R² score improves as more questions are added, with a clear plateau after ~15 questions.

2. **DASS Question Network** — A graph visualization of all 42 questions, where node sizes represent PageRank importance and edges represent significant correlations (r ≥ 0.15). Larger nodes indicate more influential questions in the network.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- networkx
- matplotlib

Install with:

```bash
pip install pandas numpy scikit-learn networkx matplotlib
```

---

## Goal of This Project

To demonstrate how network analysis (PageRank) combined with ensemble machine learning (Random Forest) can intelligently reduce psychological questionnaires — making mental health assessments shorter, faster, and equally reliable.
