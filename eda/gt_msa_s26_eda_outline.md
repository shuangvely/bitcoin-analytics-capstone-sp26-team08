# Exploratory Data Analysis Outline

## 📦 Deliverable: Exploratory Data Analysis Package (Due Feb 26, 2026 at 2:00 am ET)

Each team will submit an EDA package inside this official capstone template repository. The goal is to communicate meaningful insights through a clear narrative that serves both technical and non-technical audiences.

All EDA work must live inside the `EDA` folder of your forked template repository.

---

## ✅ Submission Instructions

- All work must be completed inside your fork of the official template repo
  [https://github.com/TrilemmaFoundation/bitcoin-analytics-capstone-template](https://github.com/TrilemmaFoundation/bitcoin-analytics-capstone-template)

- Place all EDA related notebooks inside the `EDA` folder

- Your submission will consist of **two primary notebooks**

### 1. Required: [`EDA_Executive.ipynb`](EDA_Executive.ipynb)

This is the notebook we will read first.

It should be a polished, narrative driven summary of your most engaging insights.

- Focus on storytelling and interpretation
- Highlight only your strongest findings
- Reference deeper work stored in other notebooks when appropriate

### 2. Required: [`EDA.ipynb`](EDA.ipynb)

This notebook serves as your comprehensive EDA reference.

- Include all important exploratory work
- Show intermediate analysis, experiments, and supporting results
- Use this file as a technical appendix that supports `EDA_Executive.ipynb`

Push your work to your forked GitHub repository by the announced deadline. The latest commit before the deadline is what will be viewed for feedback.

---

## 📋 Notebook Structure

### 1. Executive Summary

Begin `EDA_Executive.ipynb` with an executive bullet point summary of your most novel and non-trivial findings.

A reader should understand your key conclusions in under one minute.

### 2. Data Retrieval

Load data using the official scripts and utilities provided in the template repository:

After running [`download_data.py`](download_data.py) (following the instructions in the root [`README.md`](README.md)), you can simply use `pandas` to read the CSV or Parquet files from your local `data/` directory (see [`eda_starter_template.py`](eda_starter_template.py) for an example).

Clearly state:

- What data sources you used
- Any preprocessing steps
- Assumptions or limitations

### 3. General Dataset Overview

Provide a high level understanding of the dataset.

Include:

- Data integrity checks
- Data types and ranges
- Missingness and completeness
- Descriptive statistics
- Initial exploratory visualizations

### 4. Prediction Market Exploration

Perform a detailed analysis of the Polymarket prediction market data to evaluate its utility for Bitcoin accumulation strategies. This investigation must result in one of two formally justified outcomes:

1. **Could not discover use cases** for the purpose of improving Bitcoin accumulation models (but the investigation was informative).
2. **Discovered interesting use cases** for the purpose of improving Bitcoin accumulation models.

> **Note:** To identify accumulation signals, analyze prediction market data alongside on-chain and macro features. Do not evaluate prediction market utility in isolation; this EDA focuses on cross-feature relationships.

## 📋 Notebook Tips

### Insight Showcase

Demonstrate your strongest analytical thinking.

Examples include:

- Rigorous statistical tests
- Creative or informative visualizations
- Novel metrics or transformations
- Thoughtful comparisons across time or regimes
- Reference supporting work in `EDA.ipynb` when needed.

### Narrative Driven Analysis

Organize your notebook as a clear story.

- Use descriptive section headers, markdown text, and visualizations
- Explain your reasoning and decisions (every output should support your narrative)
- Connect observations to future work and potential modeling approaches

Avoid presenting isolated plots without interpretation, or messy outputs without context.

### Success Tips

- Write with clarity and intention
- Use strong titles and captions
- Prioritize insight over code quantity
- Assume your reader has five minutes to review your work
- Link to additional notebooks in the `EDA` folder for deeper dives
- Include references to additional EDA notebooks for extended analysis (e.g., `EDA/volatility_deep_dive.ipynb` for extended analysis of volatility)

Your goal is to produce an EDA that is both technically rigorous and compelling to read.

---

## 📊 Evaluation Rubric

| Category              | Description                                          |
| --------------------- | ---------------------------------------------------- |
| Readability           | Can a layperson follow the high level narrative      |
| Technical Depth       | Can another data scientist build on your work        |
| Visualization Quality | Are plots clean, labeled, and informative            |
| Statistical Rigor     | Do you investigate deeper structure in the data      |
| Executive Summary     | Is your summary concise and insightful               |
| Engagement            | Is the notebook interesting to explore               |
| Reproducibility       | Can the notebook run top to bottom without errors    |
| Motivating            | Does the EDA motivate future work                    |

---
