# JSS Paper - Anonymous Submission

This repository contains the code and data analysis for the Journal of Sports Science (JSS) paper submission.

## Dataset Setup

**Important**: This is an anonymous submission. The dataset used in this analysis should be inserted into the `data/` folder. Please refer to the paper for the dataset source and citation.

To set up the dataset:
1. Place the dataset files in the `data/` directory
2. Ensure the data structure matches the expected format used by the analysis scripts
3. The dataset should contain running event data with athlete, team, meet, and result information

## Project Structure

- **`scripts/`**: Analysis scripts and notebooks (13 files)
  - Main analysis: `analysis.py` and `analysis.ipynb`
  - Specialized analyses: gender analysis, team participation, race counts, etc.
  - Visualization scripts for key findings

- **`key_visualizations/`**: Generated visualizations organized by research questions
  - RQ1: Fastest race position analysis
  - RQ2: Time change distribution analysis  
  - RQ3: Gender differences analysis

- **`output/`**: Detailed analysis outputs and summaries
  - Gender analysis results
  - Race participation statistics
  - Team overlap analysis

- **`data/`**: Dataset storage (see setup instructions above)

## Dependencies

The analysis requires the following Python packages:
- pandas, scipy, ipython, notebook
- matplotlib, seaborn, plotly (visualization)
- scikit-learn (statistical analysis)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Analysis Overview

This repository contains comprehensive analysis of running event data, including:
- Athlete performance trends and improvements
- Gender participation patterns and differences
- Team participation analysis
- Race frequency and retention analysis
- Geographic distribution of results

## Key Findings

The analysis addresses three main research questions:
1. **RQ1**: Performance improvement patterns across race positions
2. **RQ2**: Distribution of time changes between races
3. **RQ3**: Gender differences in participation and performance

## Reproducibility

All analysis scripts are included with clear documentation. The `analysis.ipynb` notebook provides an interactive overview of the main findings, while individual scripts focus on specific research questions.

## Citation

When using this code, please cite the associated JSS paper (reference to be added upon publication).


