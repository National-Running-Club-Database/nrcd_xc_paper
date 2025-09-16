# National Running Club Database (NRCD) - Cross Country Analysis

This repository contains the code and data analysis for the National Running Club Database (NRCD) research project, focusing on collegiate club athletes' cross country race results.

## Dataset

The NRCD aggregates 15,397 race results from 5,585 athletes across the 2023 and 2024 cross country seasons. The dataset is available in the `data/` folder and contains:
- Athlete information and team associations
- Meet and course details including weather and elevation data
- Race results and performance metrics
- Running event specifications

## Paper

This work is published on arXiv: [National Running Club Database: Assessing Collegiate Club Athletes' Cross Country Race Results](https://arxiv.org/abs/2509.10600)

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

This repository contains comprehensive analysis of collegiate club cross country data, including:
- Athlete performance trends and improvements over time
- Gender participation patterns and performance differences
- Course condition standardization (weather and elevation)
- Team participation and racing frequency analysis
- Improvement prediction modeling

## Key Findings

The analysis addresses three main research questions:

1. **RQ1**: Performance improvement patterns across race positions
   - Runners' improvement per calendar day is more pronounced in athletes with slower initial race times
   - Athletes who race more frequently show greater improvement

2. **RQ2**: Distribution of time changes between races
   - Course conditions significantly impact performance and must be standardized
   - Weather and elevation data enable proper performance comparison

3. **RQ3**: Gender differences in participation and performance
   - Gender imbalance exists in participation (3,484 men vs. 2,101 women) but racing frequency is comparable
   - Performance patterns differ between men's 8,000m and women's 6,000m races

The dataset provides valuable insights for runners, coaches, and teams, bridging the gap between raw data and applied sports science.

## Authors

**Code Authors:**
- Jonathan A. Karr Jr - [ORCID: 0009-0000-1600-6122](https://orcid.org/0009-0000-1600-6122)
- Ryan M. Fryer - [ORCID: 0009-0008-3591-3877](https://orcid.org/0009-0008-3591-3877)

## Reproducibility

All analysis scripts are included with clear documentation. The `analysis.ipynb` notebook provides an interactive overview of the main findings, while individual scripts focus on specific research questions.

## Citation

When using this code or data, please cite our arXiv paper:

```bibtex
@article{karr2025national,
  title={National Running Club Database: Assessing Collegiate Club Athletes' Cross Country Race Results},
  author={Karr Jr, Jonathan A and Darden, Ben and Pell, Nicholas and Fryer, Ryan M and Ambrose, Kayla and Hall, Evan and Bualuan, Ramzi K and Chawla, Nitesh V},
  journal={arXiv preprint arXiv:2509.10600},
  year={2025}
}
```


