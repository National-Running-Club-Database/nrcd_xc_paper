# National Running Club Database (NRCD) - Cross Country Analysis

This repository contains the code and data analysis for the National Running Club Database (NRCD) research project, focusing on collegiate club athletes' cross country race results.

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

- **`output/`**: Analysis outputs organized by research question
  - `rq1/`: Performance improvement patterns (ML model, race analyses)
  - `rq2/`: Time change distribution (percentile analysis, standardization)
  - `rq3/`: Gender differences (participation, feature importance)
  - General outputs: Charts, summaries, supporting analyses

- **`key_visualizations/`**: Generated visualizations organized by research questions
  - RQ1: Fastest race position analysis
  - RQ2: Time change distribution analysis  
  - RQ3: Gender differences analysis

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

## Running the Analysis

All scripts are run from the **main directory** (not from inside `scripts/`).

### Option 1: Run by Research Question (Recommended)

Organized by research questions with outputs in `output/rq1/`, `output/rq2/`, `output/rq3/`:

```bash
# RQ1: Performance improvement patterns
python scripts/rq1.py

# RQ2: Distribution of time changes between races
python scripts/rq2.py

# RQ3: Gender differences in participation and performance
python scripts/rq3.py

# Generate all research charts and summaries
python scripts/create_charts.py
```

### Option 2: Run Individual Scripts

If you need to run specific analyses:

#### RQ1: Performance Improvement Patterns
```bash
python scripts/first_to_last_improvement.py
python scripts/numberOfRacesQuestion.py
python scripts/numberOfRacesBrokenDown.py
python scripts/ml_improvement_prediction.py  # Main ML model with 3-year validation
```

#### RQ2: Time Change Distribution
```bash
python scripts/percentile_time_analysis_final.py
# Time standardization comparison is included in ml_improvement_prediction.py
```

#### RQ3: Gender Differences
```bash
python scripts/gender_race_participation_test.py
# Gender-specific feature importance is included in ml_improvement_prediction.py
```

#### Additional Analyses (Supporting)
```bash
python scripts/analyze_data_coverage.py
python scripts/meet_course_analysis.py
python scripts/team_race_participation.py
python scripts/nationals_overlap_analysis.py
python scripts/state_race_results_map.py
python scripts/create_visualizations.py
```

### Main ML Model Details

The main machine learning model (`ml_improvement_prediction.py`) performs comprehensive analysis with 3-year temporal validation:
- **Primary**: Train on 2023, test on 2024
- **Generalization**: Train on 2023, test on 2025
- **Extended**: Train on 2023+2024, test on 2025

This generates:
- `raw_data_athlete_features.csv`
- `raw_data_feature_importance.csv`
- `raw_data_model_performance.csv`
- `raw_data_3year_validation_results.csv`
- `raw_data_important_statistics.csv`
- `raw_data_gender_feature_importance_*.csv`
- Various visualizations (PDFs)

**Note**: `ml_improvement_prediction.py` is the main model used in the paper. It uses temporal validation to ensure the model generalizes to future years, which is critical for predicting athlete improvement. The model compares three standardization methods:
- **Raw**: Distance conversion to 6k/8k only (no course distance, weather, or terrain adjustments)
- **Converted**: Distance conversion + course distance adjustment (adjusts for long/short courses, but no weather/terrain)
- **Standardized**: Full standardization (distance + course distance + weather + terrain adjustments)

All three methods exclude nationals data for better model logic.

### Output Organization

Outputs are organized by research question:
- **`output/rq1/`**: Performance improvement patterns (ML model, race count analyses)
- **`output/rq2/`**: Time change distribution (percentile analysis, standardization)
- **`output/rq3/`**: Gender differences (participation, feature importance)
- **`output/`**: General outputs (charts, summaries, supporting analyses)

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
- Jonathan A. Karr Jr - [![ORCID](https://img.shields.io/badge/ORCID-0009--0000--1600--6122-green.svg)](https://orcid.org/0009-0000-1600-6122)
- Ryan M. Fryer - [![ORCID](https://img.shields.io/badge/ORCID-0009--0008--3591--3877-green.svg)](https://orcid.org/0009-0008-3591-3877)

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


