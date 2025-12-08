# National Running Club Database (NRCD) - Cross Country Analysis

This repository contains the code and data analysis for the National Running Club Database (NRCD) research project, focusing on collegiate club athletes' cross country race results.

## Dataset Setup

**Important**: This is an anonymous submission. The dataset used in this analysis should be inserted into the `data/` folder. Please refer to the paper for the dataset source and citation.

To set up the dataset:
1. Place the dataset files in the `data/` directory
2. Ensure the data structure matches the expected format used by the analysis scripts
3. The dataset should contain running event data with athlete, team, meet, and result information

## Project Structure

- **`scripts/`**: Analysis scripts and notebooks
  - Main analysis: `rq1.py`, `rq2.py`, `rq3.py` (orchestrate all analyses)
  - ML model: `ml_improvement_prediction.py` (main ML model with temporal validation)
  - Specialized analyses: gender analysis, team participation, race counts, top 25 teams, nationals overlap
  - Visualization scripts for key findings (including combined overlay plots for 2023-2025)

- **`output/`**: Analysis outputs organized by research question
  - `rq1/`: Performance improvement patterns (ML model, race analyses, top 25 teams, nationals overlap, overlay plots)
  - `rq2/`: Multi-season analysis with race count consistency filter
  - `rq3/`: Gender differences (participation, feature importance)
  - `FINDINGS_EXPLANATION.md`: Comprehensive findings summary

- **`key_visualizations/`**: Additional generated visualizations organized by research questions
  - RQ1: Additional visualization scripts and outputs
  - RQ2: Time change distribution analysis  
  - RQ3: Gender differences analysis

- **`data/`**: Dataset storage (see setup instructions above)

## Setup

### Virtual Environment Setup (Recommended)

It's recommended to use a virtual environment to isolate dependencies. The project uses a virtual environment named `venv`:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Dependencies

The analysis requires the following Python packages:
- pandas, numpy, scipy (data manipulation and statistics)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning and statistical analysis)
- ipython, notebook (optional, for interactive analysis)

Install dependencies with:
```bash
pip install -r requirements.txt
```

**Note**: Make sure your virtual environment is activated before installing dependencies and running scripts.

## Statistical Methods

The analysis employs rigorous statistical methods:

- **Temporal Validation**: Primary validation method - train on 2023, test on 2024 (prevents data leakage)
- **5-Fold Cross-Validation**: Model stability assessment on training data
- **Bootstrap Confidence Intervals**: 1000 resamples for robust uncertainty estimates (95% CI)
- **Multiple Comparisons Corrections**: Bonferroni corrections applied to all pairwise comparisons
  - Model comparisons: 15 comparisons (6 models) → α = 0.05/15 = 0.0033
  - Nationals overlap: 6 comparisons → α = 0.05/6 = 0.0083
  - Top 25 teams: 3 comparisons → α = 0.05/3 = 0.0167
- **Sensitivity Analysis**: Tests impact of potentially problematic features (e.g., `last_time`)
- **Chi-square Tests**: For categorical associations (nationals overlap analysis)
- **Pearson & Spearman Correlations**: For continuous relationships (top 25 teams analysis)

## Running the Analysis

All scripts are run from the **main directory** (not from inside `scripts/`).

### Option 0: Run All Research Questions (Complete Analysis)

Run all three research questions sequentially:

```bash
# Run all RQ1, RQ2, and RQ3 analyses
python scripts/run_all.py
```

This will execute:
- **RQ1**: Performance improvement patterns across race positions
- **RQ2**: Multi-season performance analysis with race count consistency filter
- **RQ3**: Gender differences in participation and performance

All outputs will be saved to their respective `output/rq1/`, `output/rq2/`, and `output/rq3/` directories. The script will display progress and total execution time.

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
# Combined overlay plots (2023, 2024, 2025) are automatically generated when running rq1.py
# To run individually: python scripts/create_combined_overlay_2023_2024_2025_mens.py
#                     python scripts/create_combined_overlay_2023_2024_2025_womens.py
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
python scripts/nationals_overlap_analysis.py  # Top 15 teams vs 4+ race athletes
python scripts/top25_team_analysis.py  # Top 25 teams correlation analysis
python scripts/state_race_results_map.py
python scripts/create_visualizations.py
```

### Main ML Model Details

The main machine learning model (`ml_improvement_prediction.py`) performs comprehensive analysis with 3-year temporal validation:
- **Primary**: Train on 2023, test on 2024
- **Generalization**: Train on 2023, test on 2025
- **Extended**: Train on 2023+2024, test on 2025

This generates:
- `raw_data_athlete_features.csv` - All athlete features used in the model
- `raw_data_feature_importance.csv` - Feature importance rankings
- `raw_data_model_performance.csv` - Model performance metrics (R², RMSE, MAE)
- `raw_data_important_statistics.csv` - Summary statistics
- `raw_data_gender_feature_importance_*.csv` - Gender-specific feature importance
- `raw_data_subgroup_analysis.csv` - Performance by year and gender
- `time_standardization_comparison.csv` - Comparison of standardization methods
- `sensitivity_analysis_last_time.csv` - Sensitivity analysis for `last_time` feature
- Various visualizations (PDFs)

**Statistical Rigor:**
- **Temporal Validation**: Train on 2023, test on 2024 (prevents data leakage)
- **Multiple Comparisons**: Bonferroni corrections applied to all statistical tests
- **Sensitivity Analysis**: Tests impact of `last_time` feature (potential data leakage)
- **Bootstrap Confidence Intervals**: 1000 resamples for robust uncertainty estimates
- **5-Fold Cross-Validation**: Model stability assessment on training data

**Note**: `ml_improvement_prediction.py` is the main model used in the paper. It uses temporal validation to ensure the model generalizes to future years, which is critical for predicting athlete improvement. The model compares three standardization methods using the **same model** (best from RQ1) for fair comparison:
- **Raw**: Distance conversion to 6k/8k only (no course distance, weather, or terrain adjustments)
- **Converted**: Distance conversion + course distance adjustment (adjusts for long/short courses, but no weather/terrain)
- **Standardized**: Full standardization (distance + course distance + weather + terrain adjustments)

All three methods exclude nationals data for better model logic.

### Output Organization

Outputs are organized by research question:
- **`output/rq1/`**: Performance improvement patterns
  - ML model results (`raw_data_*.csv`, `raw_data_*.pdf`)
  - First-to-last improvement analysis
  - Number of races analyses
  - Team race participation
  - **Top 25 teams at nationals** (`top25_teams/`)
  - **Nationals overlap analysis** (`nationals_overlap/`) - Top 15 teams vs 4+ race athletes
  - **Combined overlay plots** (`overlay_plots/`) - Combined 2023, 2024, 2025 overlay plots for men and women
  - State race results map
  - Time standardization comparison
  - Sensitivity analysis for `last_time` feature

- **`output/rq2/`**: Multi-season analysis
  - Distribution of fastest times (2023-2025)
  - Improvement patterns between seasons
  - ML model with race count consistency filter
  - Comprehensive plots by gender

- **`output/rq3/`**: Gender differences
  - Gender-specific feature importance
  - Participation patterns
  - Performance differences

- **`output/FINDINGS_EXPLANATION.md`**: Comprehensive findings summary with all results

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
   - **Best model (Random Forest) achieves 91.5% accuracy** (R² = 0.9145) using temporal validation
   - **Experience level** (num_races × season_duration) is the strongest predictor (21.2% importance)
   - Athletes who race more frequently show greater improvement
   - **Top 15 teams at nationals have significantly more athletes racing 4+ times** (60-80% overlap, Bonferroni-corrected significance in 3/6 categories)
   - Teams with athletes racing 4+ times are 2-3x more likely to make top 15 at nationals

2. **RQ2**: Multi-season analysis with race count consistency
   - Filtering for consistent race participation (difference < 2 races between consecutive seasons) reveals stable improvement patterns
   - Distribution of fastest times shows consistent trends across years (2023-2025)
   - Machine learning models achieve 82.8% accuracy (Gradient Boosting) with race count consistency filter

3. **RQ3**: Gender differences in participation and performance
   - Model shows fair performance across genders (Women: 94.5% R², Men: 90.4% R², 4.1% difference)
   - Different features matter for men vs women (gender-specific feature importance)
   - Gender differences in race participation patterns exist but are well-captured by the model

**Statistical Rigor:**
- All statistical tests use Bonferroni corrections for multiple comparisons
- Temporal validation prevents data leakage (train on 2023, test on 2024)
- Bootstrap confidence intervals (1000 resamples) for robust uncertainty estimates
- Sensitivity analysis validates feature selection

The dataset provides valuable insights for runners, coaches, and teams, bridging the gap between raw data and applied sports science.

## Authors

**Code Authors:**
Hidden for blind revision

## Reproducibility

All analysis scripts are included with clear documentation. Key features:

- **Temporal Validation**: All models use temporal splits (train on earlier years, test on later years)
- **Data Leakage Prevention**: Features calculated using only training data (e.g., percentiles based on 2023 only)
- **Multiple Comparisons Corrections**: Bonferroni corrections applied to all statistical tests
- **Sensitivity Analysis**: Automatic testing of potential data leakage features
- **Comprehensive Outputs**: All results saved to CSV/PDF with clear naming conventions

The `output/FINDINGS_EXPLANATION.md` file provides a comprehensive summary of all findings, methodologies, and statistical validation approaches.



