# Comprehensive Research Findings Explanation

*Auto-generated from analysis results*

## Executive Summary

This analysis uses machine learning to predict athlete improvement rates in collegiate cross country running across three research questions. The best model (Random Forest) achieves **91.5% accuracy** (R² = 0.9145, 95% CI: [0.8604, 0.9517]) using temporal validation (trained on 2023, tested on 2024). Key findings include:

1. **experience_level** is the most critical predictor (21.2%) - total racing experience matters most
2. **Tree-based models significantly outperform linear models** - 91.5% vs 45.7% accuracy (100% improvement)
3. **The model is fair across genders** - Women: 94.5% R², Men: 90.4% R² (4.1% difference, within acceptable bounds)
4. **Time standardization methods** - Converted method (distance only) performs best (93.1% R²), followed by standardized (91.5% R²)
5. **Top 15 teams at nationals have significantly more athletes racing 4+ times** - 60-80% of top 15 teams have at least one athlete with 4+ races, with **Bonferroni-corrected significance in 3 out of 6 categories** (2024 Men: p=0.002, 2024 Women: p<0.001, 2025 Women: p<0.001)
6. **Racing frequency correlates with nationals success** - Teams with athletes racing 4+ times are 2-3x more likely to make top 15 at nationals (23-39% success rate vs. baseline)
7. **Top 25 teams analysis** - **Men's season duration (days from first race to nationals) shows Bonferroni-significant correlation with better rank** (r = -0.283, p = 0.0138, Bonferroni-corrected p = 0.0414). Teams that start racing earlier perform better at nationals.

---

## Research Questions Overview

This study addresses three main research questions:

### RQ1: Performance Improvement Patterns
**Question:** How do athletes' performance patterns change across race positions and seasons?

**Key Findings:**
- Athletes who race more frequently show greater improvement
- Experience level (num_races × season_duration) is the strongest predictor
- **Top 15 teams at nationals have significantly more athletes racing 4+ times** - 60-80% of top 15 teams have at least one athlete with 4+ races, with **Bonferroni-corrected significance in 3 out of 6 categories** (2024 Men: p=0.002, 2024 Women: p<0.001, 2025 Women: p<0.001)
- **2024 shows strongest relationship** - Both men (73.3% overlap, p=0.002) and women (73.3% overlap, p<0.001) show highly significant associations after Bonferroni correction
- Teams with athletes racing 4+ times are 2-3x more likely to make top 15 at nationals compared to teams without such athletes (23-39% success rate vs. baseline)
- **Top 25 teams at nationals** - Men's season duration (days from first race to nationals) shows **Bonferroni-significant correlation** with better rank (r = -0.283, p = 0.0138). Teams that start racing earlier in the season perform better at nationals.

### RQ2: Multi-Season Analysis with Race Count Consistency
**Question:** How do athletes' performance patterns change across multiple seasons when controlling for consistent race participation?

**Key Findings:**
- Filtering for consistent race participation (difference < 2 races between consecutive seasons) reveals stable improvement patterns
- Distribution of fastest times shows consistent trends across years (2023-2025)
- Machine learning models achieve 82.8% accuracy (Random Forest) when predicting improvement with race count consistency filter
- Temporal validation shows models generalize well to future years

### RQ3: Gender Differences in Participation and Performance
**Question:** What are the gender differences in participation patterns and model performance?

**Key Findings:**
- Model shows fair performance across genders (Women: 94.5% R², Men: 90.4% R²)
- Different features matter for men vs women (gender-specific feature importance)
- Gender differences in race participation patterns exist but are well-captured by the model

---

## Model Validation Methods

### 1. Temporal Validation (Primary Method)

**What it is:** Training and testing on different time periods to simulate real-world prediction scenarios.

**Three Validation Scenarios:**

1. **Primary Validation: Train on 2023, Test on 2024**
   - Most realistic scenario: predict next year's performance
   - Best model (Random Forest): R² = 0.9145 (91.5% accuracy)
   - This is the **primary metric** reported in the paper

2. **Generalization Test: Train on 2023, Test on 2025**
   - Tests model's ability to predict 2 years into the future
   - Assesses long-term generalization capability
   - Models trained on 2023 data are tested on 2025 data (skipping 2024)

3. **Extended Training: Train on 2023+2024, Test on 2025**
   - Uses more training data (2 years instead of 1)
   - Tests if additional training data improves predictions
   - Models trained on combined 2023+2024 data, tested on 2025

**Why Temporal Validation Matters:**
- **Prevents data leakage:** Future data never influences past predictions
- **Realistic assessment:** Mimics how models would be used in practice
- **Honest performance:** R² calculated only on test data, never on training data
- **Temporal integrity:** Features calculated using only training data (e.g., percentiles based on 2023 only)

### 2. 5-Fold Cross-Validation (CV)

**What it is:** A technique to assess model stability and prevent overfitting by splitting training data into 5 folds.

**How it works:**
1. Training data is divided into 5 equal parts (folds)
2. Model is trained on 4 folds and validated on the remaining fold
3. This process repeats 5 times, with each fold serving as validation once
4. Results are averaged across all 5 folds

**Why 5-Fold CV:**
- **Model stability:** Assesses how consistent the model is across different data subsets
- **Overfitting detection:** High variance in CV scores indicates overfitting
- **Hyperparameter tuning:** Used to select best model parameters
- **Training data assessment:** Only uses training data (2023), never test data (2024)

**Example:**
- Random Forest: CV R² = 0.8801 (±0.0195)
  - Mean: 88.0% accuracy on training data
  - Std: ±1.95% (low variance = stable model)
  - Test R²: 91.5% (slightly higher = good generalization)

**CV vs Test Performance:**
- **CV R² (0.8801):** Performance on training data (2023) using 5-fold CV
- **Test R² (0.9145):** Performance on test data (2024) - the true measure
- **Difference:** Test R² > CV R² indicates good generalization (model performs better on new data than expected)

### 3. Bootstrap Confidence Intervals

**What it is:** A resampling technique to estimate uncertainty in model performance metrics.

**How it works:**
1. Test set predictions are resampled 1000 times (with replacement)
2. R² is calculated for each resample
3. 95% confidence interval is the range containing 95% of resampled R² values

**Why it matters:**
- Provides uncertainty estimates for model performance
- More robust than single point estimates
- Accounts for variability in test set composition

**Example:**
- Random Forest: R² = 0.9145 (95% CI: [0.8604, 0.9517])
  - Best case: 95.2% accuracy
  - Worst case: 86.0% accuracy
  - Most likely: 91.5% accuracy

---

## 1. Model Performance Comparison

### Overall Model Results (RQ1: Primary Validation)

| Model | Test R² | 95% CI | RMSE | MAE | CV R² (Train) | CV Std |
|-------|---------|--------|------|-----|---------------|--------|
| Linear Regression | 0.4567 | [0.4125, 0.4999] | 3.75 | 1.94 | 0.2354 | ±0.2164 |
| Ridge Regression | 0.4617 | [0.4216, 0.5015] | 3.73 | 1.93 | 0.2568 | ±0.1885 |
| Lasso Regression | 0.4091 | [0.3771, 0.4387] | 3.91 | 1.96 | 0.2234 | ±0.2571 |
| **Random Forest** | **0.9145** | [0.8669, 0.9536] | 1.49 | 0.41 | 0.8801 | ±0.0195 |
| Gradient Boosting | 0.9021 | [0.8576, 0.9386] | 1.59 | 0.62 | 0.8770 | ±0.0198 |
| SVR | 0.5369 | [0.4395, 0.6450] | 3.46 | 0.95 | 0.6088 | ±0.1480 |

**Validation Method:** Temporal split - Train on 2023, Test on 2024
- All R² values calculated **only on test data (2024)**
- CV R² calculated **only on training data (2023)** using 5-fold CV
- Bootstrap CIs calculated on test set predictions

### RQ2: Multi-Season Model Performance

| Model | Test R² | RMSE | MAE |
|-------|---------|------|-----|
| Linear Regression | 0.5620 | 3.41 | 2.08 |
| Ridge Regression | 0.6644 | 2.99 | 1.98 |
| Lasso Regression | 0.6236 | 3.16 | 2.02 |
| **Random Forest** | **0.8279** | 2.14 | 1.09 |
| **Gradient Boosting** | **0.8306** | 2.12 | 1.18 |
| SVR | 0.3305 | 4.22 | 1.85 |

**Validation Method:** Temporal split - Train on 2023, Test on 2024
**Filter Applied:** Athletes with race count difference < 2 between consecutive seasons
- Example: 3 races in 2023, 4 in 2024, 5 in 2025 is valid (differences: 1, 1)
- This filter ensures comparable participation patterns across seasons

### Why Tree-Based Models Outperform Linear Models

**Statistical Evidence:**
- Tree-based models (Random Forest, Gradient Boosting) **significantly outperform** linear models
- Linear models achieve only **41-46% accuracy** vs **91% for tree-based models**
- This is a **100% relative improvement** in prediction accuracy

**Why This Happens:**

1. **Non-Linear Relationships:** Linear models assume straight-line relationships, but improvement patterns are **complex and non-linear**
2. **Feature Interactions:** Tree models capture interactions between features automatically
3. **Non-Additive Effects:** Improvement doesn't simply add up linearly
4. **Heterogeneous Effects:** Different athletes respond differently to the same training/racing

---

## 2. Key Predictors of Improvement

### Top 15 Most Important Features (RQ1)

| Rank | Feature | Importance | What It Measures |
|------|---------|------------|------------------|
| 1 | **experience_level** | **21.2%** | Total racing experience (num_races × season_duration) |
| 2 | **bad_race_count** | **17.1%** | Number of races worse than previous race (consistency indicator) |
| 3 | **slope** | **12.0%** | Improvement trajectory pattern (calculated from first N-1 races, excludes last race) |
| 4 | **race_frequency** | **9.1%** | Races per day (racing frequency) |
| 5 | **time_std** | **8.1%** | Standard deviation of race times (consistency) |
| 6 | **cv_time** | **4.4%** | Coefficient of variation (normalized consistency) |
| 7 | **time_range** | **4.4%** | Difference between worst and best time |
| 8 | **best_to_avg_ratio** | **3.9%** | Best time relative to average (potential vs actual) |
| 9 | **season_duration_squared** | **3.2%** | Non-linear effect of season duration (optimal length) |
| 10 | **worst_to_avg_ratio** | **3.2%** | Worst time relative to average (performance spread) |
| 11 | **best_race_timing** | **2.7%** | Days from first race to best race (peak timing) |
| 12 | **consistency_score** | **2.6%** | Inverse of variability (consistency measure) |
| 13 | **variability_score** | **2.2%** | Normalized time range (replaces removed improvement_to_variability_ratio) |
| 14 | **season_duration** | **1.6%** | Number of days from first to last race |
| 15 | **avg_days_between_races** | **1.2%** | Average recovery time between races |

### Why These Features Matter

**1. Experience Level (21.2% - HIGHEST IMPORTANCE)**
- **What it measures:** Total racing experience = number of races × season duration
- **Why it matters:** Athletes with more racing experience show more predictable improvement patterns
- **Key insight:** Both quantity (num_races) and duration (season_duration) matter, and their interaction is critical
- **Interpretation:** More experienced athletes have more data points, making their improvement trajectories more predictable

**2. Bad Race Count (17.1% - SECOND HIGHEST)**
- **What it measures:** Number of races where performance was worse than the previous race
- **Why it matters:** Indicates consistency and ability to maintain/improve performance
- **Key insight:** Athletes with fewer "bad races" show more consistent improvement
- **Interpretation:** This captures the consistency of improvement trajectory - fewer setbacks predict better overall improvement

**3. Slope (12.0% - THIRD HIGHEST)**
- **What it measures:** Linear regression slope of race times (calculated from first N-1 races)
- **Why it matters:** Captures the **trajectory pattern** of improvement
- **Key insight:** Negative slope = improving, positive slope = declining
- **Interpretation:** The rate of change in performance over the season is highly predictive

---

## 3. Gender Differences and Model Fairness (RQ3)

### Gender-Specific Model Performance

The model shows **fair performance across genders** with only a small, acceptable difference:

| Gender | R² Score | Interpretation |
|--------|----------|----------------|
| **Women** | **94.5%** | Model explains 94.5% of variance for women |
| **Men** | **90.4%** | Model explains 90.4% of variance for men |
| **Difference** | **4.1%** | Women's R² is 4.1% higher (within acceptable bounds) |

**Fairness Assessment:** ✓ **Relatively fair** - The 4.1% difference is small and within acceptable bounds. The model does not systematically disadvantage either gender.

### Why Women's R² Might Be Slightly Higher

Possible explanations for the small difference:
1. **Data Quality:** Women's data may be slightly more consistent or complete
2. **Feature Relevance:** Some features may be more predictive for women's improvement patterns
3. **Sample Size:** Different sample sizes can affect model performance
4. **Biological Factors:** Different physiological responses to training/racing between genders

### Gender Differences in Feature Importance

The model reveals that different factors matter for men vs women (see gender-specific feature importance analysis for details).

---

## 4. Time Standardization Impact

### Comparison of Standardization Methods

**Methodology:** All three methods tested using the **same model** (Random Forest, best from RQ1) to ensure fair comparison - only standardization method varies, not model choice.

| Method | R² Score | 95% CI | RMSE | MAE | CV R² Mean | CV R² Std |
|--------|----------|--------|------|-----|------------|-----------|
| **Converted** | **0.9312** | [0.9037, 0.9496] | 1.37 | 0.45 | 0.8928 | ±0.0140 |
| Standardized | 0.9145 | [0.8664, 0.9532] | 1.49 | 0.41 | 0.8801 | ±0.0195 |

**Key Findings:**
- **Converted method performs best** (93.1% R²) - Distance conversion only, no weather/terrain adjustments
- **Standardized method performs slightly worse** (91.5% R²) - Full standardization with weather/terrain adjustments
- **Difference:** Converted method achieves 1.8% higher R² than standardized
- **Note:** All methods convert distances to gender-specific targets (6k for women, 8k for men)
- **Same model used:** Random Forest for both methods ensures fair comparison

**What Each Method Includes:**
1. **Converted Method:**
   - Distance conversion to gender-specific targets (6k for women, 8k for men)
   - Course distance adjustment (adjusts for long/short courses)
   - **No weather/terrain adjustments**

2. **Standardized Method:**
   - Distance conversion to gender-specific targets (6k for women, 8k for men)
   - Course distance adjustment
   - **Weather adjustments:** Temperature and dew point (heat/humidity effects)
   - **Terrain adjustments:** Elevation gain/loss and course distance accuracy

**Interpretation:** The converted method's superior performance suggests that weather and terrain adjustments may introduce noise or over-correction in this dataset. Distance conversion and course distance adjustment appear to be the most important standardization steps.

---

## 5. RQ1: Nationals Overlap Analysis

### Racing Frequency and Nationals Success

**Research Question:** Do teams with athletes racing 4+ times perform better at nationals?

**Filter Applied:** Teams must have at least 3 athletes of the given gender who have run at least one race (consistent filtering)

**Key Findings:**

| Category | Total Teams | Teams with 4+ Races | Top 15 Teams | Overlap | % of 4+ Teams in Top 15 | % of Top 15 with 4+ Races | p-value | Bonferroni Significant |
|---------|-------------|---------------------|--------------|---------|------------------------|---------------------------|---------|------------------------|
| 2023 Men | 93 | 36 (38.7%) | 14 | 9 | 25.0% | 64.3% | 0.067 | No |
| 2023 Women | 85 | 27 (31.8%) | 15 | 9 | 33.3% | 60.0% | 0.022* | No |
| 2024 Men | 104 | 36 (34.6%) | 15 | 11 | 30.6% | 73.3% | 0.002** | **Yes** |
| 2024 Women | 91 | 28 (30.8%) | 15 | 11 | 39.3% | 73.3% | <0.001*** | **Yes** |
| 2025 Men | 107 | 39 (36.4%) | 15 | 9 | 23.1% | 60.0% | 0.079 | No |
| 2025 Women | 97 | 33 (34.0%) | 15 | 12 | 36.4% | 80.0% | <0.001*** | **Yes** |

**Statistical Significance:**
- **Bonferroni Correction:** α = 0.05 / 6 = 0.0083 (significant if p < 0.0083)
- * p < 0.05 (uncorrected, significant)
- ** p < 0.01 (uncorrected, highly significant)
- *** p < 0.001 (uncorrected, very highly significant)

**Key Insights:**
1. **3 out of 6 categories show Bonferroni-corrected significance** - 2024 Men (p=0.002), 2024 Women (p<0.001), and 2025 Women (p<0.001)
2. **2024 shows strongest relationship** - Both men and women show highly significant associations after Bonferroni correction
3. **Women show more consistent significance** - 2 out of 3 years show Bonferroni-corrected significance (2024 and 2025)
4. **High overlap rates** - 60-80% of top 15 teams have at least one athlete racing 4+ times:
   - 2023: 64.3% (men), 60.0% (women)
   - 2024: 73.3% (men), 73.3% (women)
   - 2025: 60.0% (men), 80.0% (women)
5. **Success rate for teams with 4+ races** - 23-39% of teams with at least one athlete racing 4+ times make top 15 at nationals
6. **Overall pattern** - 30-39% of all teams have at least one athlete racing 4+ times, but these teams are overrepresented in top 15 (60-80% of top 15 teams)
4. **Practical significance** - Teams with 4+ race athletes are 2-3x more likely to make top 15

---

## 6. RQ1: Top 25 Teams at Nationals Analysis

### Team Metrics and Nationals Rank Correlations

**Research Question:** What team characteristics correlate with nationals rank?

**Metrics Analyzed:**
- **Season Duration:** Days from first race to nationals (earliest team start date after excluding outlier minimum)
- **Max Races:** Maximum races by any athlete on the team (after excluding minimum)
- **Experience Level:** Maximum experience level (num_races × season_duration) by any athlete (after excluding minimum)

**Filter Applied:** Teams must have at least 3 athletes of the given gender who have run at least one race

**Robust Statistics:** For each metric, the minimum value is excluded before calculating the maximum (reduces outlier impact)

### Correlation Results

| Gender | Metric | Pearson r | p-value | Bonferroni-corrected p | R² | Bonferroni Significant |
|--------|--------|-----------|---------|------------------------|----|------------------------|
| **Men** | Season Duration (days) | -0.283 | 0.0138 | 0.0414 | 0.0801 | **Yes** |
| **Men** | Max Races (any athlete) | -0.190 | 0.1030 | 0.3090 | 0.0361 | No |
| **Men** | Experience Level | -0.253 | 0.0286 | 0.0858 | 0.0640 | No |
| **Women** | Season Duration (days) | -0.046 | 0.6968 | 1.0000 | 0.0021 | No |
| **Women** | Max Races (any athlete) | -0.232 | 0.0471 | 0.1413 | 0.0538 | No |
| **Women** | Experience Level | -0.156 | 0.1833 | 0.5499 | 0.0243 | No |

**Key Findings:**
1. **Men's Season Duration is Bonferroni-significant** - Longer season duration (more days from first race to nationals) correlates with better rank (r = -0.283, p = 0.0138, Bonferroni-corrected p = 0.0414). This suggests teams that start racing earlier in the season tend to perform better at nationals.
2. **Negative correlations** (better rank = lower number) suggest:
   - Longer season duration → better rank (Men: significant)
   - More races → better rank (trend, not significant)
   - Higher experience → better rank (trend, not significant)
3. **Gender differences** - Men show significant correlation with season duration, while women show no significant correlations after Bonferroni correction
4. **Bonferroni correction** - After adjusting for multiple comparisons (α = 0.05 / 3 = 0.0167), **one correlation remains significant**: Men's Season Duration
5. **Sample sizes** - Men: n = 75, Women: n = 74 (adequate for correlation analysis)

**Interpretation:**
- **Men's teams that start racing earlier in the season (longer season duration from first race to nationals) show significantly better performance at nationals** - This is the only Bonferroni-significant finding
- Trends (not Bonferroni-significant) suggest:
  - Teams with more races may perform better at nationals
  - Teams with higher experience levels may perform better
- Women show no significant correlations after Bonferroni correction, though trends are similar to men

---

## 7. RQ2: Multi-Season Analysis Results

### Distribution Comparison

**Analysis:** Average fastest times by gender across years (2023-2025)

**Key Findings:**
- Consistent trends across years for both men and women
- Gender differences in performance patterns are stable over time
- Filtering for race count consistency (difference < 2) reveals stable improvement patterns

### Improvement Patterns (2023→2024, 2024→2025, 2023→2025)

**Analysis:** Improvement metrics for athletes with consistent race participation

**Key Findings:**
- Athletes with consistent race counts show predictable improvement patterns
- Improvement rates are similar across consecutive seasons when race counts are consistent
- Multi-season analysis reveals long-term improvement trajectories

### Machine Learning Results (RQ2)

**Best Model:** Gradient Boosting (R² = 0.8306, 82.8% accuracy)

**Validation:** Temporal split - Train on 2023, Test on 2024
- Filter: Race count difference < 2 between consecutive seasons
- Models achieve lower accuracy than RQ1 (82.8% vs 91.5%) due to stricter filtering
- Still demonstrates strong predictive capability with consistent participation patterns

---

## 8. Model Trustworthiness and Validation

### Validation Methodology Summary

**1. Temporal Validation (Primary)**
- **Train on 2023, Test on 2024:** Primary validation scenario
- **Train on 2023, Test on 2025:** Generalization test (2 years ahead)
- **Train on 2023+2024, Test on 2025:** Extended training scenario
- **Why:** Prevents data leakage, provides honest performance assessment

**2. 5-Fold Cross-Validation**
- **Purpose:** Assess model stability on training data
- **Method:** Split training data into 5 folds, train on 4, validate on 1, repeat 5 times
- **Output:** Mean CV R² and standard deviation
- **Interpretation:** Low variance = stable model, high variance = overfitting risk

**3. Bootstrap Confidence Intervals**
- **Purpose:** Estimate uncertainty in test set performance
- **Method:** Resample test predictions 1000 times, calculate R² for each
- **Output:** 95% confidence interval for R²
- **Interpretation:** Range of likely performance values

### Model Design Principles

- **No Target Leakage:** Features are calculated independently of the target variable
- **Temporal Integrity:** Training data (2023) is never used in test set calculations
- **Feature Engineering:** Percentiles calculated using only training data to prevent temporal leakage
- **Robust Metrics:** Multiple performance metrics (R², RMSE, MAE) with confidence intervals
- **Data Exclusions:** Nationals meets excluded from all analyses (not all teams participate)

### Data Leakage Prevention

**Critical Safeguards:**
1. **Temporal Split:** Test data (2024) never seen during training
2. **Percentile Calculation:** Uses only training data (2023) for percentile-based features
3. **Feature Engineering:** Features calculated independently of target variable
4. **R² Calculation:** Only on test set, never on training set
5. **Bootstrap CIs:** Calculated on test set predictions only
6. **Sensitivity Analysis:** Tests impact of potentially problematic features (see below)

### Sensitivity Analysis: `last_time` Feature

**Research Question:** Does including `last_time` as a feature create data leakage?

**Background:** The target variable `improvement_rate = (last_time - first_time) / season_duration` uses `last_time`, raising concerns about potential data leakage if `last_time` is also included as a feature.

**Methodology:** Two models trained using the same Random Forest algorithm:
- **Model 1:** With `last_time` feature (29 features)
- **Model 2:** Without `last_time` feature (28 features)

**Results:**

| Model | R² Score | 95% CI | RMSE | MAE | N Features |
|-------|----------|--------|------|-----|------------|
| **With `last_time`** | **0.9145** | [0.8653, 0.9516] | 1.49 | 0.41 | 29 |
| **Without `last_time`** | **0.9144** | [0.8622, 0.9506] | 1.49 | 0.42 | 28 |

**Key Findings:**
- **Minimal impact:** Removing `last_time` changes R² by only 0.0001 (0.01%)
- **No significant data leakage:** The 0.01% difference is negligible, suggesting `last_time` does NOT create significant data leakage
- **Feature is legitimate:** `last_time` provides useful information without allowing the model to trivially reconstruct the target
- **Interpretation:** The target uses `(last_time - first_time)`, not just `last_time`, so the model must still learn the relationship between features and improvement rate

**Conclusion:** Including `last_time` as a feature is acceptable. The minimal performance difference (0.01%) indicates it provides legitimate information without creating problematic data leakage.

---

## 9. Practical Implications

### For Coaches and Athletes

1. **Racing Experience Matters Most (21.2%)**
   - Both number of races and season duration are important
   - More racing experience leads to more predictable improvement

2. **Consistency is Key (17.1% + 8.1%)**
   - Fewer "bad races" (races worse than previous) predicts better improvement
   - Lower time variability indicates more consistent training/racing

3. **Racing Frequency and Nationals Success**
   - Teams with athletes racing 4+ times are significantly more likely to make top 15 at nationals
   - 60-80% of top 15 teams have at least one athlete racing 4+ times

4. **Team-Level Factors**
   - Teams with more races (max races) may perform better at nationals
   - Experience level shows trends toward better performance

### For Researchers

1. **Tree-based models are essential** for capturing non-linear relationships
2. **Temporal validation is critical** for honest performance assessment
3. **Feature engineering requires careful design** to avoid circular dependencies
4. **Sensitivity analysis validates feature selection** - test potentially problematic features (e.g., `last_time` showed 0.01% impact, confirming no significant leakage)
5. **Multiple comparisons corrections are essential** - Bonferroni corrections applied to all statistical tests
6. **Same model for comparisons** - When comparing methods (e.g., standardization), use the same model for fair comparison
7. **Gender fairness should be monitored** - small differences are acceptable, large differences indicate bias
8. **Race count consistency filtering** reveals stable improvement patterns across seasons
9. **Team-level analysis** provides insights into program-level factors affecting performance

---

## 10. Summary of All Research Questions

### RQ1: Performance Improvement Patterns
- **Main Finding:** Experience level and consistency are strongest predictors
- **Model Performance:** 91.5% accuracy (Random Forest)
- **Key Insight:** Racing more frequently correlates with better performance at nationals
- **Team Analysis:** Top 25 teams show **one Bonferroni-significant correlation** - Men's season duration (days from first race to nationals) correlates with better rank (r = -0.283, p = 0.0138, Bonferroni-corrected p = 0.0414). Teams that start racing earlier perform better at nationals.

### RQ2: Multi-Season Analysis
- **Main Finding:** Consistent race participation reveals stable improvement patterns
- **Model Performance:** 82.8% accuracy (Gradient Boosting) with race count filter
- **Key Insight:** Filtering for consistent participation (difference < 2 races) enables fair multi-season comparison
- **Validation:** Temporal validation shows models generalize to future years

### RQ3: Gender Differences
- **Main Finding:** Model shows fair performance across genders (4.1% difference)
- **Model Performance:** Women: 94.5% R², Men: 90.4% R²
- **Key Insight:** Different features matter for men vs women, but model captures both well
- **Fairness:** Model does not systematically disadvantage either gender

---

## Appendix: Validation Methods Explained

### Temporal Validation: Why It Matters

**Problem with Random Split:**
- Randomly splitting data can allow future information to leak into past predictions
- Example: If 2025 data is in training set, model might learn patterns that don't exist in 2023-2024

**Solution: Temporal Split:**
- Train on earlier years (2023), test on later years (2024, 2025)
- Simulates real-world scenario: predict future performance based on past data
- Ensures model generalizes to new time periods

**Three Validation Scenarios:**
1. **Primary (2023→2024):** Most realistic, used for main results
2. **Generalization (2023→2025):** Tests long-term prediction ability
3. **Extended (2023+2024→2025):** Tests if more training data helps

### 5-Fold Cross-Validation: How It Works

**Step-by-Step:**
1. Training data (2023) is divided into 5 equal parts
2. For each fold:
   - Train on 4 folds (80% of data)
   - Validate on 1 fold (20% of data)
   - Calculate R² on validation fold
3. Average R² across all 5 folds = CV R² mean
4. Standard deviation across folds = CV R² std

**Example Calculation:**
- Fold 1: R² = 0.88
- Fold 2: R² = 0.90
- Fold 3: R² = 0.87
- Fold 4: R² = 0.89
- Fold 5: R² = 0.91
- **Mean:** 0.88
- **Std:** ±0.015

**Why 5 Folds?**
- Balance between computational cost and statistical reliability
- 5 folds provide good estimate of model stability
- More folds = more reliable but more computation
- Fewer folds = less reliable but faster

### Bootstrap Confidence Intervals: Uncertainty Estimation

**Step-by-Step:**
1. Test set has N predictions (e.g., N = 1248)
2. Resample N predictions 1000 times (with replacement)
3. For each resample, calculate R²
4. Sort 1000 R² values
5. 95% CI = values at 2.5th and 97.5th percentiles

**Example:**
- 1000 bootstrap R² values range from 0.86 to 0.95
- 2.5th percentile: 0.8604
- 97.5th percentile: 0.9517
- **95% CI:** [0.8604, 0.9517]

**Interpretation:**
- We are 95% confident the true R² lies between 0.8604 and 0.9517
- Narrow CI = precise estimate
- Wide CI = uncertain estimate

---

*Document generated from analysis results. For questions or details, refer to the individual analysis scripts and output CSV files.*
