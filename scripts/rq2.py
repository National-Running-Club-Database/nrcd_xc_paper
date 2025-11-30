"""
RQ2: Multi-Season Performance Analysis with Race Count Consistency Filter

RESEARCH QUESTION:
How do athletes' performance patterns change across multiple seasons when controlling
for consistent race participation?

METHODOLOGY:
This script analyzes athletes across multiple seasons (2023, 2024, 2025) with
a filter requiring that the difference in number of races between consecutive
seasons is less than 2. This ensures comparable participation patterns across
seasons and controls for potential confounding effects of varying race frequency.

FILTER RATIONALE:
- Controls for participation consistency (athletes with similar race counts)
- Example: 3 races in 2023, 4 in 2024, 5 in 2025 is valid (differences: 1, 1)
- Excludes athletes with large participation changes (e.g., 3→6 races = diff of 3)
- Ensures fair comparison by focusing on athletes with consistent participation

ANALYSES:
1. Distribution Comparison: Gender time trends across years
2. Multi-Season Analysis: Improvement metrics (2023→2024, 2024→2025, 2023→2025)
3. Machine Learning: Predictive models with temporal validation
   - Generalization: Train on 2023, predict 2025
   - Extended: Train on 2023+2024, predict 2025

DATA EXCLUSIONS:
- Nationals meets excluded (not all teams participate, ensures fair comparison)
- Uses standardized times (accounts for course conditions, weather, terrain)

OUTPUT:
All outputs are saved to output/rq2/

Run from main directory: python scripts/rq2.py

Run from main directory: python scripts/rq2.py
"""

import os
import sys

# Setup paths for imports
from _setup_paths import setup_paths
setup_paths()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import functions
from ml_improvement_prediction import load_raw_data
from ml_model_rq2 import run_rq2_ml_analysis
from multi_season_analysis_rq2 import analyze_athletes_all_seasons
from distribution_comparison_rq2 import analyze_gender_time_comparison

def filter_athletes_by_race_count_diff(df, max_diff=2):
    """
    Filter athletes where the difference in number of races between consecutive
    seasons is less than max_diff.
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset with 'athlete_id', 'start_date', and other columns
    max_diff : int, default=2
        Maximum allowed difference in race count between consecutive seasons
    
    Returns:
    --------
    filtered_athlete_ids : set
        Set of athlete_ids that pass the filter
    race_counts_df : DataFrame
        DataFrame with race counts per athlete per year
    """
    print(f"\nFiltering athletes by race count difference < {max_diff}...")
    
    # Ensure dates are datetime
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['year'] = pd.to_datetime(df['start_date']).dt.year
    
    # Count races per athlete per year
    race_counts = df.groupby(['athlete_id', 'year']).size().reset_index(name='race_count')
    
    # Pivot to get race counts for each year as columns
    race_counts_pivot = race_counts.pivot(index='athlete_id', columns='year', values='race_count').fillna(0)
    race_counts_pivot = race_counts_pivot.astype(int)
    
    # Filter athletes based on consecutive season differences
    valid_athletes = set()
    
    for athlete_id in race_counts_pivot.index:
        years = sorted([col for col in race_counts_pivot.columns if col in [2023, 2024, 2025]])
        
        # Need at least 2 years of data
        if len(years) < 2:
            continue
        
        # Check differences between consecutive years
        valid = True
        for i in range(len(years) - 1):
            year1 = years[i]
            year2 = years[i + 1]
            count1 = race_counts_pivot.loc[athlete_id, year1]
            count2 = race_counts_pivot.loc[athlete_id, year2]
            
            # Skip if either year has 0 races (athlete didn't participate)
            if count1 == 0 or count2 == 0:
                valid = False
                break
            
            # Check if difference is less than max_diff
            diff = abs(count2 - count1)
            if diff >= max_diff:
                valid = False
                break
        
        if valid:
            valid_athletes.add(athlete_id)
    
    print(f"  Total athletes before filter: {len(race_counts_pivot)}")
    print(f"  Athletes passing filter: {len(valid_athletes)}")
    print(f"  Filtered out: {len(race_counts_pivot) - len(valid_athletes)}")
    
    # Create summary of race counts for valid athletes
    race_counts_summary = race_counts_pivot.loc[list(valid_athletes)].copy()
    race_counts_summary = race_counts_summary.reset_index()
    
    return valid_athletes, race_counts_summary

# Analysis functions moved to separate files:
# - rq2_multi_season_analysis.py: analyze_athletes_all_seasons, create_improvement_plot, 
#   create_rq2_comprehensive_plots, create_rq2_gender_comparison_plot
# - rq2_distribution_comparison.py: analyze_gender_time_comparison, create_gender_comparison_plot

def main():
    """Main function to run RQ2 analysis with race count difference filter."""
    print("="*60)
    print("RQ2: MULTI-SEASON ANALYSIS WITH RACE COUNT FILTER")
    print("="*60)
    print("\nFilter: Difference in number of races between consecutive seasons < 2")
    print("Example: 3 races in 2023, 4 in 2024, 5 in 2025 is valid (differences: 1, 1)")
    print("\nNOTE: Nationals meets are EXCLUDED from analysis")
    print("      (Not all teams make nationals, so excluding for fair comparison)")
    
    # Load raw data (this automatically excludes nationals via standardize_convert_exclude_nationals_df)
    df = load_raw_data(mode='standardized')
    
    # Filter athletes by race count difference
    valid_athlete_ids, race_counts_summary = filter_athletes_by_race_count_diff(df, max_diff=2)
    
    # Set output directory for RQ2
    output_dir = 'output/rq2'
    os.makedirs(output_dir, exist_ok=True)
    race_counts_summary.to_csv(f'{output_dir}/race_counts_summary.csv', index=False)
    print(f"\nSaved race counts summary to {output_dir}/race_counts_summary.csv")
    
    # Filter data to valid athletes
    df_filtered = df[df['athlete_id'].isin(valid_athlete_ids)].copy()
    print(f"\nFiltered dataset: {len(df_filtered)} records from {len(valid_athlete_ids)} athletes")
    
    # 1. Distribution comparison (gender time comparison)
    print("\n" + "="*60)
    print("PART 1: DISTRIBUTION COMPARISON")
    print("="*60)
    gender_comparison_df = analyze_gender_time_comparison(df_filtered, valid_athlete_ids, output_dir)
    
    # 2. Multi-season athlete analysis
    print("\n" + "="*60)
    print("PART 2: MULTI-SEASON ATHLETE ANALYSIS")
    print("="*60)
    athletes_df, athletes_summary_df = analyze_athletes_all_seasons(df_filtered, valid_athlete_ids, output_dir)
    
    # 3. ML Analysis
    print("\n" + "="*60)
    print("PART 3: MACHINE LEARNING ANALYSIS")
    print("="*60)
    
    # Run ML analysis (moved to separate file for organization)
    results = run_rq2_ml_analysis(df_filtered, valid_athlete_ids, output_dir=output_dir)
    
    # Check if we got valid results
    valid_results = {k: v for k, v in results.items() 
                    if k not in ['_2025_generalization', '_2025_extended'] 
                    and 'y_test' in v and 'y_pred' in v} if results else {}
    
    print("\n" + "="*60)
    print("RQ2 ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - athletes_all_seasons_detailed.csv")
    print("  - athletes_all_seasons_summary.csv")
    print("  - athletes_all_seasons_improvement.pdf")
    print("  - gender_time_comparison.csv")
    print("  - gender_time_comparison.pdf")
    print("  - race_counts_summary.csv")
    if len(valid_results) > 0:
        print("  - multi_season_feature_importance.csv")
        print("  - multi_season_model_performance.csv")
        print("  - multi_season_predictions.pdf")

if __name__ == "__main__":
    main()
