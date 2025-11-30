"""
RQ2 Distribution Comparison: Compare average times by gender across years.

This script performs the distribution comparison analysis for RQ2:
- Calculates average times by gender across years
- Creates visualizations

All outputs are saved to output/rq2/

Run from main directory: python scripts/distribution_comparison_rq2.py
Or import and call from rq2.py
"""

import os
import sys

# Setup paths for imports
from _setup_paths import setup_paths
setup_paths()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_gender_comparison_plot(comparison_df, output_dir):
    """Create visualization of gender time comparison."""
    print("\nCreating gender comparison visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    years = sorted(comparison_df['year'].unique())
    men_times = []
    women_times = []
    
    for year in years:
        year_data = comparison_df[comparison_df['year'] == year]
        men_data = year_data[year_data['gender'] == 'men']
        women_data = year_data[year_data['gender'] == 'women']
        
        men_times.append(men_data['avg_time_minutes'].iloc[0] if len(men_data) > 0 else np.nan)
        women_times.append(women_data['avg_time_minutes'].iloc[0] if len(women_data) > 0 else np.nan)
    
    ax.plot(years, men_times, marker='o', label='Men', linewidth=2, markersize=8)
    ax.plot(years, women_times, marker='s', label='Women', linewidth=2, markersize=8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Time (minutes)')
    ax.set_title('Average Race Times by Gender Across Years\n(Filtered: Race count difference < 2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gender_time_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_dir}/gender_time_comparison.pdf")

def analyze_gender_time_comparison(df, valid_athlete_ids, output_dir='output/rq2'):
    """
    Compare average times by gender across years (distribution comparison).
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    valid_athlete_ids : set
        Set of athlete IDs that passed the race count filter
    output_dir : str
        Output directory for results
    """
    print("\n" + "="*60)
    print("GENDER TIME COMPARISON (DISTRIBUTION ANALYSIS)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to valid athletes
    df_filtered = df[df['athlete_id'].isin(valid_athlete_ids)].copy()
    df_filtered['year'] = pd.to_datetime(df_filtered['start_date']).dt.year
    
    # Get athlete and gender info (only merge if gender column doesn't exist)
    if 'gender' not in df_filtered.columns:
        athlete_df = pd.read_csv('data/athlete.csv')
        df_filtered = df_filtered.merge(athlete_df[['athlete_id', 'gender']], on='athlete_id', how='left')
    
    # Ensure gender column exists
    if 'gender' not in df_filtered.columns:
        raise ValueError("Gender column not found in dataframe and could not be merged")
    
    # Calculate average times by year and gender
    comparison_rows = []
    
    for year in [2023, 2024, 2025]:
        year_data = df_filtered[df_filtered['year'] == year]
        
        for gender in ['M', 'F']:
            gender_data = year_data[year_data['gender'] == gender]
            
            if len(gender_data) > 0:
                avg_time_minutes = gender_data['standardized_to_target'].mean() / 60
                num_athletes = gender_data['athlete_id'].nunique()
                num_races = len(gender_data)
                
                comparison_rows.append({
                    'year': year,
                    'gender': 'men' if gender == 'M' else 'women',
                    'avg_time_minutes': avg_time_minutes,
                    'num_athletes': num_athletes,
                    'num_races': num_races
                })
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Pivot for easier comparison
    if len(comparison_df) > 0:
        pivot_df = comparison_df.pivot_table(
            index='year',
            columns='gender',
            values=['avg_time_minutes', 'num_athletes', 'num_races'],
            aggfunc='first'
        )
        
        # Flatten column names
        pivot_df.columns = [f'{col[1]}_{col[0]}' for col in pivot_df.columns]
        
        # Calculate difference
        if 'men_avg_time_minutes' in pivot_df.columns and 'women_avg_time_minutes' in pivot_df.columns:
            pivot_df['difference_minutes'] = pivot_df['men_avg_time_minutes'] - pivot_df['women_avg_time_minutes']
        
        # Reorder columns for readability
        cols = ['men_avg_time_minutes', 'women_avg_time_minutes', 'difference_minutes',
                'men_num_athletes', 'women_num_athletes',
                'men_num_races', 'women_num_races']
        pivot_df = pivot_df[[c for c in cols if c in pivot_df.columns]]
        
        # Rename columns for output
        pivot_df.columns = [c.replace('_avg_time_minutes', '_avg_minutes')
                           .replace('_num_athletes', '_num_athletes')
                           .replace('_num_races', '_num_races') for c in pivot_df.columns]
        
        pivot_df = pivot_df.reset_index()
        pivot_df.to_csv(f'{output_dir}/gender_time_comparison.csv', index=False)
        print(f"Saved gender time comparison to {output_dir}/gender_time_comparison.csv")
        
        # Create visualization
        create_gender_comparison_plot(comparison_df, output_dir)
    
    return comparison_df

def main():
    """Standalone execution for testing."""
    print("="*60)
    print("RQ2 DISTRIBUTION COMPARISON - STANDALONE EXECUTION")
    print("="*60)
    print("\nNote: This script is typically called from rq2.py")
    print("For standalone execution, you need to provide filtered data.")
    print("\nRun: python scripts/rq2.py (instead)")

if __name__ == "__main__":
    main()

