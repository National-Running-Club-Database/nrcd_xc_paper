"""
Top 25 Teams at Nationals Analysis

This script analyzes the top 25 teams at nationals each year (2023-2025) for men and women.
For each team, it calculates:
- Season duration (days from first race to nationals)
- Max races (maximum races by any athlete on the team)
- Experience level (maximum num_races * season_duration by any athlete)

Then performs correlation analysis to identify relationships.

Run from main directory: python scripts/top25_team_analysis.py
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
import csv
from scipy.stats import pearsonr, spearmanr
from utils import standardize_convert_exclude_nationals_df

output_dir = 'output/top25_teams'
# Directory creation moved to main() to avoid creating when imported

# ============================================================================
# TOP 25 TEAMS AT NATIONALS - LOAD FROM CSV OR DEFINE HERE
# ============================================================================

# Option 1: Load from CSV file (recommended)
# Create/edit data/top25_teams.csv with columns: year, gender, rank, team_name
# Or use the template: data/top25_teams_template.csv

# Option 2: Define directly in code (see below)

TOP_25_TEAMS = {
    2023: {
        'M': [],  # User will provide: ["Team 1", "Team 2", ..., "Team 25"]
        'F': []   # User will provide: ["Team 1", "Team 2", ..., "Team 25"]
    },
    2024: {
        'M': [],  # User will provide
        'F': []   # User will provide
    },
    2025: {
        'M': [],  # User will provide
        'F': []   # User will provide
    }
}

def load_teams_from_csv(csv_path='data/top25_teams.csv'):
    """Load top 25 teams from CSV file"""
    if not os.path.exists(csv_path):
        return None
    
    try:
        # Read CSV with proper handling of commas in team names
        teams_df = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL, skipinitialspace=True, on_bad_lines='skip')
        teams_dict = {
            2023: {'M': [], 'F': []},
            2024: {'M': [], 'F': []},
            2025: {'M': [], 'F': []}
        }
        
        for _, row in teams_df.iterrows():
            year = int(row['year'])
            gender = 'M' if row['gender'] == 'M' else 'F'
            rank = int(row['rank'])
            team_name = str(row['team_name']).strip()
            
            if team_name and team_name != 'nan':
                # Insert at correct position (rank - 1)
                teams_dict[year][gender].insert(rank - 1, team_name)
        
        return teams_dict
    except Exception as e:
        print(f"Error loading teams from CSV: {e}")
        print(f"  Make sure team names with commas are properly quoted in the CSV file")
        return None

def get_team_id_from_name(team_name, team_df):
    """Get team_id from team name (handles partial matches)"""
    matches = team_df[team_df['team_name'].str.contains(team_name, case=False, na=False)]
    if not matches.empty:
        return matches['team_id'].iloc[0]
    return None

def get_nationals_date(year):
    """Get the nationals date for a given year."""
    nationals_dates = {
        2023: pd.Timestamp(year=2023, month=11, day=11),
        2024: pd.Timestamp(year=2024, month=11, day=9),
        2025: pd.Timestamp(year=2025, month=11, day=8)
    }
    return nationals_dates.get(year)

def calculate_team_metrics(team_id, year, gender, df, athlete_team_df):
    """
    Calculate metrics for a team in a given year:
    - first_race_date: Earliest race date after excluding earliest outlier (robust min)
    - season_duration: Days from first race to nationals
    - max_races: Maximum number of races after excluding min race count (robust max)
    - experience_level: Maximum experience level after excluding min value (robust max)
    
    Filter: Requires at least 3 athletes of the given gender.
    For each metric, only the minimum value is excluded before calculating the maximum.
    Example: [1, 2, 7, 14, 20, 30] -> exclude 1 -> use [2, 7, 14, 20, 30] -> max = 30
    """
    # Get nationals date for this year
    nationals_date = get_nationals_date(year)
    if nationals_date is None:
        return None
    
    # Filter data for this year and gender
    start = pd.Timestamp(year=year, month=8, day=1)
    end = pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59)
    
    year_df = df[(df['start_date'] >= start) & (df['start_date'] <= end)].copy()
    year_df = year_df[year_df['gender'] == gender]
    
    # Get athletes on this team for this year
    team_athletes = athlete_team_df[
        (athlete_team_df['team_id'] == team_id) & 
        (athlete_team_df['gender'] == gender)
    ]['athlete_id'].unique()
    
    if len(team_athletes) == 0:
        return None
    
    # Filter races for these athletes
    team_races = year_df[year_df['athlete_id'].isin(team_athletes)].copy()
    
    if len(team_races) == 0:
        return None
    
    # Calculate metrics per athlete
    athlete_metrics = []
    for athlete_id in team_athletes:
        athlete_races = team_races[team_races['athlete_id'] == athlete_id].sort_values('start_date')
        
        if len(athlete_races) == 0:
            continue
        
        num_races = len(athlete_races)
        first_date = athlete_races.iloc[0]['start_date']
        last_date = athlete_races.iloc[-1]['start_date']
        athlete_season_duration = (last_date - first_date).days
        
        # Experience level = num_races * season_duration
        experience_level = num_races * athlete_season_duration if athlete_season_duration > 0 else 0
        
        athlete_metrics.append({
            'athlete_id': athlete_id,
            'num_races': num_races,
            'season_duration': athlete_season_duration,
            'experience_level': experience_level
        })
    
    if len(athlete_metrics) == 0:
        return None
    
    athlete_metrics_df = pd.DataFrame(athlete_metrics)
    
    # Filter: Require at least 3 athletes of the given gender (need at least 3 to exclude min)
    if len(athlete_metrics_df) < 3:
        return None
    
    # Exclude only minimum value for robust statistics
    # For first_race_date: exclude earliest date only, then take min of remaining (earliest team start)
    athlete_start_dates = []
    for athlete_id in team_athletes:
        athlete_races = team_races[team_races['athlete_id'] == athlete_id].sort_values('start_date')
        if len(athlete_races) > 0:
            athlete_start_dates.append(athlete_races.iloc[0]['start_date'])
    
    if len(athlete_start_dates) >= 3:
        # Remove min only, then take min of remaining (earliest team start after excluding outlier)
        athlete_start_dates_sorted = sorted(athlete_start_dates)
        athlete_start_dates_filtered = athlete_start_dates_sorted[1:]  # Exclude first (min) only
        first_race_date = min(athlete_start_dates_filtered)  # Min of remaining dates (earliest team start)
    else:
        first_race_date = min(athlete_start_dates)  # Fallback if not enough values
    
    # Calculate season duration: first race to nationals
    season_duration = (nationals_date - first_race_date).days
    
    # For max_races: exclude min race count only, then take max of remaining
    race_counts = athlete_metrics_df['num_races'].tolist()
    if len(race_counts) >= 3:
        race_counts_sorted = sorted(race_counts)
        race_counts_filtered = race_counts_sorted[1:]  # Exclude min only
        max_num_races = max(race_counts_filtered)  # Max of remaining
    else:
        max_num_races = max(race_counts)  # Fallback if not enough values
    
    # For experience_level: exclude min only, then take max of remaining
    experience_levels = athlete_metrics_df['experience_level'].tolist()
    if len(experience_levels) >= 3:
        experience_levels_sorted = sorted(experience_levels)
        experience_levels_filtered = experience_levels_sorted[1:]  # Exclude min only
        max_experience_level = max(experience_levels_filtered)  # Max of remaining
    else:
        max_experience_level = max(experience_levels)  # Fallback if not enough values
    
    avg_season_duration = athlete_metrics_df['season_duration'].mean()
    
    return {
        'team_id': team_id,
        'first_race_date': first_race_date,
        'max_races': int(max_num_races),  # Integer, not decimal
        'experience_level': max_experience_level,
        'season_duration': season_duration,  # Days from first race to nationals
        'avg_athlete_season_duration': avg_season_duration,
        'num_athletes': len(athlete_metrics),
        'total_races': athlete_metrics_df['num_races'].sum()
    }

def analyze_top25_teams():
    """Analyze top 25 teams for each year and gender"""
    
    # Load data
    print("Loading data...")
    df = standardize_convert_exclude_nationals_df()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Load team and athlete-team association data
    athlete_team_df = pd.read_csv('data/athlete_team_association.csv')
    team_df = pd.read_csv('data/team.csv')
    athlete_df = pd.read_csv('data/athlete.csv')
    
    # Merge athlete gender information
    athlete_df = athlete_df[['athlete_id', 'gender']]
    athlete_team_df = athlete_team_df.merge(athlete_df, on='athlete_id', how='left')
    
    # Filter out missing data
    df = df.dropna(subset=['athlete_id', 'start_date', 'gender'])
    athlete_team_df = athlete_team_df.dropna(subset=['athlete_id', 'team_id', 'gender'])
    
    all_results = []
    
    for year in [2023, 2024, 2025]:
        for gender in ['M', 'F']:
            gender_label = 'Men' if gender == 'M' else 'Women'
            teams_list = TOP_25_TEAMS[year][gender]
            
            if len(teams_list) == 0:
                print(f"\n⚠️  Skipping {year} {gender_label}: No teams provided")
                continue
            
            print(f"\n{'='*60}")
            print(f"Analyzing {year} {gender_label} - Top {len(teams_list)} Teams")
            print(f"{'='*60}")
            
            team_results = []
            
            for rank, team_name in enumerate(teams_list, 1):
                # Get team_id
                team_id = get_team_id_from_name(team_name, team_df)
                
                if team_id is None:
                    print(f"  ⚠️  Rank {rank}: '{team_name}' - Team not found in database")
                    continue
                
                # Calculate metrics
                metrics = calculate_team_metrics(team_id, year, gender, df, athlete_team_df)
                
                if metrics is None:
                    print(f"  ⚠️  Rank {rank}: '{team_name}' - No data found")
                    continue
                
                # Get actual team name from database
                actual_team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
                
                result = {
                    'year': year,
                    'gender': gender_label,
                    'rank': rank,
                    'team_name': actual_team_name,
                    'team_id': team_id,
                    'first_race_date': metrics['first_race_date'],
                    'season_duration': int(metrics['season_duration']),  # Days from first race to nationals
                    'max_races': metrics['max_races'],  # Integer, not decimal
                    'experience_level': round(metrics['experience_level'], 2),
                    'avg_athlete_season_duration': round(metrics['avg_athlete_season_duration'], 2),
                    'num_athletes': metrics['num_athletes'],
                    'total_races': metrics['total_races']
                }
                
                team_results.append(result)
                all_results.append(result)
                
                print(f"  Rank {rank:2d}: {actual_team_name[:50]:50s} | "
                      f"First Race: {metrics['first_race_date'].strftime('%Y-%m-%d')} | "
                      f"Season Duration: {metrics['season_duration']} days | "
                      f"Max Races: {metrics['max_races']} | "
                      f"Exp: {metrics['experience_level']:.0f}")
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\n❌ No data found. Please provide team lists in TOP_25_TEAMS dictionary.")
        return None
    
    # Save to CSV
    csv_path = f'{output_dir}/top25_teams_metrics.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved results to {csv_path}")
    
    return results_df

def perform_correlation_analysis(results_df):
    """Perform correlation analysis on team metrics"""
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Store all correlation results for comprehensive CSV
    all_correlation_results = []
    
    # Separate by gender
    for gender in ['Men', 'Women']:
        gender_df = results_df[results_df['gender'] == gender].copy()
        
        if len(gender_df) == 0:
            continue
        
        print(f"\n{gender}:")
        print("-" * 60)
        
        # Calculate correlations - ONLY with Nationals Rank
        # Exclude correlations between derived metrics (they're mathematically related)
        metrics = ['season_duration', 'max_races', 'experience_level']
        metric_labels = {
            'season_duration': 'Season Duration (days)',
            'max_races': 'Max Races (any athlete)',
            'experience_level': 'Experience Level'
        }
        
        correlation_results = []
        
        # Only correlate each metric with rank
        for metric in metrics:
            rank_data = gender_df['rank'].dropna()
            metric_data = gender_df[metric].dropna()
            
            # Get common indices
            common_idx = rank_data.index.intersection(metric_data.index)
            if len(common_idx) < 3:  # Need at least 3 points
                continue
            
            rank_clean = gender_df.loc[common_idx, 'rank']
            metric_clean = gender_df.loc[common_idx, metric]
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(rank_clean, metric_clean)
            
            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = spearmanr(rank_clean, metric_clean)
            
            result_dict = {
                'metric': metric_labels[metric],
                'pearson_r': round(pearson_r, 3),
                'pearson_p': round(pearson_p, 4),
                'spearman_r': round(spearman_r, 3),
                'spearman_p': round(spearman_p, 4),
                'n': len(common_idx)
            }
            
            # Note: Bonferroni-corrected p-values will be added after all metrics are collected
            
            correlation_results.append(result_dict)
            
            # Add to comprehensive results with gender
            all_correlation_results.append({
                'gender': gender,
                **result_dict
            })
        
        # Create correlation DataFrame
        corr_df = pd.DataFrame(correlation_results)
        
        if len(corr_df) > 0:
            # Apply Bonferroni correction before saving
            n_tests = len(corr_df)
            bonferroni_alpha = 0.05 / n_tests
            corr_df['r_squared'] = corr_df['pearson_r'] ** 2
            corr_df['pearson_p_bonferroni'] = (corr_df['pearson_p'] * n_tests).clip(upper=1.0).round(4)
            corr_df['spearman_p_bonferroni'] = (corr_df['spearman_p'] * n_tests).clip(upper=1.0).round(4)
            corr_df['bonferroni_alpha'] = bonferroni_alpha
            corr_df['bonferroni_sig'] = corr_df['pearson_p'] < bonferroni_alpha
            
            # Reorder columns for better readability
            corr_df = corr_df[['metric', 'pearson_r', 'pearson_p', 'pearson_p_bonferroni', 
                              'r_squared', 'spearman_r', 'spearman_p', 'spearman_p_bonferroni', 
                              'bonferroni_alpha', 'bonferroni_sig', 'n']]
            
            # Save correlation results
            corr_path = f'{output_dir}/correlations_{gender.lower()}.csv'
            corr_df.to_csv(corr_path, index=False)
            print(f"  Saved correlations with nationals rank to {corr_path}")
            
            # Print Bonferroni correction info
            print(f"\n  Bonferroni correction: α = 0.05 / {n_tests} = {bonferroni_alpha:.4f}")
            print(f"  (Significant if p < {bonferroni_alpha:.4f} after correction)")
            
            # Print significant correlations (uncorrected)
            print(f"\n  Correlations with Nationals Rank (uncorrected p < 0.05):")
            significant = corr_df[
                (corr_df['pearson_p'] < 0.05) | (corr_df['spearman_p'] < 0.05)
            ]
            
            if len(significant) > 0:
                for _, row in significant.iterrows():
                    sig_type = "Pearson" if row['pearson_p'] < 0.05 else "Spearman"
                    r_val = row['pearson_r'] if row['pearson_p'] < 0.05 else row['spearman_r']
                    p_val = row['pearson_p'] if row['pearson_p'] < 0.05 else row['spearman_p']
                    r2_val = row['r_squared']
                    bonf_status = "*** Bonferroni significant" if row['bonferroni_sig'] else "(not Bonferroni significant)"
                    direction = "worse rank" if r_val > 0 else "better rank"
                    print(f"    {row['metric']}:")
                    print(f"      {sig_type} r = {r_val:.3f}, R² = {r2_val:.3f}, p = {p_val:.4f} {bonf_status}")
                    print(f"      ({direction}, n={row['n']})")
            else:
                print("    No significant correlations with nationals rank found")
            
            # Print Bonferroni-corrected significant
            bonf_sig = corr_df[corr_df['bonferroni_sig']]
            if len(bonf_sig) > 0:
                print(f"\n  Bonferroni-corrected significant (p < {bonferroni_alpha:.4f}):")
                for _, row in bonf_sig.iterrows():
                    print(f"    {row['metric']}: r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.4f}")
            else:
                print(f"\n  No correlations remain significant after Bonferroni correction")
            
            # Print non-significant for completeness
            non_sig = corr_df[
                (corr_df['pearson_p'] >= 0.05) & (corr_df['spearman_p'] >= 0.05)
            ]
            if len(non_sig) > 0:
                print(f"\n  Non-significant correlations:")
                for _, row in non_sig.iterrows():
                    print(f"    {row['metric']}: r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.4f}")
        
        # Create correlation heatmap
        create_correlation_heatmap(gender_df, gender, metrics, metric_labels)
    
    # Save comprehensive correlations CSV (combines men and women)
    if len(all_correlation_results) > 0:
        comprehensive_corr_df = pd.DataFrame(all_correlation_results)
        # Add R² and Bonferroni-corrected p-values
        comprehensive_corr_df['r_squared'] = comprehensive_corr_df['pearson_r'] ** 2
        # Calculate Bonferroni alpha per gender (3 tests per gender)
        n_tests = 3
        comprehensive_corr_df['bonferroni_alpha'] = 0.05 / n_tests
        # Bonferroni-corrected p-values: multiply by number of tests, cap at 1.0
        comprehensive_corr_df['pearson_p_bonferroni'] = (comprehensive_corr_df['pearson_p'] * n_tests).clip(upper=1.0).round(4)
        comprehensive_corr_df['spearman_p_bonferroni'] = (comprehensive_corr_df['spearman_p'] * n_tests).clip(upper=1.0).round(4)
        comprehensive_corr_df['bonferroni_sig'] = comprehensive_corr_df['pearson_p'] < comprehensive_corr_df['bonferroni_alpha']
        
        # Reorder columns for better readability
        comprehensive_corr_df = comprehensive_corr_df[['gender', 'metric', 'pearson_r', 'pearson_p', 'pearson_p_bonferroni', 
                                                     'r_squared', 'spearman_r', 'spearman_p', 'spearman_p_bonferroni', 
                                                     'bonferroni_alpha', 'bonferroni_sig', 'n']]
        
        comprehensive_path = f'{output_dir}/correlations_comprehensive.csv'
        comprehensive_corr_df.to_csv(comprehensive_path, index=False)
        print(f"\n✅ Saved comprehensive correlations to {comprehensive_path}")

def create_correlation_heatmap(gender_df, gender, metrics, metric_labels):
    """Create correlation chart for team metrics vs Nationals Rank with R² values"""
    
    from scipy.stats import pearsonr
    
    # Calculate correlations and R² values
    correlations_data = []
    for metric in metrics:
        # Remove NaN values
        rank_data = gender_df['rank'].dropna()
        metric_data = gender_df[metric].dropna()
        common_idx = rank_data.index.intersection(metric_data.index)
        
        if len(common_idx) < 3:
            continue
        
        rank_clean = gender_df.loc[common_idx, 'rank']
        metric_clean = gender_df.loc[common_idx, metric]
        
        r_val, p_val = pearsonr(rank_clean, metric_clean)
        r_squared = r_val ** 2
        
        correlations_data.append({
            'metric': metric_labels[metric],
            'r': r_val,
            'r_squared': r_squared,
            'p': p_val
        })
    
    if len(correlations_data) == 0:
        return
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = [d['metric'] for d in correlations_data]
    r_values = [d['r'] for d in correlations_data]
    r2_values = [d['r_squared'] for d in correlations_data]
    p_values = [d['p'] for d in correlations_data]
    
    # Apply Bonferroni correction
    n_tests = len(correlations_data)
    bonferroni_alpha = 0.05 / n_tests
    
    colors = ['red' if r > 0 else 'blue' for r in r_values]
    bars = ax.barh(metric_names, r_values, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Correlation with Nationals Rank (r)', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    
    # Add correlation values, R², and significance on bars
    for i, (bar, r_val, r2_val, p_val) in enumerate(zip(bars, r_values, r2_values, p_values)):
        bonf_sig = p_val < bonferroni_alpha
        # Only show significance based on Bonferroni correction (no uncorrected "*")
        sig_marker = "***" if bonf_sig else ""
        label = f'r={r_val:.3f}, R²={r2_val:.3f}, p={p_val:.4f} {sig_marker}'
        ax.text(r_val + (0.02 if r_val > 0 else -0.02), i, label, 
                va='center', ha='left' if r_val > 0 else 'right', fontsize=10, fontweight='bold')
    
    # Add title with Bonferroni info
    title = f'{gender} - Correlation with Nationals Rank\n'
    title += f'(Positive = worse rank, Negative = better rank)\n'
    title += f'Bonferroni α = {bonferroni_alpha:.4f} (n={n_tests} tests)'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    heatmap_path = f'{output_dir}/correlation_heatmap_{gender.lower()}.pdf'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved correlation chart to {heatmap_path}")

def create_summary_plots(results_df):
    """Create summary plots for team metrics with correlation statistics"""
    
    print("\n" + "="*60)
    print("CREATING SUMMARY PLOTS")
    print("="*60)
    
    for gender in ['Men', 'Women']:
        gender_df = results_df[results_df['gender'] == gender].copy()
        
        if len(gender_df) == 0:
            continue
        
        # Calculate Bonferroni correction (3 tests)
        n_tests = 3
        bonferroni_alpha = 0.05 / n_tests
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{gender} - Top 25 Teams Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Rank vs Season Duration (First Race to Nationals)
        ax1 = axes[0, 0]
        for year in [2023, 2024, 2025]:
            year_data = gender_df[gender_df['year'] == year]
            if len(year_data) > 0:
                ax1.scatter(year_data['season_duration'], year_data['rank'],
                           label=f'{year}', alpha=0.6, s=60)
        
        # Calculate correlation stats
        rank_data = gender_df['rank'].dropna()
        days_data = gender_df['season_duration'].dropna()
        common_idx = rank_data.index.intersection(days_data.index)
        if len(common_idx) >= 3:
            r_val, p_val = pearsonr(gender_df.loc[common_idx, 'rank'], 
                                    gender_df.loc[common_idx, 'season_duration'])
            r_squared = r_val ** 2
            bonf_sig = p_val < bonferroni_alpha
            # Only show significance based on Bonferroni correction (no uncorrected "*")
            sig_marker = "***" if bonf_sig else "ns"
            stats_text = f'r = {r_val:.3f}, R² = {r_squared:.3f}\np = {p_val:.4f} {sig_marker}'
            stats_text += f'\n(Bonf. α = {bonferroni_alpha:.4f})'
            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        ax1.set_xlabel('Season Duration (days)', fontsize=11)
        ax1.set_ylabel('Nationals Rank', fontsize=11)
        ax1.set_title('Rank vs Season Duration (First Race to Nationals)', fontsize=12, fontweight='bold')
        ax1.set_ylim(bottom=0.5, top=25.5)  # Start at 1 (best rank), end at 25
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Lower rank (better) at top
        
        # Plot 2: Rank vs Max Races (with overlap detection)
        ax2 = axes[0, 1]
        
        # Detect overlapping points
        overlap_data = gender_df[['max_races', 'rank']].dropna().copy()
        overlap_counts = overlap_data.groupby(['max_races', 'rank']).size().reset_index(name='count')
        # Reset index to preserve original index for merging
        overlap_data_reset = overlap_data.reset_index()
        overlap_data_merged = overlap_data_reset.merge(overlap_counts, on=['max_races', 'rank'], how='left')
        overlap_data_merged = overlap_data_merged.set_index('index')  # Restore original index
        overlap_data = overlap_data_merged
        
        # Separate overlapping and non-overlapping points
        overlapping = overlap_data[overlap_data['count'] > 1]
        non_overlapping = overlap_data[overlap_data['count'] == 1]
        
        # Plot non-overlapping points normally (in order: 2023, 2024, 2025)
        for year in [2023, 2024, 2025]:
            year_data = gender_df[gender_df['year'] == year]
            if len(year_data) > 0:
                year_indices = year_data.index
                year_overlap_data = overlap_data[overlap_data.index.isin(year_indices)]
                year_non_overlap = year_overlap_data[year_overlap_data['count'] == 1]
                if len(year_non_overlap) > 0:
                    ax2.scatter(year_non_overlap['max_races'], year_non_overlap['rank'],
                               label=f'{year}', alpha=0.6, s=60, zorder=1)
        
        # Plot overlapping points with different styles for n=2 vs n=3
        # n=2: circle (more common), n=3: square (less common, all 3 years overlap)
        overlap_n2_plotted = False  # Track if we've added n=2 legend entry
        overlap_n3_plotted = False  # Track if we've added n=3 legend entry
        if len(overlapping) > 0:
            # Group by (max_races, rank) to handle styling per overlap group
            overlap_groups = overlapping.groupby(['max_races', 'rank'])
            
            for (max_r, rank_val), group in overlap_groups:
                count = len(group)
                # Same size as normal points
                point_size = 60
                
                # Different styles for n=2 vs n=3
                if count == 2:
                    # n=2: Use circle marker with red edge (more common)
                    marker_style = 'o'  # circle
                    edge_color = 'red'
                    edge_width = 2
                    label = 'Overlap (n=2)' if not overlap_n2_plotted else ''
                    overlap_n2_plotted = True
                else:  # count == 3 (max since only 3 years)
                    # n=3: Use square marker with red edge (less common, all 3 years)
                    marker_style = 's'  # square
                    edge_color = 'red'
                    edge_width = 2
                    label = 'Overlap (n=3)' if not overlap_n3_plotted else ''
                    overlap_n3_plotted = True
                
                # Plot all points at same location (no jitter - different styles distinguish them)
                for group_idx, row in group.iterrows():
                    # Get year for color
                    if group_idx in gender_df.index:
                        point_year = gender_df.loc[group_idx, 'year']
                    else:
                        point_year = 2023
                    year_colors = {2023: 'C0', 2024: 'C1', 2025: 'C2'}
                    point_color = year_colors.get(point_year, 'gray')
                    
                    # Plot point
                    ax2.scatter(max_r, rank_val,
                               s=point_size, alpha=0.6, c=point_color,
                               marker=marker_style,
                               edgecolors=edge_color, linewidths=edge_width,
                               label=label if group_idx == group.index[0] else '',
                               zorder=2)
                    label = ''  # Only label first point in group
        
        # Calculate correlation stats
        rank_data = gender_df['rank'].dropna()
        races_data = gender_df['max_races'].dropna()
        common_idx = rank_data.index.intersection(races_data.index)
        if len(common_idx) >= 3:
            r_val, p_val = pearsonr(gender_df.loc[common_idx, 'rank'], 
                                    gender_df.loc[common_idx, 'max_races'])
            r_squared = r_val ** 2
            bonf_sig = p_val < bonferroni_alpha
            # Only show significance based on Bonferroni correction (no uncorrected "*")
            sig_marker = "***" if bonf_sig else "ns"
            stats_text = f'r = {r_val:.3f}, R² = {r_squared:.3f}\np = {p_val:.4f} {sig_marker}'
            stats_text += f'\n(Bonf. α = {bonferroni_alpha:.4f})'
            ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel('Max Races (any athlete)', fontsize=11)
        ax2.set_ylabel('Nationals Rank', fontsize=11)
        title = 'Rank vs Max Race Participation'
        if len(overlapping) > 0:
            title += f'\n(Red edges = overlapping points, n={len(overlapping)} overlaps)'
        ax2.set_title(title, fontsize=12, fontweight='bold')
        ax2.set_ylim(bottom=0.5, top=25.5)  # Start at 1 (best rank), end at 25
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # Plot 3: Rank vs Experience Level
        ax3 = axes[1, 0]
        for year in [2023, 2024, 2025]:
            year_data = gender_df[gender_df['year'] == year]
            if len(year_data) > 0:
                ax3.scatter(year_data['experience_level'], year_data['rank'],
                           label=f'{year}', alpha=0.6, s=60)
        
        # Calculate correlation stats
        rank_data = gender_df['rank'].dropna()
        exp_data = gender_df['experience_level'].dropna()
        common_idx = rank_data.index.intersection(exp_data.index)
        if len(common_idx) >= 3:
            r_val, p_val = pearsonr(gender_df.loc[common_idx, 'rank'], 
                                    gender_df.loc[common_idx, 'experience_level'])
            r_squared = r_val ** 2
            bonf_sig = p_val < bonferroni_alpha
            # Only show significance based on Bonferroni correction (no uncorrected "*")
            sig_marker = "***" if bonf_sig else "ns"
            stats_text = f'r = {r_val:.3f}, R² = {r_squared:.3f}\np = {p_val:.4f} {sig_marker}'
            stats_text += f'\n(Bonf. α = {bonferroni_alpha:.4f})'
            ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        ax3.set_xlabel('Experience Level', fontsize=11)
        ax3.set_ylabel('Nationals Rank', fontsize=11)
        ax3.set_title('Rank vs Experience Level', fontsize=12, fontweight='bold')
        ax3.set_ylim(bottom=0.5, top=25.5)  # Start at 1 (best rank), end at 25
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()
        
        # Plot 4: Distribution of metrics by year
        ax4 = axes[1, 1]
        years = sorted(gender_df['year'].unique())
        x_pos = np.arange(len(years))
        width = 0.25
        
        avg_max_races = [gender_df[gender_df['year'] == y]['max_races'].mean() for y in years]
        avg_exp = [gender_df[gender_df['year'] == y]['experience_level'].mean() for y in years]
        avg_days = [gender_df[gender_df['year'] == y]['season_duration'].mean() for y in years]
        
        # Normalize for comparison (scale to 0-1)
        max_races_val = max(avg_max_races) if max(avg_max_races) > 0 else 1
        max_exp = max(avg_exp) if max(avg_exp) > 0 else 1
        max_days = max(avg_days) if max(avg_days) > 0 else 1
        
        ax4.bar(x_pos - width, [r/max_races_val for r in avg_max_races], width, label='Max Races (norm)', alpha=0.7)
        ax4.bar(x_pos, [e/max_exp for e in avg_exp], width, label='Exp Level (norm)', alpha=0.7)
        ax4.bar(x_pos + width, [d/max_days for d in avg_days], width, label='Season Duration (norm)', alpha=0.7)
        
        ax4.set_xlabel('Year', fontsize=11)
        ax4.set_ylabel('Normalized Value', fontsize=11)
        ax4.set_title('Average Metrics by Year (Normalized)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(years)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = f'{output_dir}/summary_plots_{gender.lower()}.pdf'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved summary plots to {plot_path}")

def main():
    """Main function"""
    # Create output directory only when run directly
    os.makedirs(output_dir, exist_ok=True)
    print("="*60)
    print("TOP 25 TEAMS AT NATIONALS ANALYSIS")
    print("="*60)
    print("\nThis script analyzes top 25 teams at nationals each year.")
    print("Teams can be provided via CSV file or defined in code.")
    
    # Try to load from CSV first
    global TOP_25_TEAMS
    csv_teams = load_teams_from_csv('data/top25_teams.csv')
    if csv_teams is not None:
        # Check if CSV has data
        has_data = any(
            len(csv_teams[year][gender]) > 0
            for year in [2023, 2024, 2025]
            for gender in ['M', 'F']
        )
        if has_data:
            print("✅ Loaded teams from data/top25_teams.csv")
            TOP_25_TEAMS = csv_teams
        else:
            print("⚠️  CSV file exists but is empty. Using code definition.")
    else:
        print("ℹ️  No CSV file found. Using teams defined in code.")
        print("   (You can create data/top25_teams.csv using data/top25_teams_template.csv)")
    
    # Check if teams are provided
    all_empty = all(
        len(TOP_25_TEAMS[year][gender]) == 0
        for year in [2023, 2024, 2025]
        for gender in ['M', 'F']
    )
    
    if all_empty:
        print("\n⚠️  WARNING: No teams provided.")
        print("\nOption 1: Create data/top25_teams.csv with columns: year, gender, rank, team_name")
        print("          (Template available at data/top25_teams_template.csv)")
        print("\nOption 2: Edit this script and add teams to TOP_25_TEAMS dictionary:")
        print("          TOP_25_TEAMS = {")
        print("              2023: {")
        print("                  'M': ['Team 1', 'Team 2', ..., 'Team 25'],")
        print("                  'F': ['Team 1', 'Team 2', ..., 'Team 25']")
        print("              },")
        print("              ...")
        print("          }")
        return
    
    # Analyze teams
    results_df = analyze_top25_teams()
    
    if results_df is None or len(results_df) == 0:
        return
    
    # Perform correlation analysis
    perform_correlation_analysis(results_df)
    
    # Create summary plots
    create_summary_plots(results_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}/")
    print("  - top25_teams_metrics.csv: Team metrics for all years/genders")
    print("  - correlations_comprehensive.csv: Correlation analysis (men and women combined)")
    print("  - correlations_men.csv: Correlation analysis for men")
    print("  - correlations_women.csv: Correlation analysis for women")
    print("  - correlation_heatmap_men.pdf: Correlation heatmap for men")
    print("  - correlation_heatmap_women.pdf: Correlation heatmap for women")
    print("  - summary_plots_men.pdf: Summary plots for men")
    print("  - summary_plots_women.pdf: Summary plots for women")

if __name__ == '__main__':
    main()

