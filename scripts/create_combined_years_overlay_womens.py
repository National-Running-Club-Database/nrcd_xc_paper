"""
Script to create combined RQ1 overlay plots combining all data from 2023, 2024, and 2025.

This script creates two plots (standardized and non-standardized) that combine all women's data
across the three years into single plots showing:
- Average (First Race - Fastest Other Race) vs. First Race Minute
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)
from utils import convert_exclude_nationals, standardize_convert_exclude_nationals_df

# Set data directory path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(workspace_root, 'data')

# Set default output directory (can be overridden by main() parameter)
default_output_dir = os.path.join(workspace_root, 'key_visualizations', 'RQ1', 'combined_years_overlay')

# Set matplotlib style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.color'] = 'lightgray'

def filter_year_data(df, year):
    """Filter data for a specific year (August 25 to November 27)."""
    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    start = pd.Timestamp(year=year, month=8, day=25)
    end = pd.Timestamp(year=year, month=11, day=27, hour=23, minute=59, second=59)
    year_df = df[(df['start_date'] >= start) & (df['start_date'] <= end)].copy()
    return year_df

def convert_time_distance(time_seconds, from_dist, to_dist, gender):
    """Convert time from one distance to another using power law."""
    if pd.isna(time_seconds) or from_dist == 0 or to_dist == 0:
        return time_seconds
    power = 1.08 if gender == 'F' else 1.055
    ratio = to_dist / from_dist
    return time_seconds * (ratio ** power)

def format_time_string(seconds):
    """Format seconds to time string."""
    if pd.isna(seconds):
        return None
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes}:{secs:05.2f}"

def load_data_from_custom_path(data_dir):
    """Load data files from the specified data directory."""
    results_df = pd.read_csv(os.path.join(data_dir, 'result.csv'))
    meet_df = pd.read_csv(os.path.join(data_dir, 'meet.csv'))
    athlete_df = pd.read_csv(os.path.join(data_dir, 'athlete.csv'))
    running_event_df = pd.read_csv(os.path.join(data_dir, 'running_event.csv'))
    sport_df = pd.read_csv(os.path.join(data_dir, 'sport.csv'))
    
    # Get cross country sport ID
    cross_country_sport_id = sport_df[
        sport_df['sport_name'].str.contains('Cross Country', case=False, na=False)
    ]['sport_id'].values[0]
    
    # Filter meets to only cross country
    cross_country_meets = meet_df[meet_df['sport_id'] == cross_country_sport_id].copy()
    
    # Filter results to only cross country meets
    results_df = results_df[results_df['meet_id'].isin(cross_country_meets['meet_id'])].copy()
    
    # Merge with running_event_df and athlete_df
    results_df = results_df.merge(running_event_df[['running_event_id', 'event_name']], on='running_event_id', how='left')
    results_df = results_df.merge(athlete_df[['athlete_id', 'gender']], on='athlete_id', how='left')
    
    # Filter to specific distances - women only
    valid_events_women = ['6000m', '5000m']
    
    women_results = results_df[(results_df['gender'] == 'F') & (results_df['event_name'].isin(valid_events_women))].copy()
    
    # Get event IDs
    event_6000m_id = running_event_df[running_event_df['event_name'] == '6000m']['running_event_id'].values[0]
    
    # Convert 5000m to 6000m for women
    women_5000m = women_results[women_results['event_name'] == '5000m'].copy()
    women_6000m = women_results[women_results['event_name'] == '6000m'].copy()
    
    if len(women_5000m) > 0:
        from utils import parse_time
        women_5000m['result_time_seconds'] = women_5000m['result_time'].apply(parse_time)
        women_5000m['result_time_seconds'] = women_5000m.apply(
            lambda row: convert_time_distance(row['result_time_seconds'], 5000, 6000, 'F'),
            axis=1
        )
        women_5000m['event_name'] = '6000m'
        women_5000m['running_event_id'] = event_6000m_id
        women_5000m['result_time'] = women_5000m['result_time_seconds'].apply(format_time_string)
        women_5000m = women_5000m.drop(columns=['result_time_seconds'], errors='ignore')
    
    # Combine results
    women_list = [women_6000m] if len(women_6000m) > 0 else []
    if len(women_5000m) > 0:
        women_list.append(women_5000m)
    women_final = pd.concat(women_list, ignore_index=True) if women_list else pd.DataFrame()
    
    results_df = women_final if len(women_final) > 0 else pd.DataFrame()
    
    # Update running_event_df
    target_event_names = ['6000m']
    running_event_df = running_event_df[running_event_df['event_name'].isin(target_event_names)].copy()
    
    # Load course_details if available
    course_details_path = os.path.join(data_dir, 'course_details.csv')
    if os.path.exists(course_details_path):
        course_details_df = pd.read_csv(course_details_path)
    else:
        course_details_df = pd.DataFrame()
    
    return results_df, meet_df, athlete_df, running_event_df, course_details_df

def calculate_first_to_fastest_diff(df):
    """Calculate time difference between first race and fastest other race for each athlete."""
    df = df.copy()
    df = df.dropna(subset=['standardized_to_target', 'start_date', 'gender', 'athlete_id'])
    
    athlete_data = []
    athletes = df['athlete_id'].unique()
    
    for athlete_id in athletes:
        athlete_races = df[df['athlete_id'] == athlete_id].sort_values('start_date')
        
        if len(athlete_races) < 2:
            continue
        
        # Get first race time
        first_race = athlete_races.iloc[0]
        first_time = first_race['standardized_to_target']
        
        if pd.isna(first_time):
            continue
        
        # Get fastest OTHER race (excluding first race)
        other_races = athlete_races.iloc[1:]
        other_times = other_races['standardized_to_target'].dropna()
        
        if len(other_times) == 0:
            continue
        
        fastest_other = other_times.min()
        
        # Calculate difference: first_race - fastest_other
        time_diff = first_time - fastest_other
        
        # Convert first race time to minutes
        first_race_minutes = first_time / 60.0
        
        athlete_data.append({
            'athlete_id': athlete_id,
            'gender': first_race['gender'],
            'num_races': len(athlete_races),
            'first_race_minutes': first_race_minutes,
            'time_diff_seconds': time_diff
        })
    
    return pd.DataFrame(athlete_data)

def create_overlay_plot_from_year_averages(year_data_dict, mode, output_path):
    """
    Create overlay plot by averaging the bin values from individual years.
    
    Parameters:
    - year_data_dict: dict with keys 2023, 2024, 2025, each containing a DataFrame
                      with columns: athlete_id, gender, num_races, first_race_minutes, time_diff_seconds
    - mode: 'standardized' or 'non-standardized'
    - output_path: path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    years = [2023, 2024, 2025]
    race_counts = [2, 3, 4]
    
    # Define colors and markers
    color_map = {
        2: '#1f77b4',  # Blue
        3: '#ff7f0e',  # Orange
        4: '#2ca02c'   # Green
    }
    marker_map = {
        2: 'o',  # circles
        3: 's',  # squares
        4: '^'   # triangles
    }
    
    # Process each year separately and collect bin means
    all_year_bins = {}  # {(num_races, minute): {2023: value, 2024: value, 2025: value}}
    all_minutes = set()
    
    # Calculate overall IQR from all years combined for visualization
    all_women_data = []
    for year in years:
        if year in year_data_dict and len(year_data_dict[year]) > 0:
            women_df_year = year_data_dict[year][year_data_dict[year]['gender'] == 'F'].copy()
            all_women_data.append(women_df_year)
    
    if len(all_women_data) > 0:
        all_women_combined = pd.concat(all_women_data, ignore_index=True)
        women_q1 = all_women_combined['first_race_minutes'].quantile(0.25) if len(all_women_combined) > 0 else None
        women_q3 = all_women_combined['first_race_minutes'].quantile(0.75) if len(all_women_combined) > 0 else None
        women_iqr = (women_q3 - women_q1) if (women_q1 is not None and women_q3 is not None) else None
        women_outlier_threshold = (women_q3 + 1.5 * women_iqr) if women_iqr is not None else None
    else:
        women_q1 = women_q3 = women_iqr = women_outlier_threshold = None
    
    # Process each year
    for year in years:
        if year not in year_data_dict or len(year_data_dict[year]) == 0:
            continue
            
        women_df = year_data_dict[year][year_data_dict[year]['gender'] == 'F'].copy()
        
        for num_races in race_counts:
            race_df = women_df[women_df['num_races'] == num_races].copy()
            if len(race_df) == 0:
                continue
            
            # Remove outliers using IQR
            Q1 = race_df['first_race_minutes'].quantile(0.25)
            Q3 = race_df['first_race_minutes'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            race_df_clean = race_df[
                (race_df['first_race_minutes'] >= lower_bound) & 
                (race_df['first_race_minutes'] <= upper_bound)
            ].copy()
            
            if len(race_df_clean) < 3:
                continue
            
            # Truncate (floor) first race minutes
            race_df_clean['first_race_minutes_rounded'] = np.floor(race_df_clean['first_race_minutes']).astype(int)
            
            # Group by whole minute values
            bin_stats = race_df_clean.groupby('first_race_minutes_rounded', observed=True).agg({
                'time_diff_seconds': 'mean',
                'athlete_id': 'count'
            }).reset_index()
            bin_stats.columns = ['first_race_minutes', 'time_diff_seconds', 'athlete_count']
            
            # Filter to only include minute values with at least 5 athletes
            bin_means = bin_stats[bin_stats['athlete_count'] >= 5].copy()
            
            # Store the bin means for this year
            for _, row in bin_means.iterrows():
                key = (num_races, int(row['first_race_minutes']))
                if key not in all_year_bins:
                    all_year_bins[key] = {}
                all_year_bins[key][year] = row['time_diff_seconds']
                all_minutes.add(int(row['first_race_minutes']))
    
    # Now average across years for each (num_races, minute) combination
    all_women_minutes = []
    for num_races in race_counts:
        # Collect all minute values for this num_races
        minute_values = sorted([m for (nr, m) in all_year_bins.keys() if nr == num_races])
        
        averaged_bins = []
        for minute in minute_values:
            key = (num_races, minute)
            if key in all_year_bins:
                year_values = all_year_bins[key]
                # Only include if we have at least 2 out of 3 years
                if len(year_values) >= 2:
                    avg_value = np.mean(list(year_values.values()))
                    averaged_bins.append({
                        'first_race_minutes': minute,
                        'time_diff_seconds': avg_value
                    })
        
        if len(averaged_bins) > 1:
            bin_means_df = pd.DataFrame(averaged_bins)
            bin_means_df = bin_means_df.sort_values('first_race_minutes')
            
            all_women_minutes.extend(bin_means_df['first_race_minutes'].tolist())
            ax.plot(bin_means_df['first_race_minutes'], bin_means_df['time_diff_seconds'], 
                    marker=marker_map[num_races], color=color_map[num_races], 
                    linewidth=2, markersize=6, linestyle='-',
                    label=f'{num_races} races', alpha=1.0, markeredgecolor='none')
    
    ax.set_xlabel('Minute Value of First Race', fontsize=10)
    ax.set_ylabel('Avg Time Diff (s)', fontsize=10)
    ax.set_axisbelow(True)
    
    # Set grid lines at every whole minute, but labels every 2 minutes
    if len(all_women_minutes) > 0:
        x_min_full = int(min(all_women_minutes))
        x_max_full = int(max(all_women_minutes)) + 1
        # Set x-axis limits
        ax.set_xlim(x_min_full - 0.5, x_max_full + 0.5)
        # Grid lines at every whole minute (minor ticks)
        all_grid_lines = list(range(x_min_full, x_max_full + 1))
        ax.set_xticks(all_grid_lines, minor=True)
        ax.grid(True, alpha=0.3, linestyle='-', color='lightgray', which='minor')
        # X-axis labels every 2 minutes (major ticks)
        label_positions = list(range((x_min_full // 2) * 2, x_max_full + 2, 2))
        ax.set_xticks(label_positions)
    else:
        ax.grid(True, alpha=0.3, linestyle='-', color='lightgray')
    
    # Add IQR visualization
    if women_q1 is not None and women_q3 is not None:
        x_min, x_max = ax.get_xlim()
        
        # Add shaded regions (behind everything)
        ax.axvspan(x_min, women_q1, alpha=0.2, color='lightblue', zorder=0)
        ax.axvspan(women_q1, women_q3, alpha=0.2, color='lightgreen', zorder=0)
        if women_outlier_threshold is not None:
            ax.axvspan(women_q3, x_max, alpha=0.2, color='lightcoral', zorder=0)
        
        # Add vertical lines (on top)
        ax.axvline(x=women_q1, color='blue', linestyle='--', linewidth=1.5, zorder=10)
        ax.axvline(x=women_q3, color='red', linestyle='--', linewidth=1.5, zorder=10)
        if women_outlier_threshold is not None:
            ax.axvline(x=women_outlier_threshold, color='black', linestyle='--', linewidth=1.5, zorder=10)
        
        # Update title to include IQR info
        if mode == 'standardized':
            title_text = f'Combined 2023-2025: Standardized (Weather & Elevation) - Female\nIQR: Q1 = {int(round(women_q1))} min, Q3 = {int(round(women_q3))} min'
        else:
            title_text = f'Combined 2023-2025: Converted Only (6000m) - Female\nIQR: Q1 = {int(round(women_q1))} min, Q3 = {int(round(women_q3))} min'
    else:
        if mode == 'standardized':
            title_text = f'Combined 2023-2025: Standardized (Weather & Elevation) - Female'
        else:
            title_text = f'Combined 2023-2025: Converted Only (6000m) - Female'
    
    ax.set_title(title_text, fontsize=11, fontweight='normal')
    ax.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='black')
    
    # Overall title
    if mode == 'standardized':
        mode_label = 'Standardized'
    else:
        mode_label = 'Non-standardized'
    fig.suptitle(f'Overlay: Avg (First Race - Fastest Other Race) vs. First Race Minute {mode_label} (2023-2025 Combined)', 
                 fontsize=12, fontweight='normal', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved overlay plot to: {output_path}')
    plt.close()

def create_overlay_plot(df_diff, mode, output_path):
    """
    Create overlay plot showing:
    - First race time (minutes) on x-axis
    - Time difference (first - fastest other) on y-axis
    - Different lines for different numbers of races (2, 3, 4)
    - Women's data only
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Filter to women only
    women_df = df_diff[df_diff['gender'] == 'F'].copy()
    
    # Calculate IQR values
    women_q1 = women_df['first_race_minutes'].quantile(0.25) if len(women_df) > 0 else None
    women_q3 = women_df['first_race_minutes'].quantile(0.75) if len(women_df) > 0 else None
    women_iqr = (women_q3 - women_q1) if (women_q1 is not None and women_q3 is not None) else None
    women_outlier_threshold = (women_q3 + 1.5 * women_iqr) if women_iqr is not None else None
    
    # Focus on 2, 3, and 4 races
    race_counts = [2, 3, 4]
    
    # Define colors and markers
    color_map = {
        2: '#1f77b4',  # Blue
        3: '#ff7f0e',  # Orange
        4: '#2ca02c'   # Green
    }
    marker_map = {
        2: 'o',  # circles
        3: 's',  # squares
        4: '^'   # triangles
    }
    
    all_women_minutes = []  # Track all minute values that will be plotted
    
    if len(women_df) > 0:
        for num_races in race_counts:
            race_df = women_df[women_df['num_races'] == num_races].copy()
            if len(race_df) == 0:
                continue
            
            # Remove outliers using IQR
            Q1 = race_df['first_race_minutes'].quantile(0.25)
            Q3 = race_df['first_race_minutes'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            race_df_clean = race_df[
                (race_df['first_race_minutes'] >= lower_bound) & 
                (race_df['first_race_minutes'] <= upper_bound)
            ].copy()
            
            if len(race_df_clean) < 3:
                continue
            
            # Truncate (floor) first race minutes to whole numbers for binning
            race_df_clean['first_race_minutes_rounded'] = np.floor(race_df_clean['first_race_minutes']).astype(int)
            
            # Group by whole minute values and count athletes in each minute
            bin_stats = race_df_clean.groupby('first_race_minutes_rounded', observed=True).agg({
                'time_diff_seconds': 'mean',
                'athlete_id': 'count'
            }).reset_index()
            bin_stats.columns = ['first_race_minutes', 'time_diff_seconds', 'athlete_count']
            
            # Filter to only include minute values with at least 5 athletes
            bin_means = bin_stats[bin_stats['athlete_count'] >= 5].copy()
            
            # Additional outlier removal: Remove extreme outliers in time_diff_seconds for each minute bin
            # Calculate IQR for time_diff_seconds within each minute bin
            if len(bin_means) > 0:
                time_diff_q1 = bin_means['time_diff_seconds'].quantile(0.25)
                time_diff_q3 = bin_means['time_diff_seconds'].quantile(0.75)
                time_diff_iqr = time_diff_q3 - time_diff_q1
                time_diff_lower = time_diff_q1 - 3.0 * time_diff_iqr  # More aggressive (3.0 instead of 1.5)
                time_diff_upper = time_diff_q3 + 3.0 * time_diff_iqr
                
                # Filter out extreme outliers, especially in the 33 minute range
                bin_means = bin_means[
                    (bin_means['time_diff_seconds'] >= time_diff_lower) & 
                    (bin_means['time_diff_seconds'] <= time_diff_upper)
                ].copy()
                
                # Specifically remove outlier at 33 minutes if it exists and is extreme
                if 33 in bin_means['first_race_minutes'].values:
                    bin_33 = bin_means[bin_means['first_race_minutes'] == 33]
                    if len(bin_33) > 0:
                        # Check if the 33 minute value is an extreme outlier compared to surrounding values
                        surrounding_bins = bin_means[
                            (bin_means['first_race_minutes'] >= 30) & 
                            (bin_means['first_race_minutes'] <= 36) &
                            (bin_means['first_race_minutes'] != 33)
                        ]
                        if len(surrounding_bins) > 0:
                            surrounding_mean = surrounding_bins['time_diff_seconds'].mean()
                            surrounding_std = surrounding_bins['time_diff_seconds'].std()
                            bin_33_value = bin_33['time_diff_seconds'].iloc[0]
                            # Remove if more than 3 standard deviations away from surrounding values
                            if abs(bin_33_value - surrounding_mean) > 3 * surrounding_std:
                                bin_means = bin_means[bin_means['first_race_minutes'] != 33].copy()
            bin_means = bin_means.sort_values('first_race_minutes')
            bin_means = bin_means[['first_race_minutes', 'time_diff_seconds']]
            
            if len(bin_means) > 1:
                # Track minute values for x-axis range
                all_women_minutes.extend(bin_means['first_race_minutes'].tolist())
                ax.plot(bin_means['first_race_minutes'], bin_means['time_diff_seconds'], 
                        marker=marker_map[num_races], color=color_map[num_races], 
                        linewidth=2, markersize=6, linestyle='-',
                        label=f'{num_races} races', alpha=1.0, markeredgecolor='none')
    
    ax.set_xlabel('Minute Value of First Race', fontsize=10)
    ax.set_ylabel('Avg Time Diff (s)', fontsize=10)
    ax.set_axisbelow(True)
    
    # Set grid lines at every whole minute, but labels every 2 minutes
    if len(all_women_minutes) > 0:
        x_min_full = int(min(all_women_minutes))
        x_max_full = int(max(all_women_minutes)) + 1
        # Set x-axis limits
        ax.set_xlim(x_min_full - 0.5, x_max_full + 0.5)
        # Grid lines at every whole minute (minor ticks)
        all_grid_lines = list(range(x_min_full, x_max_full + 1))
        ax.set_xticks(all_grid_lines, minor=True)
        ax.grid(True, alpha=0.3, linestyle='-', color='lightgray', which='minor')
        # X-axis labels every 2 minutes (major ticks)
        label_positions = list(range((x_min_full // 2) * 2, x_max_full + 2, 2))
        ax.set_xticks(label_positions)
    else:
        ax.grid(True, alpha=0.3, linestyle='-', color='lightgray')
    
    # Add IQR visualization
    if women_q1 is not None and women_q3 is not None:
        x_min, x_max = ax.get_xlim()
        
        # Add shaded regions (behind everything)
        ax.axvspan(x_min, women_q1, alpha=0.2, color='lightblue', zorder=0)
        ax.axvspan(women_q1, women_q3, alpha=0.2, color='lightgreen', zorder=0)
        if women_outlier_threshold is not None:
            ax.axvspan(women_q3, x_max, alpha=0.2, color='lightcoral', zorder=0)
        
        # Add vertical lines (on top)
        ax.axvline(x=women_q1, color='blue', linestyle='--', linewidth=1.5, zorder=10)
        ax.axvline(x=women_q3, color='red', linestyle='--', linewidth=1.5, zorder=10)
        if women_outlier_threshold is not None:
            ax.axvline(x=women_outlier_threshold, color='black', linestyle='--', linewidth=1.5, zorder=10)
        
        # Update title to include IQR info
        if mode == 'standardized':
            title_text = f'Combined 2023-2025: Standardized (Weather & Elevation) - Female\nIQR: Q1 = {int(round(women_q1))} min, Q3 = {int(round(women_q3))} min'
        else:
            title_text = f'Combined 2023-2025: Converted Only (6000m) - Female\nIQR: Q1 = {int(round(women_q1))} min, Q3 = {int(round(women_q3))} min'
    else:
        if mode == 'standardized':
            title_text = f'Combined 2023-2025: Standardized (Weather & Elevation) - Female'
        else:
            title_text = f'Combined 2023-2025: Converted Only (6000m) - Female'
    
    ax.set_title(title_text, fontsize=11, fontweight='normal')
    ax.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='black')
    
    # Overall title
    if mode == 'standardized':
        mode_label = 'Standardized'
    else:
        mode_label = 'Non-standardized'
    fig.suptitle(f'Overlay: Avg (First Race - Fastest Other Race) vs. First Race Minute {mode_label} (2023-2025 Combined)', 
                 fontsize=12, fontweight='normal', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved overlay plot to: {output_path}')
    plt.close()

def main(output_dir=None):
    """Main function to create combined overlay plots.
    
    Parameters:
    - output_dir: Optional output directory. If None, uses default location.
    """
    # Use provided output_dir or default
    if output_dir is None:
        output_dir = default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("CREATING COMBINED RQ1 OVERLAY PLOTS (2023-2025 COMBINED) - WOMEN'S DATA")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Date range: August 25 to November 27 for each year")
    print("="*60)
    
    years = [2023, 2024, 2025]
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found at: {data_dir}")
        return
    
    # Load data files
    print("\n1. Loading data files...")
    try:
        results_df, meet_df, athlete_df, running_event_df, course_details_df = load_data_from_custom_path(data_dir)
        print(f"   Loaded data files successfully")
    except Exception as e:
        print(f"   ERROR loading data files: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process each year separately
    print("\n2. Processing data for each year separately...")
    
    # Non-standardized (converted only) - process each year separately
    print("   Processing non-standardized data for each year...")
    year_data_dict_conv = {}
    try:
        original_dir = os.getcwd()
        data_parent = os.path.dirname(data_dir)
        try:
            os.chdir(data_parent)
            df_conv = convert_exclude_nationals(
                results_df=results_df,
                meet_df=meet_df,
                athlete_df=athlete_df,
                running_event_df=running_event_df
            )
        finally:
            os.chdir(original_dir)
        
        df_conv['start_date'] = pd.to_datetime(df_conv['start_date'], errors='coerce')
        
        for year in years:
            df_conv_year = filter_year_data(df_conv, year)
            if len(df_conv_year) > 0:
                df_diff_conv = calculate_first_to_fastest_diff(df_conv_year)
                year_data_dict_conv[year] = df_diff_conv
                print(f"      {year}: Found {len(df_diff_conv)} athletes")
            else:
                year_data_dict_conv[year] = pd.DataFrame()
                print(f"      {year}: No data found")
    except Exception as e:
        print(f"      ERROR processing non-standardized data: {e}")
        import traceback
        traceback.print_exc()
        for year in years:
            year_data_dict_conv[year] = pd.DataFrame()
    
    # Standardized - process each year separately
    print("   Processing standardized data for each year...")
    year_data_dict_std = {}
    try:
        df_std = standardize_convert_exclude_nationals_df(
            results_df=results_df,
            course_details_df=course_details_df,
            meet_df=meet_df,
            athlete_df=athlete_df,
            running_event_df=running_event_df
        )
        df_std['start_date'] = pd.to_datetime(df_std['start_date'], errors='coerce')
        
        for year in years:
            df_std_year = filter_year_data(df_std, year)
            if len(df_std_year) > 0:
                df_diff_std = calculate_first_to_fastest_diff(df_std_year)
                year_data_dict_std[year] = df_diff_std
                print(f"      {year}: Found {len(df_diff_std)} athletes")
            else:
                year_data_dict_std[year] = pd.DataFrame()
                print(f"      {year}: No data found")
    except Exception as e:
        print(f"      ERROR processing standardized data: {e}")
        import traceback
        traceback.print_exc()
        for year in years:
            year_data_dict_std[year] = pd.DataFrame()
    
    # Create plots by averaging year values
    print("\n3. Creating overlay plots (averaging across years)...")
    
    output_path_conv = os.path.join(output_dir, 'combined_2023_2024_2025_non-standardized_womens.pdf')
    print(f"   Creating non-standardized plot...")
    create_overlay_plot_from_year_averages(year_data_dict_conv, 'non-standardized', output_path_conv)
    
    output_path_std = os.path.join(output_dir, 'combined_2023_2024_2025_standardized_womens.pdf')
    print(f"   Creating standardized plot...")
    create_overlay_plot_from_year_averages(year_data_dict_std, 'standardized', output_path_std)
    
    print("\n" + "="*60)
    print("COMBINED OVERLAY PLOT CREATION COMPLETE")
    print("="*60)
    print(f"\nPlots saved to: {output_dir}/")
    print(f"  - combined_2023_2024_2025_non-standardized_womens.pdf")
    print(f"  - combined_2023_2024_2025_standardized_womens.pdf")

if __name__ == '__main__':
    main()

