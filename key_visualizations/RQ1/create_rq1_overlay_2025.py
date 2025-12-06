"""
Script to create RQ1 overlay plots for 2025 data.

This script recreates the overlay plots showing:
1. First race time (in minutes) vs time difference (first race - fastest other race)
2. Grouped by number of races per season (2, 3, 4)
3. Both standardized and non-standardized versions

Based on the existing overlay_2024 plots in key_visualizations/RQ1/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import utils - adjust path as needed
# Change to scripts directory to match utils.py expectations
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)
from utils import convert_exclude_nationals, standardize_convert_exclude_nationals_df

# Set data directory path - data is in National_Running_Club_Database/data/data
# From scripts directory, go up two levels to workspace root, then to data/data
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(workspace_root, 'data', 'data')

# Set output directory - create a dedicated directory for 2025 overlay plots
base_dir = os.path.dirname(script_dir)  # Parent of scripts directory
output_dir = os.path.join(base_dir, 'key_visualizations', 'RQ1', 'overlay_2025')
os.makedirs(output_dir, exist_ok=True)

# Set matplotlib style - match the reference plot exactly
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
    """
    Convert time from one distance to another using power law.
    Formula: time_new = time_old * (dist_new / dist_old) ^ power
    Power: 1.08 for women, 1.055 for men (from utils.py)
    """
    if pd.isna(time_seconds) or from_dist == 0 or to_dist == 0:
        return time_seconds
    
    power = 1.08 if gender == 'F' else 1.055
    ratio = to_dist / from_dist
    return time_seconds * (ratio ** power)

def format_time_string(seconds):
    """Format seconds to time string (handle hours if needed)."""
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
    import pandas as pd
    
    results_df = pd.read_csv(os.path.join(data_dir, 'result.csv'))
    meet_df = pd.read_csv(os.path.join(data_dir, 'meet.csv'))
    athlete_df = pd.read_csv(os.path.join(data_dir, 'athlete.csv'))
    running_event_df = pd.read_csv(os.path.join(data_dir, 'running_event.csv'))
    
    # Load sport.csv to identify cross country vs track and field
    sport_df = pd.read_csv(os.path.join(data_dir, 'sport.csv'))
    
    # Get cross country sport ID
    cross_country_sport_id = sport_df[
        sport_df['sport_name'].str.contains('Cross Country', case=False, na=False)
    ]['sport_id'].values
    
    # Also identify track and field sport IDs for verification
    track_field_sport_ids = sport_df[
        sport_df['sport_name'].str.contains('Track', case=False, na=False) |
        sport_df['sport_name'].str.contains('Field', case=False, na=False)
    ]['sport_id'].values
    
    if len(cross_country_sport_id) == 0:
        print("   ERROR: Could not find Cross Country sport ID. Cannot proceed.")
        raise ValueError("Cross Country sport ID not found in sport.csv")
    
    cross_country_sport_id = cross_country_sport_id[0]
    # Filter meets to only cross country (exclude track and field)
    cross_country_meets = meet_df[meet_df['sport_id'] == cross_country_sport_id].copy()
    total_meets = len(meet_df)
    track_field_meets = len(meet_df[meet_df['sport_id'].isin(track_field_sport_ids)]) if len(track_field_sport_ids) > 0 else 0
    
    print(f"   Total meets in database: {total_meets}")
    print(f"   Cross country meets: {len(cross_country_meets)} (sport_id={cross_country_sport_id})")
    if track_field_meets > 0:
        print(f"   Track & Field meets (excluded): {track_field_meets}")
    
    # Filter results to only cross country meets (excludes all track and field results)
    results_before_filter = len(results_df)
    results_df = results_df[results_df['meet_id'].isin(cross_country_meets['meet_id'])].copy()
    results_after_filter = len(results_df)
    
    print(f"   Results before cross-country filter: {results_before_filter}")
    print(f"   Results after cross-country filter: {results_after_filter}")
    print(f"   Track & Field results excluded: {results_before_filter - results_after_filter}")
    
    # Merge with running_event_df and athlete_df to get event names and genders
    results_df = results_df.merge(running_event_df[['running_event_id', 'event_name']], on='running_event_id', how='left')
    results_df = results_df.merge(athlete_df[['athlete_id', 'gender']], on='athlete_id', how='left')
    
    # Filter to only specific distances: 8000m for men, 6000m for women, and 5000m cross country (will convert)
    # Exclude track and field 5000m and other non-cross-country events
    valid_events_men = ['8000m', '5000m']  # Will convert 5000m to 8000m
    valid_events_women = ['6000m', '5000m']  # Will convert 5000m to 6000m
    
    men_results = results_df[
        (results_df['gender'] == 'M') & 
        (results_df['event_name'].isin(valid_events_men))
    ].copy()
    
    women_results = results_df[
        (results_df['gender'] == 'F') & 
        (results_df['event_name'].isin(valid_events_women))
    ].copy()
    
    print(f"   Found {len(men_results)} men's results (8000m and 5000m cross country)")
    print(f"   Found {len(women_results)} women's results (6000m and 5000m cross country)")
    
    # Get event IDs for 8000m and 6000m to update converted results
    event_8000m_id = running_event_df[running_event_df['event_name'] == '8000m']['running_event_id'].values
    event_6000m_id = running_event_df[running_event_df['event_name'] == '6000m']['running_event_id'].values
    
    if len(event_8000m_id) == 0:
        print("   ERROR: Could not find 8000m event in running_event_df")
        event_8000m_id = None
    else:
        event_8000m_id = event_8000m_id[0]
    
    if len(event_6000m_id) == 0:
        print("   ERROR: Could not find 6000m event in running_event_df")
        event_6000m_id = None
    else:
        event_6000m_id = event_6000m_id[0]
    
    # Convert 5000m cross country times to target distances
    # Men: 5000m -> 8000m
    men_5000m = men_results[men_results['event_name'] == '5000m'].copy()
    men_8000m = men_results[men_results['event_name'] == '8000m'].copy()
    
    if len(men_5000m) > 0 and event_8000m_id is not None:
        # Parse times and convert
        from utils import parse_time
        men_5000m['result_time_seconds'] = men_5000m['result_time'].apply(parse_time)
        men_5000m['result_time_seconds'] = men_5000m.apply(
            lambda row: convert_time_distance(row['result_time_seconds'], 5000, 8000, 'M'),
            axis=1
        )
        # Update event name and running_event_id to 8000m after conversion
        men_5000m['event_name'] = '8000m'
        men_5000m['running_event_id'] = event_8000m_id
        # Format back to time string
        men_5000m['result_time'] = men_5000m['result_time_seconds'].apply(format_time_string)
        men_5000m = men_5000m.drop(columns=['result_time_seconds'], errors='ignore')
        print(f"   Converted {len(men_5000m)} men's 5000m cross country results to 8000m")
    
    # Women: 5000m -> 6000m
    women_5000m = women_results[women_results['event_name'] == '5000m'].copy()
    women_6000m = women_results[women_results['event_name'] == '6000m'].copy()
    
    if len(women_5000m) > 0 and event_6000m_id is not None:
        # Parse times and convert
        from utils import parse_time
        women_5000m['result_time_seconds'] = women_5000m['result_time'].apply(parse_time)
        women_5000m['result_time_seconds'] = women_5000m.apply(
            lambda row: convert_time_distance(row['result_time_seconds'], 5000, 6000, 'F'),
            axis=1
        )
        # Update event name and running_event_id to 6000m after conversion
        women_5000m['event_name'] = '6000m'
        women_5000m['running_event_id'] = event_6000m_id
        # Format back to time string
        women_5000m['result_time'] = women_5000m['result_time_seconds'].apply(format_time_string)
        women_5000m = women_5000m.drop(columns=['result_time_seconds'], errors='ignore')
        print(f"   Converted {len(women_5000m)} women's 5000m cross country results to 6000m")
    
    # Combine converted results - handle empty dataframes
    men_list = [men_8000m] if len(men_8000m) > 0 else []
    if len(men_5000m) > 0:
        men_list.append(men_5000m)
    men_final = pd.concat(men_list, ignore_index=True) if men_list else pd.DataFrame()
    
    women_list = [women_6000m] if len(women_6000m) > 0 else []
    if len(women_5000m) > 0:
        women_list.append(women_5000m)
    women_final = pd.concat(women_list, ignore_index=True) if women_list else pd.DataFrame()
    
    # Combine and update results_df
    results_list = []
    if len(men_final) > 0:
        results_list.append(men_final)
    if len(women_final) > 0:
        results_list.append(women_final)
    
    if results_list:
        results_df = pd.concat(results_list, ignore_index=True)
    else:
        results_df = pd.DataFrame()
    
    # Update running_event_df to reflect only the target events
    target_event_names = ['8000m', '6000m']
    running_event_df = running_event_df[
        running_event_df['event_name'].isin(target_event_names)
    ].copy()
    
    print(f"   Final filtered results: {len(results_df)} records")
    print(f"     - Men (8000m): {len(men_final)} records")
    print(f"     - Women (6000m): {len(women_final)} records")
    
    # Try to load course_details if available
    course_details_path = os.path.join(data_dir, 'course_details.csv')
    if os.path.exists(course_details_path):
        course_details_df = pd.read_csv(course_details_path)
    else:
        course_details_df = pd.DataFrame()
    
    return results_df, meet_df, athlete_df, running_event_df, course_details_df

def calculate_first_to_fastest_diff(df):
    """
    Calculate time difference between first race and fastest other race for each athlete.
    Returns DataFrame with: athlete_id, gender, num_races, first_race_minutes, time_diff_seconds
    """
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
        # Positive = improvement (first race was slower than fastest other race)
        # Negative = first race was faster than fastest other race
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

def create_overlay_plot(df_diff, year, mode, output_path):
    """
    Create overlay plot showing:
    - First race time (minutes) on x-axis
    - Time difference (first - fastest other) on y-axis
    - Different lines for different numbers of races (2, 3, 4)
    - Separate plots for men and women
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # Split by gender
    men_df = df_diff[df_diff['gender'] == 'M'].copy()
    women_df = df_diff[df_diff['gender'] == 'F'].copy()
    
    # Calculate IQR values for each gender (from all data, before filtering)
    men_q1 = men_df['first_race_minutes'].quantile(0.25) if len(men_df) > 0 else None
    men_q3 = men_df['first_race_minutes'].quantile(0.75) if len(men_df) > 0 else None
    men_iqr = (men_q3 - men_q1) if (men_q1 is not None and men_q3 is not None) else None
    men_outlier_threshold = (men_q3 + 1.5 * men_iqr) if men_iqr is not None else None
    
    women_q1 = women_df['first_race_minutes'].quantile(0.25) if len(women_df) > 0 else None
    women_q3 = women_df['first_race_minutes'].quantile(0.75) if len(women_df) > 0 else None
    women_iqr = (women_q3 - women_q1) if (women_q1 is not None and women_q3 is not None) else None
    women_outlier_threshold = (women_q3 + 1.5 * women_iqr) if women_iqr is not None else None
    
    # Focus on 2, 3, and 4 races (as shown in the reference plot)
    race_counts = [2, 3, 4]
    
    # Define colors and markers to match the reference plot exactly
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
    
    # Plot for men
    ax1 = axes[0]
    all_men_minutes = []  # Track all minute values that will be plotted
    if len(men_df) > 0:
        for num_races in race_counts:
            race_df = men_df[men_df['num_races'] == num_races].copy()
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
            bin_means = bin_means.sort_values('first_race_minutes')
            bin_means = bin_means[['first_race_minutes', 'time_diff_seconds']]  # Drop count column
            
            if len(bin_means) > 1:
                # Track minute values for x-axis range
                all_men_minutes.extend(bin_means['first_race_minutes'].tolist())
                ax1.plot(bin_means['first_race_minutes'], bin_means['time_diff_seconds'], 
                        marker=marker_map[num_races], color=color_map[num_races], 
                        linewidth=2, markersize=6, linestyle='-',
                        label=f'{num_races} races', alpha=1.0, markeredgecolor='none')
    
    ax1.set_xlabel('Minute Value of First Race', fontsize=10)
    ax1.set_ylabel('Avg Time Diff (s)', fontsize=10)
    ax1.set_axisbelow(True)
    # Set grid lines at every whole minute, but labels every 2 minutes
    if len(all_men_minutes) > 0:
        x_min_full = int(min(all_men_minutes))
        x_max_full = int(max(all_men_minutes)) + 1
        # Set x-axis limits
        ax1.set_xlim(x_min_full - 0.5, x_max_full + 0.5)
        # Grid lines at every whole minute (minor ticks)
        all_grid_lines = list(range(x_min_full, x_max_full + 1))
        ax1.set_xticks(all_grid_lines, minor=True)
        ax1.grid(True, alpha=0.3, linestyle='-', color='lightgray', which='minor')
        # X-axis labels every 2 minutes (major ticks)
        label_positions = list(range((x_min_full // 2) * 2, x_max_full + 2, 2))
        ax1.set_xticks(label_positions)
    else:
        ax1.grid(True, alpha=0.3, linestyle='-', color='lightgray')
    
    # Add IQR visualization for men (after x-axis limits are set)
    if men_q1 is not None and men_q3 is not None:
        x_min, x_max = ax1.get_xlim()
        
        # Add shaded regions (behind everything)
        ax1.axvspan(x_min, men_q1, alpha=0.2, color='lightblue', zorder=0)  # Left of Q1
        ax1.axvspan(men_q1, men_q3, alpha=0.2, color='lightgreen', zorder=0)  # IQR region
        if men_outlier_threshold is not None:
            ax1.axvspan(men_q3, x_max, alpha=0.2, color='lightcoral', zorder=0)  # Right of Q3
        
        # Add vertical lines (on top)
        ax1.axvline(x=men_q1, color='blue', linestyle='--', linewidth=1.5, zorder=10)  # Q1 line
        ax1.axvline(x=men_q3, color='red', linestyle='--', linewidth=1.5, zorder=10)  # Q3 line
        if men_outlier_threshold is not None:
            ax1.axvline(x=men_outlier_threshold, color='black', linestyle='--', linewidth=1.5, zorder=10)  # Outlier threshold
        
        # Update title to include IQR info
        if mode == 'standardized':
            title_text = f'2025: Standardized (Weather & Elevation) - Male\nIQR: Q1 = {int(round(men_q1))} min, Q3 = {int(round(men_q3))} min'
        else:
            title_text = f'2025: Converted Only (8000m) - Male\nIQR: Q1 = {int(round(men_q1))} min, Q3 = {int(round(men_q3))} min'
    else:
        if mode == 'standardized':
            title_text = f'2025: Standardized (Weather & Elevation) - Male'
        else:
            title_text = f'2025: Converted Only (8000m) - Male'
    
    ax1.set_title(title_text, fontsize=11, fontweight='normal')
    ax1.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='black')
    
    # Plot for women
    ax2 = axes[1]
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
            bin_means = bin_means.sort_values('first_race_minutes')
            bin_means = bin_means[['first_race_minutes', 'time_diff_seconds']]  # Drop count column
            
            if len(bin_means) > 1:
                # Track minute values for x-axis range
                all_women_minutes.extend(bin_means['first_race_minutes'].tolist())
                ax2.plot(bin_means['first_race_minutes'], bin_means['time_diff_seconds'], 
                        marker=marker_map[num_races], color=color_map[num_races], 
                        linewidth=2, markersize=6, linestyle='-',
                        label=f'{num_races} races', alpha=1.0, markeredgecolor='none')
    
    ax2.set_xlabel('Minute Value of First Race', fontsize=10)
    ax2.set_ylabel('Avg Time Diff (s)', fontsize=10)
    ax2.set_axisbelow(True)
    # Set grid lines at every whole minute, but labels every 2 minutes
    if len(all_women_minutes) > 0:
        x_min_full = int(min(all_women_minutes))
        x_max_full = int(max(all_women_minutes)) + 1
        # Set x-axis limits
        ax2.set_xlim(x_min_full - 0.5, x_max_full + 0.5)
        # Grid lines at every whole minute (minor ticks)
        all_grid_lines = list(range(x_min_full, x_max_full + 1))
        ax2.set_xticks(all_grid_lines, minor=True)
        ax2.grid(True, alpha=0.3, linestyle='-', color='lightgray', which='minor')
        # X-axis labels every 2 minutes (major ticks)
        label_positions = list(range((x_min_full // 2) * 2, x_max_full + 2, 2))
        ax2.set_xticks(label_positions)
    else:
        ax2.grid(True, alpha=0.3, linestyle='-', color='lightgray')
    
    # Add IQR visualization for women (after x-axis limits are set)
    if women_q1 is not None and women_q3 is not None:
        x_min, x_max = ax2.get_xlim()
        
        # Add shaded regions (behind everything)
        ax2.axvspan(x_min, women_q1, alpha=0.2, color='lightblue', zorder=0)  # Left of Q1
        ax2.axvspan(women_q1, women_q3, alpha=0.2, color='lightgreen', zorder=0)  # IQR region
        if women_outlier_threshold is not None:
            ax2.axvspan(women_q3, x_max, alpha=0.2, color='lightcoral', zorder=0)  # Right of Q3
        
        # Add vertical lines (on top)
        ax2.axvline(x=women_q1, color='blue', linestyle='--', linewidth=1.5, zorder=10)  # Q1 line
        ax2.axvline(x=women_q3, color='red', linestyle='--', linewidth=1.5, zorder=10)  # Q3 line
        if women_outlier_threshold is not None:
            ax2.axvline(x=women_outlier_threshold, color='black', linestyle='--', linewidth=1.5, zorder=10)  # Outlier threshold
        
        # Update title to include IQR info
        if mode == 'standardized':
            title_text = f'2025: Standardized (Weather & Elevation) - Female\nIQR: Q1 = {int(round(women_q1))} min, Q3 = {int(round(women_q3))} min'
        else:
            title_text = f'2025: Converted Only (6000m) - Female\nIQR: Q1 = {int(round(women_q1))} min, Q3 = {int(round(women_q3))} min'
    else:
        if mode == 'standardized':
            title_text = f'2025: Standardized (Weather & Elevation) - Female'
        else:
            title_text = f'2025: Converted Only (6000m) - Female'
    
    ax2.set_title(title_text, fontsize=11, fontweight='normal')
    ax2.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='black')
    
    # Overall title - match exact format from reference
    if mode == 'non-standardized':
        mode_label = 'Non-standardized'
    else:
        mode_label = 'Standardized'
    fig.suptitle(f'Overlay: Avg (First Race - Fastest Other Race) vs. First Race Minute {mode_label} ({year})', 
                 fontsize=12, fontweight='normal', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved overlay plot to: {output_path}')
    plt.close()

def main():
    """Main function to create overlay plots for 2025."""
    print("="*60)
    print("CREATING RQ1 OVERLAY PLOTS FOR 2025")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Date range: August 25, 2025 to November 27, 2025")
    print("="*60)
    
    year = 2025
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found at: {data_dir}")
        print("Please verify the data path.")
        return
    
    # Load data files from custom path
    print("\n1. Loading data files from custom path...")
    try:
        results_df, meet_df, athlete_df, running_event_df, course_details_df = load_data_from_custom_path(data_dir)
        print(f"   Loaded data files successfully")
        print(f"   - Results: {len(results_df)} records")
        print(f"   - Meets: {len(meet_df)} records")
        print(f"   - Athletes: {len(athlete_df)} records")
    except Exception as e:
        print(f"   ERROR loading data files: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load data - non-standardized (converted only)
    print("\n2. Processing non-standardized data (converted only)...")
    try:
        # Change to data directory temporarily for convert_exclude_nationals
        original_dir = os.getcwd()
        data_parent = os.path.dirname(data_dir)
        try:
            os.chdir(data_parent)  # Change to parent of data/data so 'data' path works
            df_conv = convert_exclude_nationals(
                results_df=results_df,
                meet_df=meet_df,
                athlete_df=athlete_df,
                running_event_df=running_event_df
            )
        finally:
            os.chdir(original_dir)
        df_conv['start_date'] = pd.to_datetime(df_conv['start_date'], errors='coerce')
        df_conv_year = filter_year_data(df_conv, year)
        print(f"   Found {len(df_conv_year)} records for {year} (August 25 - November 27)")
        
        if len(df_conv_year) == 0:
            print(f"   WARNING: No data found for {year} in the specified date range!")
            print(f"   Date range: August 25, 2025 to November 27, 2025")
            return
        
        # Calculate first to fastest difference
        print("3. Calculating first race vs fastest other race differences (non-standardized)...")
        df_diff_conv = calculate_first_to_fastest_diff(df_conv_year)
        print(f"   Processed {len(df_diff_conv)} athletes")
        
        if len(df_diff_conv) > 0:
            # Create overlay plot
            output_path_conv = os.path.join(output_dir, f'overlay_{year}_non-standardized.pdf')
            print(f"4. Creating non-standardized overlay plot...")
            create_overlay_plot(df_diff_conv, year, 'non-standardized', output_path_conv)
        else:
            print(f"   WARNING: No athlete data calculated for {year} (non-standardized)")
    
    except Exception as e:
        print(f"   ERROR processing non-standardized data: {e}")
        import traceback
        traceback.print_exc()
    
    # Load data - standardized (with weather/terrain adjustments)
    print("\n5. Processing standardized data...")
    try:
        df_std = standardize_convert_exclude_nationals_df(
            results_df=results_df,
            course_details_df=course_details_df,
            meet_df=meet_df,
            athlete_df=athlete_df,
            running_event_df=running_event_df
        )
        df_std['start_date'] = pd.to_datetime(df_std['start_date'], errors='coerce')
        df_std_year = filter_year_data(df_std, year)
        print(f"   Found {len(df_std_year)} records for {year} (August 25 - November 27)")
        
        if len(df_std_year) > 0:
            # Calculate first to fastest difference
            print("6. Calculating first race vs fastest other race differences (standardized)...")
            df_diff_std = calculate_first_to_fastest_diff(df_std_year)
            print(f"   Processed {len(df_diff_std)} athletes")
            
            if len(df_diff_std) > 0:
                # Create overlay plot
                output_path_std = os.path.join(output_dir, f'overlay_{year}_standardized.pdf')
                print(f"7. Creating standardized overlay plot...")
                create_overlay_plot(df_diff_std, year, 'standardized', output_path_std)
            else:
                print(f"   WARNING: No athlete data calculated for {year} (standardized)")
        else:
            print(f"   WARNING: No standardized data found for {year} in the specified date range")
    
    except Exception as e:
        print(f"   ERROR processing standardized data: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("OVERLAY PLOT CREATION COMPLETE")
    print("="*60)
    print(f"\nPlots saved to: {output_dir}/")
    print(f"  - overlay_{year}_non-standardized.pdf")
    print(f"  - overlay_{year}_standardized.pdf")
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")
    print(f"\nDate range used: August 25, 2025 to November 27, 2025")

if __name__ == '__main__':
    main()

