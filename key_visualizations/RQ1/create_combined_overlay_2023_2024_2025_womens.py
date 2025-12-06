"""
Script to create combined RQ1 overlay plots for 2023, 2024, and 2025 women's data.

This script creates a 3x2 grid combining:
- Left column: 2023, 2024, 2025 Standardized plots
- Right column: 2023, 2024, 2025 Converted Only plots

All plots show women's data only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import utils - adjust path as needed
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)
from utils import convert_exclude_nationals, standardize_convert_exclude_nationals_df

# Set data directory path
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(workspace_root, 'data', 'data')

# Set output directory
base_dir = os.path.dirname(script_dir)
output_dir = os.path.join(base_dir, 'key_visualizations', 'RQ1', 'combined_overlay_2023_2024_2025')
os.makedirs(output_dir, exist_ok=True)

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

def plot_womens_subplot(ax, women_df, year, mode):
    """
    Plot women's data on a single subplot axis.
    
    Parameters:
    - ax: matplotlib axis object
    - women_df: DataFrame with women's athlete data
    - year: year for title
    - mode: 'standardized' or 'non-standardized'
    """
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
    
    # Calculate IQR values
    women_q1 = women_df['first_race_minutes'].quantile(0.25) if len(women_df) > 0 else None
    women_q3 = women_df['first_race_minutes'].quantile(0.75) if len(women_df) > 0 else None
    women_iqr = (women_q3 - women_q1) if (women_q1 is not None and women_q3 is not None) else None
    women_outlier_threshold = (women_q3 + 1.5 * women_iqr) if women_iqr is not None else None
    
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
            title_text = f'{year}: Standardized (weather & elevation) - Female\nIQR: Q1={int(round(women_q1)):.0f}min, Q3={int(round(women_q3)):.0f}min'
        else:
            title_text = f'{year}: Converted Only (distance) - Female\nIQR: Q1={int(round(women_q1)):.0f}min, Q3={int(round(women_q3)):.0f}min'
    else:
        if mode == 'standardized':
            title_text = f'{year}: Standardized (weather & elevation) - Female'
        else:
            title_text = f'{year}: Converted Only (distance) - Female'
    
    ax.set_title(title_text, fontsize=11, fontweight='normal')
    ax.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='black')

def main():
    """Main function to create combined overlay plots."""
    print("="*60)
    print("CREATING COMBINED RQ1 OVERLAY PLOTS FOR 2023, 2024, 2025 - WOMEN'S DATA")
    print("="*60)
    print(f"Data directory: {data_dir}")
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
    
    # Process data for each year
    all_data = {}  # {year: {'standardized': df_diff_std, 'non-standardized': df_diff_conv}}
    
    for year in years:
        print(f"\n2. Processing {year} data...")
        
        # Non-standardized (converted only)
        print(f"   Processing {year} non-standardized data...")
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
            df_conv_year = filter_year_data(df_conv, year)
            
            if len(df_conv_year) > 0:
                df_diff_conv = calculate_first_to_fastest_diff(df_conv_year)
                df_diff_conv_women = df_diff_conv[df_diff_conv['gender'] == 'F'].copy()
                all_data[year] = {'non-standardized': df_diff_conv_women}
                print(f"      Found {len(df_diff_conv_women)} women athletes")
            else:
                all_data[year] = {'non-standardized': pd.DataFrame()}
                print(f"      No data found for {year} (non-standardized)")
        except Exception as e:
            print(f"      ERROR processing {year} non-standardized data: {e}")
            all_data[year] = {'non-standardized': pd.DataFrame()}
        
        # Standardized
        print(f"   Processing {year} standardized data...")
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
            
            if len(df_std_year) > 0:
                df_diff_std = calculate_first_to_fastest_diff(df_std_year)
                df_diff_std_women = df_diff_std[df_diff_std['gender'] == 'F'].copy()
                if year not in all_data:
                    all_data[year] = {}
                all_data[year]['standardized'] = df_diff_std_women
                print(f"      Found {len(df_diff_std_women)} women athletes")
            else:
                if year not in all_data:
                    all_data[year] = {}
                all_data[year]['standardized'] = pd.DataFrame()
                print(f"      No data found for {year} (standardized)")
        except Exception as e:
            print(f"      ERROR processing {year} standardized data: {e}")
            if year not in all_data:
                all_data[year] = {}
            all_data[year]['standardized'] = pd.DataFrame()
    
    # Create combined plot
    print("\n3. Creating combined overlay plot...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.patch.set_facecolor('white')
    
    # Plot order:
    # Left column: 2023, 2024, 2025 Standardized
    # Right column: 2023, 2024, 2025 Converted Only
    
    for i, year in enumerate(years):
        # Left column - Standardized
        ax_left = axes[i, 0]
        if year in all_data and 'standardized' in all_data[year] and len(all_data[year]['standardized']) > 0:
            plot_womens_subplot(ax_left, all_data[year]['standardized'], year, 'standardized')
        else:
            ax_left.set_title(f'{year}: Standardized (weather & elevation) - Female\nNo data available', fontsize=11)
        
        # Right column - Converted Only
        ax_right = axes[i, 1]
        if year in all_data and 'non-standardized' in all_data[year] and len(all_data[year]['non-standardized']) > 0:
            plot_womens_subplot(ax_right, all_data[year]['non-standardized'], year, 'non-standardized')
        else:
            ax_right.set_title(f'{year}: Converted Only (distance) - Female\nNo data available', fontsize=11)
    
    # Overall title
    fig.suptitle('Combined Overlay Plots: Females - Avg (First Race - Fastest Other Race) vs. First Race Minute 2023 & 2024 & 2025 with IQR Regions and Outlier Boundaries', 
                 fontsize=12, fontweight='normal', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_path = os.path.join(output_dir, 'combined_overlay_2023_2024_2025_womens.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved combined overlay plot to: {output_path}')
    plt.close()
    
    print("\n" + "="*60)
    print("COMBINED OVERLAY PLOT CREATION COMPLETE")
    print("="*60)
    print(f"\nPlot saved to: {output_path}")

if __name__ == '__main__':
    main()

