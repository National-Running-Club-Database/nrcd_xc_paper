"""
Create combined RQ1 overlay plots for 2023, 2024, and 2025 for a single gender.

Produces a 3x2 grid:
- Left column: 2023, 2024, 2025 Standardized (weather & elevation)
- Right column: 2023, 2024, 2025 Converted Only (distance)

This single, gender-parametrized script replaces the former
create_combined_overlay_2023_2024_2025_mens.py / _womens.py pair, which differed
only by gender filter and title labels. Call ``main(gender='M')`` or
``main(gender='F')``.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from utils import standardize_both_tiers
from load_nrcd_data import get_data_dir

data_dir = get_data_dir()
workspace_root = os.path.dirname(script_dir)
default_output_dir = os.path.join(workspace_root, 'output', 'rq1', 'overlay_plots')

# Gender-specific display / file naming metadata.
GENDER_META = {
    'M': {'label': 'Male', 'plural': 'Males', 'suffix': 'mens'},
    'F': {'label': 'Female', 'plural': 'Females', 'suffix': 'womens'},
}

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
    return df[(df['start_date'] >= start) & (df['start_date'] <= end)].copy()


def load_data_from_custom_path(data_dir, gender_filter=None):
    """Load comprehensive-era Cross Country results (optionally one gender)."""
    from load_nrcd_data import load_analysis_tables
    tables = load_analysis_tables(data_dir, era="comprehensive")
    results_df = tables["result"].copy()
    meet_df = tables["meet"].copy()
    athlete_df = tables["athlete"].copy()
    running_event_df = tables["running_event"].copy()
    course_details_df = tables["course_details"].copy()
    if "gender" not in results_df.columns:
        results_df = results_df.merge(athlete_df[["athlete_id", "gender"]], on="athlete_id", how="left")
    if gender_filter is not None:
        results_df = results_df[results_df["gender"] == gender_filter].copy()
    return results_df, meet_df, athlete_df, running_event_df, course_details_df


def calculate_first_to_fastest_diff(df):
    """Time difference between first race and fastest other race for each athlete."""
    df = df.copy()
    df = df.dropna(subset=['standardized_to_target', 'start_date', 'gender', 'athlete_id'])

    athlete_data = []
    for athlete_id in df['athlete_id'].unique():
        athlete_races = df[df['athlete_id'] == athlete_id].sort_values('start_date')
        if len(athlete_races) < 2:
            continue
        first_race = athlete_races.iloc[0]
        first_time = first_race['standardized_to_target']
        if pd.isna(first_time):
            continue
        other_times = athlete_races.iloc[1:]['standardized_to_target'].dropna()
        if len(other_times) == 0:
            continue
        athlete_data.append({
            'athlete_id': athlete_id,
            'gender': first_race['gender'],
            'num_races': len(athlete_races),
            'first_race_minutes': first_time / 60.0,
            'time_diff_seconds': first_time - other_times.min(),
        })
    return pd.DataFrame(athlete_data)


def plot_gender_subplot(ax, gender_df, year, mode, gender_label):
    """Plot one gender's data on a single subplot axis."""
    race_counts = [2, 3, 4]
    color_map = {2: '#1f77b4', 3: '#ff7f0e', 4: '#2ca02c'}
    marker_map = {2: 'o', 3: 's', 4: '^'}

    q1 = gender_df['first_race_minutes'].quantile(0.25) if len(gender_df) > 0 else None
    q3 = gender_df['first_race_minutes'].quantile(0.75) if len(gender_df) > 0 else None
    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
    outlier_threshold = (q3 + 1.5 * iqr) if iqr is not None else None

    all_minutes = []
    if len(gender_df) > 0:
        for num_races in race_counts:
            race_df = gender_df[gender_df['num_races'] == num_races].copy()
            if len(race_df) == 0:
                continue
            Q1 = race_df['first_race_minutes'].quantile(0.25)
            Q3 = race_df['first_race_minutes'].quantile(0.75)
            IQR = Q3 - Q1
            race_df_clean = race_df[
                (race_df['first_race_minutes'] >= Q1 - 1.5 * IQR) &
                (race_df['first_race_minutes'] <= Q3 + 1.5 * IQR)
            ].copy()
            if len(race_df_clean) < 3:
                continue
            race_df_clean['first_race_minutes_rounded'] = np.floor(race_df_clean['first_race_minutes']).astype(int)
            bin_stats = race_df_clean.groupby('first_race_minutes_rounded', observed=True).agg({
                'time_diff_seconds': 'mean',
                'athlete_id': 'count'
            }).reset_index()
            bin_stats.columns = ['first_race_minutes', 'time_diff_seconds', 'athlete_count']
            bin_means = bin_stats[bin_stats['athlete_count'] >= 5].copy()
            bin_means = bin_means.sort_values('first_race_minutes')[['first_race_minutes', 'time_diff_seconds']]
            if len(bin_means) > 1:
                all_minutes.extend(bin_means['first_race_minutes'].tolist())
                ax.plot(bin_means['first_race_minutes'], bin_means['time_diff_seconds'],
                        marker=marker_map[num_races], color=color_map[num_races],
                        linewidth=2, markersize=6, linestyle='-',
                        label=f'{num_races} races', alpha=1.0, markeredgecolor='none')

    ax.set_xlabel('Minute Value of First Race', fontsize=10)
    ax.set_ylabel('Avg Time Diff (s)', fontsize=10)
    ax.set_axisbelow(True)

    if len(all_minutes) > 0:
        x_min_full = int(min(all_minutes))
        x_max_full = int(max(all_minutes)) + 1
        ax.set_xlim(x_min_full - 0.5, x_max_full + 0.5)
        ax.set_xticks(list(range(x_min_full, x_max_full + 1)), minor=True)
        ax.grid(True, alpha=0.3, linestyle='-', color='lightgray', which='minor')
        ax.set_xticks(list(range((x_min_full // 2) * 2, x_max_full + 2, 2)))
    else:
        ax.grid(True, alpha=0.3, linestyle='-', color='lightgray')

    mode_desc = 'Standardized (weather & elevation)' if mode == 'standardized' else 'Converted Only (distance)'
    if q1 is not None and q3 is not None:
        x_min, x_max = ax.get_xlim()
        ax.axvspan(x_min, q1, alpha=0.2, color='lightblue', zorder=0)
        ax.axvspan(q1, q3, alpha=0.2, color='lightgreen', zorder=0)
        if outlier_threshold is not None:
            ax.axvspan(q3, x_max, alpha=0.2, color='lightcoral', zorder=0)
        ax.axvline(x=q1, color='blue', linestyle='--', linewidth=1.5, zorder=10)
        ax.axvline(x=q3, color='red', linestyle='--', linewidth=1.5, zorder=10)
        if outlier_threshold is not None:
            ax.axvline(x=outlier_threshold, color='black', linestyle='--', linewidth=1.5, zorder=10)
        title_text = (f'{year}: {mode_desc} - {gender_label}\n'
                      f'IQR: Q1={int(round(q1)):.0f}min, Q3={int(round(q3)):.0f}min')
    else:
        title_text = f'{year}: {mode_desc} - {gender_label}'

    ax.set_title(title_text, fontsize=11, fontweight='normal')
    ax.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='black')


def main(gender='M', output_dir=None):
    """Create the combined 2023-2025 overlay grid for one gender ('M' or 'F')."""
    if gender not in GENDER_META:
        raise ValueError(f"gender must be 'M' or 'F', got {gender!r}")
    meta = GENDER_META[gender]
    output_dir = output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"CREATING COMBINED RQ1 OVERLAY PLOTS 2023-2025 - {meta['plural'].upper()}")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    years = [2023, 2024, 2025]
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found at: {data_dir}")
        return

    print("\n1. Loading data files...")
    try:
        results_df, meet_df, athlete_df, running_event_df, course_details_df = load_data_from_custom_path(data_dir)
    except Exception as e:
        print(f"   ERROR loading data files: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n2. Standardizing both tiers once...")
    try:
        df_conv, df_std = standardize_both_tiers(
            results_df=results_df,
            course_details_df=course_details_df,
            meet_df=meet_df,
            athlete_df=athlete_df,
            running_event_df=running_event_df,
            use_cache=False,
        )
        df_conv = df_conv.copy()
        df_std = df_std.copy()
        df_conv["start_date"] = pd.to_datetime(df_conv["start_date"], errors="coerce")
        df_std["start_date"] = pd.to_datetime(df_std["start_date"], errors="coerce")
    except Exception as e:
        print(f"   ERROR standardizing: {e}")
        import traceback
        traceback.print_exc()
        return

    all_data = {}
    for year in years:
        print(f"\n3. Processing {year} data...")
        all_data[year] = {}
        try:
            df_conv_year = filter_year_data(df_conv, year)
            if len(df_conv_year) > 0:
                df_diff = calculate_first_to_fastest_diff(df_conv_year)
                all_data[year]["non-standardized"] = df_diff[df_diff["gender"] == gender].copy()
            else:
                all_data[year]["non-standardized"] = pd.DataFrame()
        except Exception as e:
            print(f"      ERROR processing {year} non-standardized data: {e}")
            all_data[year]["non-standardized"] = pd.DataFrame()

        try:
            df_std_year = filter_year_data(df_std, year)
            if len(df_std_year) > 0:
                df_diff = calculate_first_to_fastest_diff(df_std_year)
                all_data[year]["standardized"] = df_diff[df_diff["gender"] == gender].copy()
            else:
                all_data[year]["standardized"] = pd.DataFrame()
        except Exception as e:
            print(f"      ERROR processing {year} standardized data: {e}")
            all_data[year]["standardized"] = pd.DataFrame()

    print("\n3. Creating combined overlay plot...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.patch.set_facecolor('white')

    for i, year in enumerate(years):
        ax_left = axes[i, 0]
        if len(all_data[year].get('standardized', pd.DataFrame())) > 0:
            plot_gender_subplot(ax_left, all_data[year]['standardized'], year, 'standardized', meta['label'])
        else:
            ax_left.set_title(f'{year}: Standardized (weather & elevation) - {meta["label"]}\nNo data available', fontsize=11)

        ax_right = axes[i, 1]
        if len(all_data[year].get('non-standardized', pd.DataFrame())) > 0:
            plot_gender_subplot(ax_right, all_data[year]['non-standardized'], year, 'non-standardized', meta['label'])
        else:
            ax_right.set_title(f'{year}: Converted Only (distance) - {meta["label"]}\nNo data available', fontsize=11)

    fig.suptitle(
        f'Combined Overlay Plots: {meta["plural"]} - Avg (First Race - Fastest Other Race) vs. '
        f'First Race Minute 2023 & 2024 & 2025 with IQR Regions and Outlier Boundaries',
        fontsize=12, fontweight='normal', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_path = os.path.join(output_dir, f'combined_overlay_2023_2024_2025_{meta["suffix"]}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved combined overlay plot to: {output_path}')
    plt.close()


if __name__ == '__main__':
    for g in ('M', 'F'):
        main(gender=g)
