"""
Create RQ1 overlay plots for a given season year (2023, 2024, or 2025).

For each year this produces two plots (standardized and converted-only) showing:
1. First race time (in minutes) vs time difference (first race - fastest other race)
2. Grouped by number of races per season (2, 3, 4)

This single, year-parametrized script replaces the former
create_rq1_overlay_2023.py / _2024.py / _2025.py trio, which differed only by
the year constant.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Change to scripts directory to match utils.py expectations
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)
from utils import standardize_both_tiers
from load_nrcd_data import get_data_dir

data_dir = get_data_dir()
base_dir = os.path.dirname(script_dir)

# Matplotlib style
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


def default_output_dir(year):
    return os.path.join(base_dir, 'output', 'rq1', 'overlay_plots', f'overlay_{year}')


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


def _plot_gender_axis(ax, gender_df, year, mode, gender_label):
    """Draw one gender's overlay (race-count lines + IQR shading) on ``ax``."""
    q1 = gender_df['first_race_minutes'].quantile(0.25) if len(gender_df) > 0 else None
    q3 = gender_df['first_race_minutes'].quantile(0.75) if len(gender_df) > 0 else None
    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
    outlier_threshold = (q3 + 1.5 * iqr) if iqr is not None else None

    race_counts = [2, 3, 4]
    color_map = {2: '#1f77b4', 3: '#ff7f0e', 4: '#2ca02c'}
    marker_map = {2: 'o', 3: 's', 4: '^'}

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

    mode_label = 'Non-standardized' if mode == 'non-standardized' else 'Standardized'
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
        title_text = (f'{mode_label} - {gender_label} ({year})\n'
                      f'IQR: Q1 = {int(round(q1))} min, Q3 = {int(round(q3))} min')
    else:
        title_text = f'{mode_label} - {gender_label} ({year})'

    ax.set_title(title_text, fontsize=11, fontweight='normal')
    ax.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='black')


def create_overlay_plot(df_diff, year, mode, output_path):
    """Create a men (left) / women (right) overlay plot for one mode."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    _plot_gender_axis(axes[0], df_diff[df_diff['gender'] == 'M'].copy(), year, mode, 'Male')
    _plot_gender_axis(axes[1], df_diff[df_diff['gender'] == 'F'].copy(), year, mode, 'Female')

    mode_label = 'Non-standardized' if mode == 'non-standardized' else 'Standardized'
    fig.suptitle(
        f'Overlay: Avg (First Race - Fastest Other Race) vs. First Race Minute {mode_label} ({year})',
        fontsize=12, fontweight='normal', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved overlay plot to: {output_path}')
    plt.close()


def main(year=2023, output_dir=None):
    """Create standardized and converted-only overlay plots for ``year``."""
    output_dir = output_dir or default_output_dir(year)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"CREATING RQ1 OVERLAY PLOTS FOR {year}")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Date range: August 25, {year} to November 27, {year}")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found at: {data_dir}")
        return

    print("\n1. Loading data files...")
    try:
        results_df, meet_df, athlete_df, running_event_df, course_details_df = load_data_from_custom_path(data_dir)
        print(f"   - Results: {len(results_df)} records | Meets: {len(meet_df)} | Athletes: {len(athlete_df)}")
    except Exception as e:
        print(f"   ERROR loading data files: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n2. Standardizing both tiers (converted + weather/elevation)...")
    try:
        df_conv, df_std = standardize_both_tiers(
            results_df=results_df,
            course_details_df=course_details_df,
            meet_df=meet_df,
            athlete_df=athlete_df,
            running_event_df=running_event_df,
            use_cache=False,
        )
        for label, df_tier in (
            ("non-standardized", df_conv),
            ("standardized", df_std),
        ):
            df_tier = df_tier.copy()
            df_tier["start_date"] = pd.to_datetime(df_tier["start_date"], errors="coerce")
            df_year = filter_year_data(df_tier, year)
            print(f"   {label}: {len(df_year)} records for {year}")
            if len(df_year) == 0:
                continue
            df_diff = calculate_first_to_fastest_diff(df_year)
            if len(df_diff) > 0:
                create_overlay_plot(
                    df_diff,
                    year,
                    label,
                    os.path.join(output_dir, f"overlay_{year}_{label}.pdf"),
                )
    except Exception as e:
        print(f"   ERROR processing standardized data: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nPlots saved to: {os.path.abspath(output_dir)}/")


if __name__ == '__main__':
    years = [int(a) for a in sys.argv[1:]] or [2023, 2024, 2025]
    for y in years:
        main(y)
