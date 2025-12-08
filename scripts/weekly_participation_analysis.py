"""
Weekly Participation Analysis

This script analyzes weekly participation patterns:
- Number of meets per week per year
- Number of athletes competing per week per year
- Number of athletes per meet per week per year

Part of RQ1 analysis.

Run from main directory: python scripts/weekly_participation_analysis.py
"""

import os
import sys

# Setup paths for imports
from _setup_paths import setup_paths
setup_paths()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

output_dir = 'output/weekly_participation'
# Directory creation moved to main() to avoid creating when imported

def load_data():
    """Load meet and result data."""
    data_dir = 'data'
    
    meet_df = pd.read_csv(os.path.join(data_dir, 'meet.csv'))
    result_df = pd.read_csv(os.path.join(data_dir, 'result.csv'))
    athlete_df = pd.read_csv(os.path.join(data_dir, 'athlete.csv'))
    
    # Convert dates
    meet_df['start_date'] = pd.to_datetime(meet_df['start_date'], errors='coerce')
    meet_df['end_date'] = pd.to_datetime(meet_df['end_date'], errors='coerce')
    
    # Exclude nationals
    meet_df = meet_df[meet_df['nationals'] == False].copy()
    
    # Filter results to only non-nationals meets
    non_nationals_meet_ids = meet_df['meet_id'].unique()
    result_df = result_df[result_df['meet_id'].isin(non_nationals_meet_ids)].copy()
    
    return meet_df, result_df, athlete_df

def calculate_weekly_stats(meet_df, result_df, athlete_df):
    """Calculate weekly participation statistics."""
    
    # Merge results with meets to get dates
    result_with_dates = result_df.merge(
        meet_df[['meet_id', 'start_date']], 
        on='meet_id', 
        how='left'
    )
    
    # Merge with athlete data to get gender
    result_with_dates = result_with_dates.merge(
        athlete_df[['athlete_id', 'gender']],
        on='athlete_id',
        how='left'
    )
    
    # Filter out rows with missing dates
    result_with_dates = result_with_dates.dropna(subset=['start_date'])
    
    # Add year and week columns
    result_with_dates['year'] = result_with_dates['start_date'].dt.year
    result_with_dates['week'] = result_with_dates['start_date'].dt.isocalendar().week
    meet_df['year'] = meet_df['start_date'].dt.year
    meet_df['week'] = meet_df['start_date'].dt.isocalendar().week
    
    # Filter for 2023, 2024, 2025
    years = [2023, 2024, 2025]
    result_with_dates = result_with_dates[result_with_dates['year'].isin(years)].copy()
    meet_df = meet_df[meet_df['year'].isin(years)].copy()
    
    # Filter to only include meets from August 1 through October 31 (by actual date, not week)
    # This means if a week starts before Oct 31 but has meets after, we still include the week
    # but only count meets on or before Oct 31
    for year in years:
        aug_1 = pd.Timestamp(year=year, month=8, day=1)
        oct_31 = pd.Timestamp(year=year, month=10, day=31, hour=23, minute=59, second=59)
        
        # Filter meets for this year
        year_meet_mask = (meet_df['year'] == year) & (meet_df['start_date'] >= aug_1) & (meet_df['start_date'] <= oct_31)
        meet_df = meet_df[year_meet_mask | (meet_df['year'] != year)].copy()
        
        # Filter results for this year (based on meet dates)
        year_result_mask = (result_with_dates['year'] == year) & (result_with_dates['start_date'] >= aug_1) & (result_with_dates['start_date'] <= oct_31)
        result_with_dates = result_with_dates[year_result_mask | (result_with_dates['year'] != year)].copy()
    
    # Find actual earliest and latest dates for x-axis title
    earliest_date = meet_df['start_date'].min()
    latest_date = meet_df['start_date'].max()
    print(f"Earliest date across all years: {earliest_date.strftime('%Y-%m-%d')}")
    print(f"Latest date across all years: {latest_date.strftime('%Y-%m-%d')}")
    
    # Format date range for x-axis title
    def format_date_for_title(d):
        return d.strftime('%b %d').lstrip('0').replace('  ', ' ')
    
    date_range_title = f"Date Range: {format_date_for_title(earliest_date)} - {format_date_for_title(latest_date)}"
    
    # Calculate week_number for each meet and result based on days from August 1 of THAT YEAR
    # This ensures the same calendar week across years gets the same week_number
    # For example, Aug 25 in any year will be week 4 (since Aug 25 is 24 days after Aug 1 of that year)
    def calculate_week_number(row):
        year = row['start_date'].year
        aug_1 = pd.Timestamp(year=year, month=8, day=1)
        days_from_aug1 = (row['start_date'] - aug_1).days
        return (days_from_aug1 // 7) + 1
    
    meet_df['week_number'] = meet_df.apply(calculate_week_number, axis=1)
    result_with_dates['week_number'] = result_with_dates.apply(calculate_week_number, axis=1)
    
    # Initialize results list
    weekly_stats = []
    
    # Get all unique week_numbers across all years (this ensures proper overlay)
    all_week_numbers = sorted(meet_df['week_number'].unique())
    
    for week_number in all_week_numbers:
        # Get all meets and results for this week_number (across all years)
        week_meets = meet_df[meet_df['week_number'] == week_number]
        week_results = result_with_dates[result_with_dates['week_number'] == week_number]
        
        # Group by year to calculate statistics per year
        for year in years:
            year_week_meets = week_meets[week_meets['year'] == year]
            year_week_results = week_results[week_results['year'] == year]
            
            # Skip if no data for this year/week
            if len(year_week_meets) == 0:
                continue
            
            # Number of meets this week for this year
            num_meets = len(year_week_meets)
            
            # Number of unique athletes competing this week for this year
            num_athletes = year_week_results['athlete_id'].nunique()
            
            # Number of athletes per meet (average)
            athletes_per_meet = num_athletes / num_meets if num_meets > 0 else 0
            
            # Get date range for this week (for labeling) - use this year's dates
            week_start_date = year_week_meets['start_date'].min()
            week_end_date = year_week_meets['start_date'].max()
            
            # Always show just the start date (simplified)
            # Format: remove leading zero from day (e.g., "Sep 01" -> "Sep 1")
            def format_date(d):
                return d.strftime('%b %d').lstrip('0').replace('  ', ' ')
            
            week_date_label = format_date(week_start_date)
            
            # Also calculate by gender
            for gender in ['M', 'F']:
                gender_results = year_week_results[year_week_results['gender'] == gender]
                gender_athletes = gender_results['athlete_id'].nunique()
                gender_athletes_per_meet = gender_athletes / num_meets if num_meets > 0 else 0
                
                weekly_stats.append({
                    'year': year,
                    'week_number': week_number,  # Week number relative to earliest date (1, 2, 3, ...)
                    'week_start_date': week_start_date,
                    'week_end_date': week_end_date,
                    'week_date_label': week_date_label,  # For x-axis labels (actual dates)
                    'gender': gender,
                    'num_meets': num_meets,
                    'num_athletes': gender_athletes,
                    'athletes_per_meet': gender_athletes_per_meet
                })
            
            # Also add overall (both genders combined)
            weekly_stats.append({
                'year': year,
                'week_number': week_number,  # Week number relative to earliest date (1, 2, 3, ...)
                'week_start_date': week_start_date,
                'week_end_date': week_end_date,
                'week_date_label': week_date_label,  # For x-axis labels (actual dates)
                'gender': 'All',
                'num_meets': num_meets,
                'num_athletes': num_athletes,
                'athletes_per_meet': athletes_per_meet
            })
    
    stats_df = pd.DataFrame(weekly_stats)
    
    # Store date range title as an attribute for use in plotting
    stats_df.attrs['date_range_title'] = date_range_title
    
    return stats_df

def create_weekly_participation_plots(stats_df, output_dir):
    """Create visualizations for weekly participation."""
    
    # Get date range title from stats_df attributes
    date_range_title = stats_df.attrs.get('date_range_title', 'Date Range')
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Weekly Participation Patterns by Year (August through October)', fontsize=16, fontweight='bold')
    
    years = sorted(stats_df['year'].unique())
    colors = {2023: 'C0', 2024: 'C1', 2025: 'C2'}
    
    # Get all unique week numbers across all years and genders (for consistent x-axis)
    all_weeks = sorted(stats_df['week_number'].unique())
    min_week = min(all_weeks)
    max_week = max(all_weeks)
    
    # Set x-axis labels to show week number and date range
    # For each week, get the date range (prefer reference year, but use any year if needed)
    tick_weeks = all_weeks[::max(1, len(all_weeks)//10)]  # Show every nth week to avoid crowding
    tick_labels = []
    for week_num in tick_weeks:
        # Try reference year first
        week_data = stats_df[(stats_df['year'] == years[0]) & (stats_df['gender'] == 'All') & (stats_df['week_number'] == week_num)]
        if len(week_data) > 0:
            date_label = week_data.iloc[0]['week_date_label']
            tick_labels.append(f"Week {week_num}\n{date_label}")
        else:
            # Try any year
            any_week_data = stats_df[(stats_df['gender'] == 'All') & (stats_df['week_number'] == week_num)]
            if len(any_week_data) > 0:
                date_label = any_week_data.iloc[0]['week_date_label']
                tick_labels.append(f"Week {week_num}\n{date_label}")
            else:
                tick_labels.append(f"Week {week_num}")
    
    # Plot 1: Number of meets per week
    ax1 = axes[0]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'All')].sort_values('week_number')
        # Use week_number for x-axis positioning (so years overlay)
        ax1.plot(year_data['week_number'], year_data['num_meets'], 
                marker='o', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    
    # Set consistent x-axis limits and ticks for all plots
    ax1.set_xlim(min_week - 0.5, max_week + 0.5)
    ax1.set_xticks(tick_weeks)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    ax1.set_xlabel('Date Range', fontsize=12)
    ax1.set_ylabel('Number of Meets', fontsize=12)
    ax1.set_title('Number of Meets per Week by Year', fontsize=13, fontweight='bold')
    ax1.legend(title='Year', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of athletes competing per week
    ax2 = axes[1]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'All')].sort_values('week_number')
        ax2.plot(year_data['week_number'], year_data['num_athletes'], 
                marker='s', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    
    # Use same tick labels and limits for all plots
    ax2.set_xlim(min_week - 0.5, max_week + 0.5)
    ax2.set_xticks(tick_weeks)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    ax2.set_xlabel(date_range_title, fontsize=12)
    ax2.set_ylabel('Number of Athletes Competing', fontsize=12)
    ax2.set_title('Number of Athletes Competing per Week by Year', fontsize=13, fontweight='bold')
    ax2.legend(title='Year', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Number of athletes per meet per week
    ax3 = axes[2]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'All')].sort_values('week_number')
        ax3.plot(year_data['week_number'], year_data['athletes_per_meet'], 
                marker='^', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    
    # Use same tick labels and limits for all plots
    ax3.set_xlim(min_week - 0.5, max_week + 0.5)
    ax3.set_xticks(tick_weeks)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    ax3.set_xlabel(date_range_title, fontsize=12)
    ax3.set_ylabel('Athletes per Meet (Average)', fontsize=12)
    ax3.set_title('Average Number of Athletes per Meet per Week by Year', fontsize=13, fontweight='bold')
    ax3.legend(title='Year', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'weekly_participation_overview.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved weekly participation overview to {output_path}")
    
    # Create gender comparison plots
    create_gender_comparison_plots(stats_df, output_dir)

def create_gender_comparison_plots(stats_df, output_dir):
    """Create gender-specific comparison plots."""
    
    # Get date range title from stats_df attributes
    date_range_title_gender = stats_df.attrs.get('date_range_title', 'Date Range')
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle('Weekly Participation Patterns by Gender and Year (August through October)', fontsize=16, fontweight='bold')
    
    years = sorted(stats_df['year'].unique())
    colors = {2023: 'C0', 2024: 'C1', 2025: 'C2'}
    genders = ['M', 'F']
    gender_labels = {'M': 'Men', 'F': 'Women'}
    
    
    # Get all unique week numbers (for consistent x-axis across all gender plots)
    all_weeks_gender = sorted(stats_df['week_number'].unique())
    min_week_gender = min(all_weeks_gender)
    max_week_gender = max(all_weeks_gender)
    
    # For each week, get the date range (prefer reference year, but use any year if needed)
    tick_weeks_gender = all_weeks_gender[::max(1, len(all_weeks_gender)//10)]  # Show every nth week
    tick_labels_gender = []
    for week_num in tick_weeks_gender:
        # Try reference year first
        week_data = stats_df[(stats_df['year'] == years[0]) & (stats_df['gender'] == 'M') & (stats_df['week_number'] == week_num)]
        if len(week_data) > 0:
            date_label = week_data.iloc[0]['week_date_label']
            tick_labels_gender.append(f"Week {week_num}\n{date_label}")
        else:
            # Try any year
            any_week_data = stats_df[(stats_df['gender'] == 'M') & (stats_df['week_number'] == week_num)]
            if len(any_week_data) > 0:
                date_label = any_week_data.iloc[0]['week_date_label']
                tick_labels_gender.append(f"Week {week_num}\n{date_label}")
            else:
                tick_labels_gender.append(f"Week {week_num}")
    
    # Plot 1: Meets per week (Men)
    ax1 = axes[0, 0]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'M')].sort_values('week_number')
        ax1.plot(year_data['week_number'], year_data['num_meets'], 
                marker='o', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    
    # For each week, get the date range (prefer reference year, but use any year if needed)
    tick_weeks_gender = all_weeks_gender
    tick_labels_gender = []
    for week_num in tick_weeks_gender:
        # Try reference year first
        week_data = stats_df[(stats_df['year'] == years[0]) & (stats_df['gender'] == 'M') & (stats_df['week_number'] == week_num)]
        if len(week_data) > 0:
            date_label = week_data.iloc[0]['week_date_label']
            tick_labels_gender.append(f"Week {week_num}\n{date_label}")
        else:
            # Try any year
            any_week_data = stats_df[(stats_df['gender'] == 'M') & (stats_df['week_number'] == week_num)]
            if len(any_week_data) > 0:
                date_label = any_week_data.iloc[0]['week_date_label']
                tick_labels_gender.append(f"Week {week_num}\n{date_label}")
            else:
                tick_labels_gender.append(f"Week {week_num}")
    
    ax1.set_xticks(tick_weeks_gender)
    ax1.set_xticklabels(tick_labels_gender, rotation=45, ha='right', fontsize=8)
    ax1.set_title('Men: Number of Meets per Week', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date Range', fontsize=11)
    ax1.set_ylabel('Number of Meets', fontsize=11)
    ax1.legend(title='Year', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Meets per week (Women)
    ax2 = axes[0, 1]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'F')].sort_values('week_number')
        ax2.plot(year_data['week_number'], year_data['num_meets'], 
                marker='o', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    ax2.set_xlim(min_week_gender - 0.5, max_week_gender + 0.5)
    ax2.set_xticks(tick_weeks_gender)
    ax2.set_xticklabels(tick_labels_gender, rotation=45, ha='right', fontsize=8)
    ax2.set_title('Women: Number of Meets per Week', fontsize=12, fontweight='bold')
    ax2.set_xlabel(date_range_title_gender, fontsize=11)
    ax2.set_ylabel('Number of Meets', fontsize=11)
    ax2.legend(title='Year', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Athletes per week (Men)
    ax3 = axes[1, 0]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'M')].sort_values('week_number')
        ax3.plot(year_data['week_number'], year_data['num_athletes'], 
                marker='s', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    ax3.set_xlim(min_week_gender - 0.5, max_week_gender + 0.5)
    ax3.set_xticks(tick_weeks_gender)
    ax3.set_xticklabels(tick_labels_gender, rotation=45, ha='right', fontsize=8)
    ax3.set_title('Men: Number of Athletes Competing per Week', fontsize=12, fontweight='bold')
    ax3.set_xlabel(date_range_title_gender, fontsize=11)
    ax3.set_ylabel('Number of Athletes', fontsize=11)
    ax3.legend(title='Year', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Athletes per week (Women)
    ax4 = axes[1, 1]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'F')].sort_values('week_number')
        ax4.plot(year_data['week_number'], year_data['num_athletes'], 
                marker='s', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    ax4.set_xlim(min_week_gender - 0.5, max_week_gender + 0.5)
    ax4.set_xticks(tick_weeks_gender)
    ax4.set_xticklabels(tick_labels_gender, rotation=45, ha='right', fontsize=8)
    ax4.set_title('Women: Number of Athletes Competing per Week', fontsize=12, fontweight='bold')
    ax4.set_xlabel(date_range_title_gender, fontsize=11)
    ax4.set_ylabel('Number of Athletes', fontsize=11)
    ax4.legend(title='Year', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Athletes per meet (Men)
    ax5 = axes[2, 0]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'M')].sort_values('week_number')
        ax5.plot(year_data['week_number'], year_data['athletes_per_meet'], 
                marker='^', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    ax5.set_xlim(min_week_gender - 0.5, max_week_gender + 0.5)
    ax5.set_xticks(tick_weeks_gender)
    ax5.set_xticklabels(tick_labels_gender, rotation=45, ha='right', fontsize=8)
    ax5.set_title('Men: Average Athletes per Meet per Week', fontsize=12, fontweight='bold')
    ax5.set_xlabel(date_range_title_gender, fontsize=11)
    ax5.set_ylabel('Athletes per Meet', fontsize=11)
    ax5.legend(title='Year', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Athletes per meet (Women)
    ax6 = axes[2, 1]
    for year in years:
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'F')].sort_values('week_number')
        ax6.plot(year_data['week_number'], year_data['athletes_per_meet'], 
                marker='^', label=f'{year}', linewidth=2, markersize=6, color=colors[year])
    ax6.set_xlim(min_week_gender - 0.5, max_week_gender + 0.5)
    ax6.set_xticks(tick_weeks_gender)
    ax6.set_xticklabels(tick_labels_gender, rotation=45, ha='right', fontsize=8)
    ax6.set_title('Women: Average Athletes per Meet per Week', fontsize=12, fontweight='bold')
    ax6.set_xlabel(date_range_title_gender, fontsize=11)
    ax6.set_ylabel('Athletes per Meet', fontsize=11)
    ax6.legend(title='Year', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'weekly_participation_by_gender.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved gender comparison plots to {output_path}")

def main():
    """Main analysis function."""
    # Create output directory only when run directly
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("WEEKLY PARTICIPATION ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    meet_df, result_df, athlete_df = load_data()
    print(f"  Loaded {len(meet_df)} meets (excluding nationals)")
    print(f"  Loaded {len(result_df)} results")
    
    # Calculate weekly statistics
    print("\nCalculating weekly statistics...")
    stats_df = calculate_weekly_stats(meet_df, result_df, athlete_df)
    print(f"  Calculated stats for {len(stats_df)} week-gender combinations")
    
    # Save statistics to CSV
    csv_path = os.path.join(output_dir, 'weekly_participation_stats.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"  Saved statistics to {csv_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_weekly_participation_plots(stats_df, output_dir)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for year in sorted(stats_df['year'].unique()):
        year_data = stats_df[(stats_df['year'] == year) & (stats_df['gender'] == 'All')]
        print(f"\n{year}:")
        print(f"  Average meets per week: {year_data['num_meets'].mean():.1f}")
        print(f"  Average athletes per week: {year_data['num_athletes'].mean():.0f}")
        print(f"  Average athletes per meet: {year_data['athletes_per_meet'].mean():.1f}")
        print(f"  Peak week (most meets): Week {year_data.loc[year_data['num_meets'].idxmax(), 'week_number']} ({year_data['num_meets'].max()} meets)")
        print(f"  Peak week (most athletes): Week {year_data.loc[year_data['num_athletes'].idxmax(), 'week_number']} ({year_data['num_athletes'].max():.0f} athletes)")
    
    print("\n✅ Weekly participation analysis complete!")

if __name__ == '__main__':
    main()

