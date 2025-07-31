import pandas as pd
import numpy as np
import os
from utils import standardize_convert_exclude_nationals_df, convert_exclude_nationals

output_dir = 'output/PercentileTimeAnalysis'
os.makedirs(output_dir, exist_ok=True)

def remove_outliers_iqr(data, factor=1.5):
    """Remove outliers using 1.5 × IQR rule."""
    if len(data) == 0:
        return data
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def calculate_slope_cv(slopes):
    """Calculate coefficient of variation for slopes.
    
    This measures the consistency of improvement rates across athletes.
    Lower CV = more consistent improvement rates.
    Higher CV = more variable improvement rates.
    Note: Negative slopes indicate improvement (getting faster).
    """
    if len(slopes) == 0:
        return np.nan
    mean_slope = np.mean(slopes)
    if mean_slope == 0:
        return np.nan
    std_slope = np.std(slopes)
    return (std_slope / abs(mean_slope))  # Return as decimal, not percentage

def seconds_to_min_sec(seconds):
    """Convert seconds to min:sec.decimal format."""
    if pd.isna(seconds):
        return np.nan
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:04.1f}"

def analyze_percentile_ranges_final(df, year, gender, mode):
    """Final optimized analysis of athletes by percentile ranges of starting time."""
    print(f"Processing {year} {gender} {mode}...")
    
    # Filter data for year and gender
    start = pd.Timestamp(year=year, month=8, day=1)
    end = pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59)
    
    year_df = df[(df['start_date'] >= start) & (df['start_date'] <= end)].copy()
    gender_df = year_df[year_df['gender'] == gender].copy()
    
    print(f"  Found {len(gender_df)} total records")
    
    # Get athletes with at least 2 races
    athlete_counts = gender_df.groupby('athlete_id').size()
    valid_athletes = athlete_counts[athlete_counts >= 2].index
    
    print(f"  Found {len(valid_athletes)} athletes with 2+ races")
    
    if len(valid_athletes) == 0:
        return pd.DataFrame()
    
    # Calculate starting times and improvements for each athlete
    athlete_data = []
    
    for i, athlete_id in enumerate(valid_athletes):
        if i % 200 == 0:  # Less frequent progress updates
            print(f"  Processing athlete {i+1}/{len(valid_athletes)}")
        
        athlete_races = gender_df[gender_df['athlete_id'] == athlete_id].sort_values('start_date')
        
        if len(athlete_races) < 2:
            continue
        
        # Get first and last race times for the season
        first_time = athlete_races.iloc[0]['standardized_to_target']
        last_time = athlete_races.iloc[-1]['standardized_to_target']
        
        if pd.isna(first_time) or pd.isna(last_time):
            continue
        
        # Calculate total improvement from first to last race (negative = getting faster, positive = getting slower)
        improvement = last_time - first_time
        
        # Calculate slope (improvement rate per day from first to last race)
        first_date = athlete_races.iloc[0]['start_date']
        last_date = athlete_races.iloc[-1]['start_date']
        days_diff = (last_date - first_date).days
        
        if days_diff <= 0:
            continue
        
        slope = improvement / days_diff  # seconds per day (negative = getting faster, positive = getting slower)
        
        athlete_data.append({
            'athlete_id': athlete_id,
            'starting_time': first_time,  # First race time (for percentile grouping)
            'improvement': improvement,
            'slope': slope,
            'num_races': len(athlete_races)
        })
    
    print(f"  Processed {len(athlete_data)} athletes with valid data")
    
    if len(athlete_data) == 0:
        return pd.DataFrame()
    
    # Convert to DataFrame
    athlete_df = pd.DataFrame(athlete_data)
    
    # Remove outliers from starting times using IQR first
    print(f"  Removing outliers from {len(athlete_df)} athletes...")
    
    # Remove outliers from starting times (1.5 × IQR rule)
    starting_times_clean = remove_outliers_iqr(athlete_df['starting_time'].values)
    
    # Create mask for valid starting times
    valid_mask = athlete_df['starting_time'].isin(starting_times_clean)
    athlete_df_clean = athlete_df[valid_mask].copy()
    
    print(f"  After outlier removal: {len(athlete_df_clean)} athletes")
    
    if len(athlete_df_clean) == 0:
        return pd.DataFrame()
    
    # Create percentile ranges (10% each)
    athlete_df_clean['percentile_rank'] = athlete_df_clean['starting_time'].rank(pct=True) * 100
    
    # Define percentile ranges (0-10, 10-20, ..., 90-100)
    percentile_ranges = []
    for i in range(10):
        lower = i * 10
        upper = (i + 1) * 10
        range_mask = (athlete_df_clean['percentile_rank'] > lower) & (athlete_df_clean['percentile_rank'] <= upper)
        range_data = athlete_df_clean[range_mask]
        
        if len(range_data) > 0:
            # Calculate statistics for this percentile range
            starting_time_min = range_data['starting_time'].min()
            starting_time_max = range_data['starting_time'].max()
            median_slope = round(range_data['slope'].median(), 3)  # Median slope (negative = getting faster, positive = getting slower)
            slope_cv = round(calculate_slope_cv(range_data['slope'].values), 3)  # CV of slopes (decimal)
            num_athletes = len(range_data)
            
            # Convert times to min:sec format
            starting_time_min_formatted = seconds_to_min_sec(starting_time_min)
            starting_time_max_formatted = seconds_to_min_sec(starting_time_max)
            
            percentile_ranges.append({
                'percentile_range': f"{lower}-{upper}%",
                'starting_time_min_formatted': starting_time_min_formatted,
                'starting_time_max_formatted': starting_time_max_formatted,
                'starting_time_range': f"{starting_time_min_formatted}-{starting_time_max_formatted}",
                'median_slope': median_slope,
                'slope_cv': slope_cv,
                'num_athletes': num_athletes
            })
    
    print(f"  Created {len(percentile_ranges)} percentile ranges")
    return pd.DataFrame(percentile_ranges)

def main():
    """Main analysis function."""
    print("Loading data...")
    
    # Load both standardized and converted data
    df_std = standardize_convert_exclude_nationals_df()
    df_conv = convert_exclude_nationals()
    
    # Ensure dates are datetime
    df_std['start_date'] = pd.to_datetime(df_std['start_date'], errors='coerce')
    df_conv['start_date'] = pd.to_datetime(df_conv['start_date'], errors='coerce')
    
    print(f"Standardized data: {len(df_std)} records")
    print(f"Converted data: {len(df_conv)} records")
    
    # Analyze each combination and save separate CSV files
    for year in [2023, 2024]:
        for gender in ['M', 'F']:
            gender_label = 'Men' if gender == 'M' else 'Women'
            
            for mode in ['standardized', 'converted']:
                print(f"\nAnalyzing {year} {gender_label} - {mode}...")
                
                # Select appropriate dataset
                df = df_std if mode == 'standardized' else df_conv
                
                # Perform analysis
                results = analyze_percentile_ranges_final(df, year, gender, mode)
                
                if len(results) > 0:
                    # Save individual CSV file
                    filename = f'percentile_analysis_{year}_{gender}_{mode}.csv'
                    filepath = os.path.join(output_dir, filename)
                    results.to_csv(filepath, index=False)
                    print(f"Saved results to: {filepath}")
                    
                    # Print summary
                    print(f"Summary for {year} {gender_label} {mode}:")
                    print(f"  Total athletes: {results['num_athletes'].sum()}")
                    print(f"  Median slope range: {results['median_slope'].min():.4f} to {results['median_slope'].max():.4f} seconds/day")
                    print(f"  Slope CV range: {results['slope_cv'].min():.1f}% to {results['slope_cv'].max():.1f}%")
                    print()
                else:
                    print(f"No results for {year} {gender_label} {mode}")
    
    print("\nAll analyses completed!")

if __name__ == '__main__':
    main() 