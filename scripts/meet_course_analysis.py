#!/usr/bin/env python3
"""
Meet and Course Details Analysis Script

This script analyzes meet data by month (August, September, October, November)
excluding nationals, and tracks course details for both meets and results by gender.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data():
    """Load all necessary data files."""
    data_dir = "data/data"
    
    # Load main data files
    meets = pd.read_csv(os.path.join(data_dir, "meet.csv"))
    course_details = pd.read_csv(os.path.join(data_dir, "course_details.csv"))
    athletes = pd.read_csv(os.path.join(data_dir, "athlete.csv"))
    results = pd.read_csv(os.path.join(data_dir, "result.csv"))
    
    return meets, course_details, athletes, results

def filter_meets_by_criteria(meets):
    """
    Filter meets to include only:
    - August, September, October, November
    - Exclude nationals
    - Only 2023 and 2024
    """
    # Convert start_date to datetime
    meets['start_date'] = pd.to_datetime(meets['start_date'], errors='coerce')
    
    # Filter by month (August = 8, September = 9, October = 10, November = 11)
    # and year (2023, 2024)
    meets_filtered = meets[
        (meets['start_date'].dt.month.isin([8, 9, 10, 11])) &
        (meets['start_date'].dt.year.isin([2023, 2024])) &
        (meets['nationals'] == False) &
        (meets['start_date'].notna())
    ].copy()
    
    # Add year and month columns for easier analysis
    meets_filtered['year'] = meets_filtered['start_date'].dt.year
    meets_filtered['month'] = meets_filtered['start_date'].dt.month
    meets_filtered['month_name'] = meets_filtered['start_date'].dt.strftime('%B')
    
    return meets_filtered

def analyze_meet_course_details(meets_filtered, course_details):
    """Analyze which meets have course details and which don't."""
    
    # Get unique meet IDs that have course details
    meets_with_course_details = course_details['meet_id'].unique()
    
    # Add course details flag to meets
    meets_filtered['has_course_details'] = meets_filtered['meet_id'].isin(meets_with_course_details)
    
    return meets_filtered

def analyze_results_by_month_and_gender(meets_filtered, course_details, results, athletes):
    """Analyze results with course details by month and gender."""
    
    # Get meets with course details
    meets_with_course_details = course_details['meet_id'].unique()
    
    # Add year and month to results for joining
    results['meet_id'] = results['meet_id'].astype(int)
    
    # Join results with meets to get year and month info
    results_with_meets = results.merge(
        meets_filtered[['meet_id', 'year', 'month', 'month_name', 'has_course_details']], 
        on='meet_id', 
        how='inner'
    )
    
    # Join with athletes to get gender information
    results_with_gender = results_with_meets.merge(
        athletes[['athlete_id', 'gender']], 
        on='athlete_id', 
        how='left'
    )
    
    # Group by year, month, and gender to count results
    monthly_gender_results = results_with_gender.groupby(['year', 'month', 'month_name', 'gender']).agg({
        'result_id': 'count',  # Total results
        'has_course_details': ['sum', lambda x: (~x).sum()]  # Results with/without course details
    }).reset_index()
    
    # Flatten column names
    monthly_gender_results.columns = ['year', 'month', 'month_name', 'gender', 'total_results', 'results_with_course_details', 'results_without_course_details']
    
    return monthly_gender_results

def create_monthly_summary(meets_filtered):
    """Create summary by year and month."""
    
    # Group by year and month
    monthly_summary = meets_filtered.groupby(['year', 'month', 'month_name']).agg({
        'meet_id': 'count',
        'has_course_details': ['sum', lambda x: (~x).sum()]
    }).reset_index()
    
    # Flatten column names
    monthly_summary.columns = ['year', 'month', 'month_name', 'total_meets', 'meets_with_course_details', 'meets_without_course_details']
    
    # Sort by year and month
    monthly_summary = monthly_summary.sort_values(['year', 'month'])
    
    return monthly_summary

def main():
    """Main analysis function."""
    print("Loading data...")
    meets, course_details, athletes, results = load_data()
    
    print("Filtering meets...")
    meets_filtered = filter_meets_by_criteria(meets)
    
    print("Analyzing meet course details...")
    meets_filtered = analyze_meet_course_details(meets_filtered, course_details)
    
    print("Analyzing results by month and gender...")
    monthly_gender_results = analyze_results_by_month_and_gender(meets_filtered, course_details, results, athletes)
    
    print("Creating detailed analysis...")
    monthly_summary = create_monthly_summary(meets_filtered)
    
    # Create output directory if it doesn't exist
    output_dir = "output/MeetCourseAnalysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed analysis by merging meet and results data
    detailed_analysis = monthly_summary.merge(
        monthly_gender_results[['year', 'month', 'gender', 'total_results', 'results_with_course_details', 'results_without_course_details']], 
        on=['year', 'month'], 
        how='left'
    )
    
    # Sort by year, month, and gender (M first, then F)
    detailed_analysis = detailed_analysis.sort_values(['year', 'month', 'gender'], key=lambda x: x.map({'M': 0, 'F': 1}) if x.name == 'gender' else x)
    
    detailed_analysis.to_csv(os.path.join(output_dir, "detailed_meet_course_analysis.csv"), index=False)
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Total meets analyzed: {len(meets_filtered)}")
    print(f"Meets with course details: {meets_filtered['has_course_details'].sum()}")
    print(f"Meets without course details: {(~meets_filtered['has_course_details']).sum()}")
    print(f"Total results: {monthly_gender_results['total_results'].sum()}")
    print(f"Results with course details: {monthly_gender_results['results_with_course_details'].sum()}")
    print(f"Results without course details: {monthly_gender_results['results_without_course_details'].sum()}")
    
    # Print gender breakdown
    gender_summary = monthly_gender_results.groupby('gender').agg({
        'total_results': 'sum',
        'results_with_course_details': 'sum',
        'results_without_course_details': 'sum'
    })
    print(f"\nGender breakdown:")
    for gender in gender_summary.index:
        total = gender_summary.loc[gender, 'total_results']
        with_course = gender_summary.loc[gender, 'results_with_course_details']
        without_course = gender_summary.loc[gender, 'results_without_course_details']
        print(f"{gender}: {total} total results ({with_course} with course details, {without_course} without)")
    
    print(f"\nResults saved to: {output_dir}/detailed_meet_course_analysis.csv")

if __name__ == "__main__":
    main() 