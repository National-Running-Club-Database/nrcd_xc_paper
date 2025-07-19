import pandas as pd
import os
import sys

# Create output directory if it doesn't exist
output_dir = '/Users/ryanfryer/Documents/VS Code/National Running Club Database/jss_paper/output/NumberOfRacesQuestion'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data():
    """Load all CSV files from the data directory with error handling."""
    # Use the full data including postseason meets
    directory_path = '../data/jss_data'
    required_files = [
        'team.csv', 'athlete.csv', 'sport.csv', 'running_event.csv',
        'meet.csv', 'result.csv', 'course_details.csv', 'athlete_team_association.csv'
    ]
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Data directory not found: {directory_path}")
    
    # Check if all required files exist
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(directory_path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
    
    try:
        # Load each CSV file into a pandas DataFrame
        team_df = pd.read_csv(os.path.join(directory_path, 'team.csv'))
        athlete_df = pd.read_csv(os.path.join(directory_path, 'athlete.csv'))
        sport_df = pd.read_csv(os.path.join(directory_path, 'sport.csv'))
        running_event_df = pd.read_csv(os.path.join(directory_path, 'running_event.csv'))
        meet_df = pd.read_csv(os.path.join(directory_path, 'meet.csv'))
        result_df = pd.read_csv(os.path.join(directory_path, 'result.csv'))
        course_details_df = pd.read_csv(os.path.join(directory_path, 'course_details.csv'))
        athlete_team_association_df = pd.read_csv(os.path.join(directory_path, 'athlete_team_association.csv'))
        
        return (team_df, athlete_df, sport_df, running_event_df, meet_df, 
                result_df, course_details_df, athlete_team_association_df)
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def save_athlete_list(athlete_ids, filename, description):
    """Save a list of athlete IDs to a file."""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        for athlete_id in athlete_ids:
            f.write(f"{athlete_id}\n")
    print(f"{description} saved to: {filepath}")

def save_dataframe_to_csv(df, filename, description="Save a dataframe to CSV file with optional description."):
    """Save a pandas DataFrame to a CSV file."""
    filepath = os.path.join("/Users/ryanfryer/Documents/VS Code/National Running Club Database/jss_paper/output/NumberOfRacesQuestion", filename)
    df.to_csv(filepath, index=False)
    print(f"{description} saved to: {filepath}")

def categorize_athletes(race_counts_df):
    """Categorize athletes by race count."""
    two_races = race_counts_df[race_counts_df['race_count'] == 2]['athlete_id'].tolist()
    three_races = race_counts_df[race_counts_df['race_count'] == 3]['athlete_id'].tolist()
    four_races = race_counts_df[race_counts_df['race_count'] == 4]['athlete_id'].tolist()
    five_races = race_counts_df[race_counts_df['race_count'] == 5]['athlete_id'].tolist()
    six_races = race_counts_df[race_counts_df['race_count'] == 6]['athlete_id'].tolist()
    return two_races, three_races, four_races, five_races, six_races

def filter_results_by_year(result_df, meet_df, year, running_event_df, athlete_df):
    """Filter results for a given season (Aug 1 - Nov 28 of the given year), and only include 8000m for men and 6000m for women."""
    start_date = pd.Timestamp(year=year, month=8, day=1)
    end_date = pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59)
    # Merge to get meet dates
    merged = pd.merge(result_df, meet_df[['meet_id', 'start_date']], on='meet_id', how='left')
    merged['start_date'] = pd.to_datetime(merged['start_date'], errors='coerce')
    # Merge to get event_name
    merged = pd.merge(merged, running_event_df[['running_event_id', 'event_name']], on='running_event_id', how='left')
    # Merge to get gender
    merged = pd.merge(merged, athlete_df[['athlete_id', 'gender']], on='athlete_id', how='left')
    # Filter for date
    filtered = merged[(merged['start_date'] >= start_date) & (merged['start_date'] <= end_date)]
    # Only keep 8000m for men and 6000m for women
    filtered = filtered[((filtered['gender'] == 'M') & (filtered['event_name'] == '8000m')) |
                        ((filtered['gender'] == 'F') & (filtered['event_name'] == '6000m'))]
    return filtered

def get_time_diff_for_athletes(athlete_ids, year_results):
    """Return a DataFrame with athlete_id and (slowest-fastest) time in seconds for each athlete in the group."""
    diffs = []
    for athlete_id in athlete_ids:
        athlete_times = year_results[year_results['athlete_id'] == athlete_id]['result_time']
        # Parse times to seconds, skip if any are missing or not parseable
        parsed_times = []
        for t in athlete_times:
            try:
                if pd.notna(t):
                    # Try to parse as MM:SS.sss or M:SS.sss
                    parts = str(t).split(':')
                    if len(parts) == 3:
                        h, m, s = map(float, parts)
                        total = h*3600 + m*60 + s
                    elif len(parts) == 2:
                        m, s = map(float, parts)
                        total = m*60 + s
                    else:
                        total = float(t)
                    parsed_times.append(total)
            except Exception:
                continue
        if len(parsed_times) > 1:
            slowest = max(parsed_times)
            fastest = min(parsed_times)
            diff = slowest - fastest
            diffs.append({'athlete_id': athlete_id, 'time_diff_seconds': diff})
    return pd.DataFrame(diffs)

def process_year(year, result_df, meet_df, athlete_df, running_event_df, gender_label):
    print(f"\n{'='*50}\nPROCESSING {year} SEASON ({gender_label})\n{'='*50}")
    # Filter results for the year (now also by event/gender)
    year_results = filter_results_by_year(result_df, meet_df, year, running_event_df, athlete_df)
    # Count races for each athlete
    athlete_race_counts = year_results.groupby('athlete_id').size().reset_index(name='race_count')
    # Merge with athlete data to get gender
    athlete_race_counts = pd.merge(athlete_race_counts, athlete_df[['athlete_id', 'gender']], on='athlete_id')
    # Separate by gender
    male_race_counts = athlete_race_counts[athlete_race_counts['gender'] == 'M']
    female_race_counts = athlete_race_counts[athlete_race_counts['gender'] == 'F']
    # Categorize
    male_2, male_3, male_4, male_5, male_6 = categorize_athletes(male_race_counts)
    female_2, female_3, female_4, female_5, female_6 = categorize_athletes(female_race_counts)
    # Print summary
    print(f"MALE ATHLETES:")
    print(f"2 races: {len(male_2)} athletes")
    print(f"3 races: {len(male_3)} athletes")
    print(f"4 races: {len(male_4)} athletes")
    print(f"5 races: {len(male_5)} athletes")
    print(f"6 races: {len(male_6)} athletes")
    print(f"Total male athletes: {len(male_race_counts)}")
    print(f"FEMALE ATHLETES:")
    print(f"2 races: {len(female_2)} athletes")
    print(f"3 races: {len(female_3)} athletes")
    print(f"4 races: {len(female_4)} athletes")
    print(f"5 races: {len(female_5)} athletes")
    print(f"6 races: {len(female_6)} athletes")
    print(f"Total female athletes: {len(female_race_counts)}")
    # Save lists
    save_athlete_list(male_2, f'male_athletes_2_races_{year}.txt', f"Male athletes with 2 races in {year}")
    save_athlete_list(male_3, f'male_athletes_3_races_{year}.txt', f"Male athletes with 3 races in {year}")
    save_athlete_list(male_4, f'male_athletes_4_races_{year}.txt', f"Male athletes with 4 races in {year}")
    save_athlete_list(male_5, f'male_athletes_5_races_{year}.txt', f"Male athletes with 5 races in {year}")
    save_athlete_list(male_6, f'male_athletes_6_races_{year}.txt', f"Male athletes with 6 races in {year}")
    save_athlete_list(female_2, f'female_athletes_2_races_{year}.txt', f"Female athletes with 2 races in {year}")
    save_athlete_list(female_3, f'female_athletes_3_races_{year}.txt', f"Female athletes with 3 races in {year}")
    save_athlete_list(female_4, f'female_athletes_4_races_{year}.txt', f"Female athletes with 4 races in {year}")
    save_athlete_list(female_5, f'female_athletes_5_races_{year}.txt', f"Female athletes with 5 races in {year}")
    save_athlete_list(female_6, f'female_athletes_6_races_{year}.txt', f"Female athletes with 6 races in {year}")
    # Save summary
    race_count_summary = pd.DataFrame({
        'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female'],
        'Race_Count_Category': ['2', '3', '4', '5', '6', '2', '3', '4', '5', '6'],
        'Athlete_Count': [len(male_2), len(male_3), len(male_4), len(male_5), len(male_6),
                         len(female_2), len(female_3), len(female_4), len(female_5), len(female_6)]
    })
    save_dataframe_to_csv(race_count_summary, f'athlete_race_count_summary_{year}.csv', f"Athlete race count summary {year}")
    # Save detailed
    detailed_race_counts = athlete_race_counts.sort_values(['gender', 'race_count'], ascending=[True, False])
    save_dataframe_to_csv(detailed_race_counts, f'detailed_athlete_race_counts_{year}.csv', f"Detailed race counts for each athlete {year}")
    # Compute and save time differences for each group
    for group_name, athlete_ids, gender in [
        (f'male_2', male_2, 'M'),
        (f'male_3', male_3, 'M'),
        (f'male_4', male_4, 'M'),
        (f'male_5', male_5, 'M'),
        (f'male_6', male_6, 'M'),
        (f'female_2', female_2, 'F'),
        (f'female_3', female_3, 'F'),
        (f'female_4', female_4, 'F'),
        (f'female_5', female_5, 'F'),
        (f'female_6', female_6, 'F'),
    ]:
        group_results = year_results[year_results['gender'] == gender]
        time_diffs_df = get_time_diff_for_athletes(athlete_ids, group_results)
        # Add average row to the CSV
        if not time_diffs_df.empty:
            avg_time_diff = time_diffs_df['time_diff_seconds'].mean()
            avg_row = pd.DataFrame({'athlete_id': ['AVERAGE'], 'time_diff_seconds': [avg_time_diff]})
            time_diffs_df = pd.concat([time_diffs_df, avg_row], ignore_index=True)
        save_dataframe_to_csv(time_diffs_df, f'{group_name}_time_diff_{year}.csv', f"Time difference (slowest-fastest) for {group_name} in {year}")

def main():
    print("=== ATHLETE RACE COUNT ANALYSIS BY YEAR ===")
    print("Using full dataset including postseason meets")
    print()
    # Load the data
    try:
        team_df, athlete_df, sport_df, running_event_df, meet_df, result_df, course_details_df, athlete_team_association_df = load_data()
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        return
    print(f"Total results: {len(result_df):,}")
    print(f"Total athletes: {len(athlete_df):,}")
    print(f"Total meets: {len(meet_df):,}")
    # Process for 2023 and 2024
    process_year(2023, result_df, meet_df, athlete_df, running_event_df, '2023')
    process_year(2024, result_df, meet_df, athlete_df, running_event_df, '2024')

if __name__ == "__main__":
    main()