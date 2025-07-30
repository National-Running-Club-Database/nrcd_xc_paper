import pandas as pd
import numpy as np
import os

def load_data():
    """Load the data files."""
    data_dir = 'data/jss_data'
    
    # Load the main data files
    result_df = pd.read_csv(os.path.join(data_dir, 'result.csv'))
    course_details_df = pd.read_csv(os.path.join(data_dir, 'course_details.csv'))
    running_event_df = pd.read_csv(os.path.join(data_dir, 'running_event.csv'))
    athlete_df = pd.read_csv(os.path.join(data_dir, 'athlete.csv'))
    meet_df = pd.read_csv(os.path.join(data_dir, 'meet.csv'))
    
    return result_df, course_details_df, running_event_df, athlete_df, meet_df

def analyze_missing_course_details_by_event(result_df, course_details_df, running_event_df, athlete_df):
    """Analyze how many unique meet-event-gender combinations don't have course details."""
    
    # Merge data to get complete picture
    merged_df = result_df.merge(
        athlete_df[['athlete_id', 'gender']], 
        on='athlete_id', 
        how='left'
    ).merge(
        running_event_df[['running_event_id', 'event_name']], 
        on='running_event_id', 
        how='left'
    )
    
    # Filter for cross country events (all cross country distances)
    cross_country_events = ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']
    cc_df = merged_df[merged_df['event_name'].isin(cross_country_events)].copy()
    
    # Get unique meet-event-gender combinations from results
    unique_events = cc_df.groupby(['meet_id', 'running_event_id', 'gender', 'event_name']).size().reset_index()
    unique_events = unique_events[['meet_id', 'running_event_id', 'gender', 'event_name']]
    
    # Create course details key
    course_details_df['course_match_key'] = course_details_df.apply(
        lambda row: f"{row['meet_id']}_{row['running_event_id']}_{row['gender']}", 
        axis=1
    )
    
    # Create key for unique events
    unique_events['course_match_key'] = unique_events.apply(
        lambda row: f"{row['meet_id']}_{row['running_event_id']}_{row['gender']}", 
        axis=1
    )
    
    # Check which events have course details
    events_with_course_details = unique_events['course_match_key'].isin(course_details_df['course_match_key'])
    
    # Calculate statistics
    total_events = len(unique_events)
    events_with_details = events_with_course_details.sum()
    events_without_details = total_events - events_with_details
    
    print("=== COURSE DETAILS ANALYSIS BY EVENT ===")
    print(f"Total unique cross country events: {total_events:,}")
    print(f"Events with course details: {events_with_details:,} ({events_with_details/total_events*100:.1f}%)")
    print(f"Events without course details: {events_without_details:,} ({events_without_details/total_events*100:.1f}%)")
    
    # Break down by gender
    print("\n--- By Gender ---")
    for gender in ['M', 'F']:
        gender_events = unique_events[unique_events['gender'] == gender]
        gender_total = len(gender_events)
        gender_with_details = gender_events['course_match_key'].isin(course_details_df['course_match_key']).sum()
        gender_without_details = gender_total - gender_with_details
        
        print(f"{'Men' if gender == 'M' else 'Women'}:")
        print(f"  Total events: {gender_total:,}")
        print(f"  With course details: {gender_with_details:,} ({gender_with_details/gender_total*100:.1f}%)")
        print(f"  Without course details: {gender_without_details:,} ({gender_without_details/gender_total*100:.1f}%)")
    
    # Break down by distance
    print("\n--- By Distance ---")
    for distance in ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']:
        distance_events = unique_events[unique_events['event_name'] == distance]
        distance_total = len(distance_events)
        distance_with_details = distance_events['course_match_key'].isin(course_details_df['course_match_key']).sum()
        distance_without_details = distance_total - distance_with_details
        
        print(f"{distance}:")
        print(f"  Total events: {distance_total:,}")
        print(f"  With course details: {distance_with_details:,} ({distance_with_details/distance_total*100:.1f}%)")
        print(f"  Without course details: {distance_without_details:,} ({distance_without_details/distance_total*100:.1f}%)")
    
    # Show some examples of events without course details
    events_without_details_df = unique_events[~unique_events['course_match_key'].isin(course_details_df['course_match_key'])]
    
    print(f"\n--- Sample Events Without Course Details ---")
    print("First 10 events without course details:")
    for idx, row in events_without_details_df.head(10).iterrows():
        print(f"  Meet ID: {row['meet_id']}, Event: {row['event_name']}, Gender: {row['gender']}")
    
    return unique_events, course_details_df

def analyze_missing_course_details_by_race(result_df, course_details_df, running_event_df, athlete_df):
    """Analyze how many individual race results don't have course details."""
    
    # Merge data to get complete picture
    merged_df = result_df.merge(
        athlete_df[['athlete_id', 'gender']], 
        on='athlete_id', 
        how='left'
    ).merge(
        running_event_df[['running_event_id', 'event_name']], 
        on='running_event_id', 
        how='left'
    )
    
    # Filter for cross country events (all cross country distances)
    cross_country_events = ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']
    cc_df = merged_df[merged_df['event_name'].isin(cross_country_events)].copy()
    
    # Create course details key
    course_details_df['course_match_key'] = course_details_df.apply(
        lambda row: f"{row['meet_id']}_{row['running_event_id']}_{row['gender']}", 
        axis=1
    )
    
    # Create key for individual races
    cc_df['course_match_key'] = cc_df.apply(
        lambda row: f"{row['meet_id']}_{row['running_event_id']}_{row['gender']}", 
        axis=1
    )
    
    # Check which individual races have course details
    races_with_course_details = cc_df['course_match_key'].isin(course_details_df['course_match_key'])
    
    # Calculate statistics
    total_races = len(cc_df)
    races_with_details = races_with_course_details.sum()
    races_without_details = total_races - races_with_details
    
    print("=== INDIVIDUAL RACE COURSE DETAILS ANALYSIS ===")
    print(f"Total individual cross country races: {total_races:,}")
    print(f"Races with course details: {races_with_details:,} ({races_with_details/total_races*100:.1f}%)")
    print(f"Races without course details: {races_without_details:,} ({races_without_details/total_races*100:.1f}%)")
    
    # Break down by gender
    print("\n--- By Gender ---")
    for gender in ['M', 'F']:
        gender_df = cc_df[cc_df['gender'] == gender]
        gender_total = len(gender_df)
        gender_with_details = gender_df['course_match_key'].isin(course_details_df['course_match_key']).sum()
        gender_without_details = gender_total - gender_with_details
        
        print(f"{'Men' if gender == 'M' else 'Women'}:")
        print(f"  Total races: {gender_total:,}")
        print(f"  With course details: {gender_with_details:,} ({gender_with_details/gender_total*100:.1f}%)")
        print(f"  Without course details: {gender_without_details:,} ({gender_without_details/gender_total*100:.1f}%)")
    
    # Break down by distance
    print("\n--- By Distance ---")
    for distance in ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']:
        distance_df = cc_df[cc_df['event_name'] == distance]
        distance_total = len(distance_df)
        distance_with_details = distance_df['course_match_key'].isin(course_details_df['course_match_key']).sum()
        distance_without_details = distance_total - distance_with_details
        
        print(f"{distance}:")
        print(f"  Total races: {distance_total:,}")
        print(f"  With course details: {distance_with_details:,} ({distance_with_details/distance_total*100:.1f}%)")
        print(f"  Without course details: {distance_without_details:,} ({distance_without_details/distance_total*100:.1f}%)")
    
    return cc_df, course_details_df

def analyze_missing_course_details_by_meet_race(result_df, course_details_df, running_event_df, athlete_df):
    """Analyze how many unique meet-gender-race combinations don't have course details."""
    
    # Merge data to get complete picture
    merged_df = result_df.merge(
        athlete_df[['athlete_id', 'gender']], 
        on='athlete_id', 
        how='left'
    ).merge(
        running_event_df[['running_event_id', 'event_name']], 
        on='running_event_id', 
        how='left'
    )
    
    # Filter for cross country events
    cross_country_events = ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']
    cc_df = merged_df[merged_df['event_name'].isin(cross_country_events)].copy()
    
    # Get unique meet-gender-race combinations
    unique_meet_race = cc_df.groupby(['meet_id', 'gender', 'event_name', 'running_event_id']).size().reset_index()
    unique_meet_race = unique_meet_race[['meet_id', 'gender', 'event_name', 'running_event_id']]
    
    # Create course details key
    course_details_df['course_match_key'] = course_details_df.apply(
        lambda row: f"{row['meet_id']}_{row['running_event_id']}_{row['gender']}", 
        axis=1
    )
    
    # Create key for unique meet-race combinations
    unique_meet_race['course_match_key'] = unique_meet_race.apply(
        lambda row: f"{row['meet_id']}_{row['running_event_id']}_{row['gender']}", 
        axis=1
    )
    
    # Check which meet-race combinations have course details
    combinations_with_course_details = unique_meet_race['course_match_key'].isin(course_details_df['course_match_key'])
    
    # Calculate statistics
    total_combinations = len(unique_meet_race)
    combinations_with_details = combinations_with_course_details.sum()
    combinations_without_details = total_combinations - combinations_with_details
    
    print("\n=== MISSING COURSE DETAILS BY MEET-RACE COMBINATION ===")
    print(f"Total unique meet-gender-race combinations: {total_combinations:,}")
    print(f"Combinations with course details: {combinations_with_details:,} ({combinations_with_details/total_combinations*100:.1f}%)")
    print(f"Combinations without course details: {combinations_without_details:,} ({combinations_without_details/total_combinations*100:.1f}%)")
    
    # Show combinations without course details
    combinations_without_details_df = unique_meet_race[~combinations_with_course_details]
    
    print(f"\n--- Meet-Race Combinations Without Course Details ---")
    print("First 20 combinations without course details:")
    for idx, row in combinations_without_details_df.head(20).iterrows():
        print(f"  Meet {row['meet_id']}, {row['event_name']}, {row['gender']}")
    
    # Break down by gender
    print(f"\n--- By Gender ---")
    for gender in ['M', 'F']:
        gender_combinations = unique_meet_race[unique_meet_race['gender'] == gender]
        gender_total = len(gender_combinations)
        gender_with_details = gender_combinations['course_match_key'].isin(course_details_df['course_match_key']).sum()
        gender_without_details = gender_total - gender_with_details
        
        print(f"{'Men' if gender == 'M' else 'Women'}:")
        print(f"  Total combinations: {gender_total:,}")
        print(f"  With course details: {gender_with_details:,} ({gender_with_details/gender_total*100:.1f}%)")
        print(f"  Without course details: {gender_without_details:,} ({gender_without_details/gender_total*100:.1f}%)")
    
    # Break down by distance
    print(f"\n--- By Distance ---")
    for distance in ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']:
        distance_combinations = unique_meet_race[unique_meet_race['event_name'] == distance]
        distance_total = len(distance_combinations)
        distance_with_details = distance_combinations['course_match_key'].isin(course_details_df['course_match_key']).sum()
        distance_without_details = distance_total - distance_with_details
        
        print(f"{distance}:")
        print(f"  Total combinations: {distance_total:,}")
        print(f"  With course details: {distance_with_details:,} ({distance_with_details/distance_total*100:.1f}%)")
        print(f"  Without course details: {distance_without_details:,} ({distance_without_details/distance_total*100:.1f}%)")
    
    return unique_meet_race, combinations_without_details_df

def analyze_distance_mismatches_by_event(result_df, running_event_df, athlete_df):
    """Analyze how many men's events aren't 8K and women's events aren't 6K."""
    
    # Merge data
    merged_df = result_df.merge(
        athlete_df[['athlete_id', 'gender']], 
        on='athlete_id', 
        how='left'
    ).merge(
        running_event_df[['running_event_id', 'event_name']], 
        on='running_event_id', 
        how='left'
    )
    
    # Filter for cross country events (6K and 8K, plus 5K for women)
    cross_country_events = ['6000m', '8000m', '5000m']
    cc_df = merged_df[merged_df['event_name'].isin(cross_country_events)].copy()
    
    # Get unique events
    unique_events = cc_df.groupby(['meet_id', 'running_event_id', 'gender', 'event_name']).size().reset_index()
    unique_events = unique_events[['meet_id', 'running_event_id', 'gender', 'event_name']]
    
    # Define expected distances (all cross country distances are valid)
    expected_distances = {
        'M': ['5000m', '6000m', '8000m', '4 Mile', '5 Mile'],  # Men can run any cross country distance
        'F': ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']   # Women can run any cross country distance
    }
    
    # Check for mismatches
    mismatches = []
    for gender in ['M', 'F']:
        gender_events = unique_events[unique_events['gender'] == gender]
        expected_distance = expected_distances[gender]
        
        # Count events that are not in the expected cross country distances
        mismatched_events = gender_events[~gender_events['event_name'].isin(expected_distance)]
        
        total_events = len(gender_events)
        mismatched_count = len(mismatched_events)
        
        mismatches.append({
            'gender': gender,
            'expected_distance': expected_distance,
            'total_events': total_events,
            'mismatched_events': mismatched_count,
            'mismatch_percentage': mismatched_count / total_events * 100 if total_events > 0 else 0
        })
    
    print("\n=== DISTANCE MISMATCH ANALYSIS BY EVENT ===")
    for mismatch in mismatches:
        gender_name = "Men" if mismatch['gender'] == 'M' else "Women"
        expected = mismatch['expected_distance']
        expected_str = "any cross country distance"
        print(f"{gender_name} (expected {expected_str}):")
        print(f"  Total events: {mismatch['total_events']:,}")
        print(f"  Events not {expected_str}: {mismatch['mismatched_events']:,}")
        print(f"  Mismatch percentage: {mismatch['mismatch_percentage']:.1f}%")
    
    return mismatches

def analyze_distance_mismatches_by_meet(result_df, running_event_df, athlete_df):
    """Analyze how many meets have men not running 8K and women not running 6K."""
    
    # Merge data
    merged_df = result_df.merge(
        athlete_df[['athlete_id', 'gender']], 
        on='athlete_id', 
        how='left'
    ).merge(
        running_event_df[['running_event_id', 'event_name']], 
        on='running_event_id', 
        how='left'
    )
    
    # Filter for cross country events
    cross_country_events = ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']
    cc_df = merged_df[merged_df['event_name'].isin(cross_country_events)].copy()
    
    # Get unique meet-gender combinations
    unique_meet_gender = cc_df.groupby(['meet_id', 'gender', 'event_name']).size().reset_index()
    unique_meet_gender = unique_meet_gender[['meet_id', 'gender', 'event_name']]
    
    # Analyze by meet
    print("\n=== DISTANCE MISMATCH ANALYSIS BY MEET ===")
    
    # For men - check meets where they don't run 8K
    men_meets = unique_meet_gender[unique_meet_gender['gender'] == 'M']
    total_men_meets = men_meets['meet_id'].nunique()
    men_meets_not_8k = men_meets[men_meets['event_name'] != '8000m']['meet_id'].nunique()
    
    print(f"Men's meets:")
    print(f"  Total meets with men: {total_men_meets}")
    print(f"  Meets where men don't run 8K: {men_meets_not_8k}")
    print(f"  Percentage: {men_meets_not_8k/total_men_meets*100:.1f}%")
    
    # Show which meets have men not running 8K
    men_not_8k_meets = men_meets[men_meets['event_name'] != '8000m']
    print(f"\n  Meets where men don't run 8K:")
    for meet_id in men_not_8k_meets['meet_id'].unique():
        meet_events = men_not_8k_meets[men_not_8k_meets['meet_id'] == meet_id]
        events_str = ', '.join(meet_events['event_name'].tolist())
        print(f"    Meet {meet_id}: {events_str}")
    
    # For women - check meets where they don't run 6K
    women_meets = unique_meet_gender[unique_meet_gender['gender'] == 'F']
    total_women_meets = women_meets['meet_id'].nunique()
    women_meets_not_6k = women_meets[women_meets['event_name'] != '6000m']['meet_id'].nunique()
    
    print(f"\nWomen's meets:")
    print(f"  Total meets with women: {total_women_meets}")
    print(f"  Meets where women don't run 6K: {women_meets_not_6k}")
    print(f"  Percentage: {women_meets_not_6k/total_women_meets*100:.1f}%")
    
    # Show which meets have women not running 6K
    women_not_6k_meets = women_meets[women_meets['event_name'] != '6000m']
    print(f"\n  Meets where women don't run 6K:")
    for meet_id in women_not_6k_meets['meet_id'].unique():
        meet_events = women_not_6k_meets[women_not_6k_meets['meet_id'] == meet_id]
        events_str = ', '.join(meet_events['event_name'].tolist())
        print(f"    Meet {meet_id}: {events_str}")
    
    return men_not_8k_meets, women_not_6k_meets

def analyze_course_details_coverage(course_details_df):
    """Analyze what course details are available."""
    
    print("\n=== COURSE DETAILS COVERAGE ===")
    total_course_details = len(course_details_df)
    
    # Check which fields have data
    fields_to_check = [
        'elevation_gain', 'elevation_loss', 'estimated_course_distance',
        'temperature', 'dew_point', 'humidity', 'weather_conditions'
    ]
    
    for field in fields_to_check:
        if field in course_details_df.columns:
            non_null_count = course_details_df[field].notna().sum()
            percentage = non_null_count / total_course_details * 100
            print(f"{field}: {non_null_count:,}/{total_course_details:,} ({percentage:.1f}%)")
        else:
            print(f"{field}: Not available in dataset")

def analyze_non_standard_races_by_gender(result_df, running_event_df, athlete_df):
    """Analyze what percentage of meet-race-event combinations by gender are not standard distances (6K women, 8K men)."""
    
    # Merge data to get complete picture
    merged_df = result_df.merge(
        athlete_df[['athlete_id', 'gender']], 
        on='athlete_id', 
        how='left'
    ).merge(
        running_event_df[['running_event_id', 'event_name']], 
        on='running_event_id', 
        how='left'
    )
    
    # Filter for cross country events
    cross_country_events = ['5000m', '6000m', '8000m', '4 Mile', '5 Mile']
    cc_df = merged_df[merged_df['event_name'].isin(cross_country_events)].copy()
    
    # Get unique meet-gender-event combinations
    unique_meet_events = cc_df.groupby(['meet_id', 'gender', 'event_name']).size().reset_index()
    unique_meet_events = unique_meet_events[['meet_id', 'gender', 'event_name']]
    
    # Define standard distances
    standard_distances = {
        'M': ['8000m'],  # Standard for men
        'F': ['6000m']   # Standard for women
    }
    
    print("\n=== NON-STANDARD MEET-EVENT COMBINATIONS BY GENDER ===")
    
    for gender in ['M', 'F']:
        gender_events = unique_meet_events[unique_meet_events['gender'] == gender]
        total_events = len(gender_events)
        
        # Count events that are NOT the standard distance
        standard_distance_events = gender_events[gender_events['event_name'].isin(standard_distances[gender])]
        non_standard_events = total_events - len(standard_distance_events)
        
        print(f"\n{('Men' if gender == 'M' else 'Women')} (Standard: {standard_distances[gender][0]}):")
        print(f"  Total unique meet-event combinations: {total_events:,}")
        print(f"  Standard distance events: {len(standard_distance_events):,} ({len(standard_distance_events)/total_events*100:.1f}%)")
        print(f"  Non-standard distance events: {non_standard_events:,} ({non_standard_events/total_events*100:.1f}%)")
        
        # Break down non-standard distances
        non_standard_df = gender_events[~gender_events['event_name'].isin(standard_distances[gender])]
        distance_counts = non_standard_df['event_name'].value_counts()
        
        print(f"  Non-standard distances breakdown:")
        for distance, count in distance_counts.items():
            print(f"    {distance}: {count:,} ({count/total_events*100:.1f}%)")
    
    return unique_meet_events

def main():
    """Main analysis function."""
    print("Loading data...")
    result_df, course_details_df, running_event_df, athlete_df, meet_df = load_data()
    
    print(f"Loaded {len(result_df):,} results")
    print(f"Loaded {len(course_details_df):,} course details")
    print(f"Loaded {len(running_event_df):,} running events")
    print(f"Loaded {len(athlete_df):,} athletes")
    print(f"Loaded {len(meet_df):,} meets")
    
    # Analyze missing course details by individual race
    cc_df, course_details_df = analyze_missing_course_details_by_race(
        result_df, course_details_df, running_event_df, athlete_df
    )
    
    # Analyze missing course details by event
    unique_events, course_details_df = analyze_missing_course_details_by_event(
        result_df, course_details_df, running_event_df, athlete_df
    )
    
    # Analyze missing course details by meet-race combination
    unique_meet_race, combinations_without_details = analyze_missing_course_details_by_meet_race(
        result_df, course_details_df, running_event_df, athlete_df
    )
    
    # Analyze non-standard races by gender
    cc_df = analyze_non_standard_races_by_gender(result_df, running_event_df, athlete_df)
    
    # Analyze distance mismatches by event
    mismatches = analyze_distance_mismatches_by_event(result_df, running_event_df, athlete_df)
    
    # Analyze distance mismatches by meet
    men_not_8k, women_not_6k = analyze_distance_mismatches_by_meet(result_df, running_event_df, athlete_df)
    
    # Analyze course details coverage
    analyze_course_details_coverage(course_details_df)
    
    print("\n=== SUMMARY ===")
    print("This analysis shows:")
    print("1. How many individual race results are missing course details")
    print("2. How many unique meet-event-gender combinations are missing course details")
    print("3. How many unique meet-gender-race combinations are missing course details")
    print("4. What percentage of races by gender are not standard distances (6K women, 8K men)")
    print("5. How many men's events aren't 8K and women's events aren't 6K")
    print("6. How many meets have men not running 8K and women not running 6K")
    print("7. What course detail information is available in the dataset")

if __name__ == "__main__":
    main() 