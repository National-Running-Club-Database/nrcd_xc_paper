import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import r2_score
import scipy.cluster.hierarchy as sch
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from IPython.display import display
from datetime import timedelta
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

def load_data():
    """Load all CSV files from the data directory with error handling."""
    directory_path = '../data/nrcd/data'
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
        meet_df = pd.read_csv(os.path.join(directory_path, 'meet_regular_season.csv'))
        result_df = pd.read_csv(os.path.join(directory_path, 'result.csv'))
        course_details_df = pd.read_csv(os.path.join(directory_path, 'course_details.csv'))
        athlete_team_association_df = pd.read_csv(os.path.join(directory_path, 'athlete_team_association.csv'))
        
        # Convert date columns to datetime with error handling
        meet_df['start_date'] = pd.to_datetime(meet_df['start_date'], format='%Y-%m-%d', errors='coerce')
        meet_df['end_date'] = pd.to_datetime(meet_df['end_date'], format='%Y-%m-%d', errors='coerce')
        
        removedMeetIDs = [61, 62, 63, 64, 65, 66, 71, 675, 676, 769, 774, 776, 770, 822]
        meet_df = meet_df[meet_df["regionals"] == False]
        meet_df = meet_df[meet_df["nationals"] == False]
        meet_df = meet_df.reset_index(drop=True)
        # Validate data
        if meet_df['start_date'].isna().any():
            print("Warning: Some start dates could not be parsed")
        if meet_df['end_date'].isna().any():
            print("Warning: Some end dates could not be parsed")

        for id in removedMeetIDs:
            result_df = result_df[result_df["meet_id"] != id]
            result_df = result_df.reset_index(drop=True)
        
        return (team_df, athlete_df, sport_df, running_event_df, meet_df, 
                result_df, course_details_df, athlete_team_association_df)
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# Load the data
try:
    team_df, athlete_df, sport_df, running_event_df, meet_df, result_df, course_details_df, athlete_team_association_df = load_data()
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error: {str(e)}")
    raise

def merge_dataframes(athlete_df, result_df, meet_df):
    """Merge the dataframes for analysis."""
    # Merge athlete and result data
    athlete_result_df = pd.merge(athlete_df, result_df, on='athlete_id')
    
    # Merge the result with meet data
    full_df = pd.merge(athlete_result_df, meet_df, on='meet_id')
    
    return full_df

# Merge dataframes
full_df = merge_dataframes(athlete_df, result_df, meet_df)
print(full_df[full_df["last_name"] == "Fryer"])
# Analyze gender distribution
gender_counts = athlete_df['gender'].value_counts()
print(f"Number of men: {gender_counts.get('M', 0)}")
print(f"Number of women: {gender_counts.get('F', 0)}")

def analyze_cross_country_performance(sport_df, meet_df, result_df, athlete_df, running_event_df):
    """Analyze cross country performance data."""
    # Get cross country sport ID
    cross_country_sport_id = sport_df[sport_df['sport_name'].str.contains('Cross Country', case=False, na=False)]['sport_id'].values[0]
    
    # Define date range
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2025, 5, 1)
    
    # Filter meets by date range and sport
    cross_country_meets = meet_df[
        (meet_df['sport_id'] == cross_country_sport_id) &
        (meet_df['start_date'] >= start_date) &
        (meet_df['start_date'] <= end_date)
    ]
    
    # Filter results for cross country meets
    cross_country_results = result_df[result_df['meet_id'].isin(cross_country_meets['meet_id'])]
    
    # Merge with athlete data
    cross_country_athletes = pd.merge(cross_country_results, athlete_df, on='athlete_id')
    print("test")
    print(cross_country_athletes[cross_country_athletes["last_name"] == "Fryer"].result_time)
    # Merge with running event data to get event names
    cross_country_athletes = pd.merge(cross_country_athletes, running_event_df[['running_event_id', 'event_name']], 
                                    on='running_event_id', how='left')
    
    # Merge with meet data to get start_date
    cross_country_athletes = pd.merge(cross_country_athletes, cross_country_meets[['meet_id', 'start_date']], 
                                    on='meet_id', how='left')
    
    return cross_country_athletes

# Analyze cross country performance
cross_country_athletes = analyze_cross_country_performance(sport_df, meet_df, result_df, athlete_df, running_event_df)
print("So Cool!")
print(cross_country_athletes[cross_country_athletes["last_name"] == "Fryer"].result_time)

def parse_time(time_str):
    """Convert time string to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        return float(time_str)

def format_parsed_time(seconds):
    """Convert seconds to formatted time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:05.2f}"
    elif minutes > 0:
        return f"{minutes}:{seconds:05.2f}"
    else:
        return f"{seconds:05.2f}"

def adjust_time_for_race(event_name, time, course_details, gender):
    """Adjust race time based on course details and conditions."""
    event_dist = None
    if event_name.endswith('m'):
        event_dist = float(event_name.replace('m', '').strip())
    elif event_name.endswith('mi'):
        event_dist = float(event_name.replace('mi', '').strip()) * 1609.34
    
    hill_slow_time = 0
    hill_speed_time = 0
    
    if event_dist:
        if pd.notna(course_details.get('elevation_gain')) or pd.notna(course_details.get('elevation_loss')):
            if pd.notna(course_details.get('elevation_gain')):
                grade_increase_percentage = course_details['elevation_gain'] / event_dist * 100
                hill_slow_time = time * 0.04 * grade_increase_percentage
            if pd.notna(course_details.get('elevation_loss')):
                grade_decrease_percentage = course_details['elevation_loss'] / event_dist * 100
                hill_speed_time = time * 0.0267 * grade_decrease_percentage
            time += hill_speed_time - hill_slow_time

        if pd.notna(course_details.get('estimated_course_distance')):
            factor = event_dist / course_details['estimated_course_distance']
            if gender == 'F':
                time *= factor ** 1.055
            elif gender == 'M':
                time *= factor ** 1.08

    if pd.notna(course_details.get('temperature')) and pd.notna(course_details.get('dew_point')):
        weather_factor = course_details['temperature'] + course_details['dew_point']
        if weather_factor > 100:
            percent_increase = 0.0015 * (weather_factor - 100) ** 2
            time *= 1 - (percent_increase / 100)

    return time

def normalize_and_adjust_time(row):
    """Normalize and adjust time for a given row of data."""
    parsed_time = parse_time(row['result_time'])
    
    if pd.notna(row.get('course_details_id')):
        course_details = {
            'elevation_gain': row.get('elevation_gain'),
            'elevation_loss': row.get('elevation_loss'),
            'estimated_course_distance': row.get('estimated_course_distance'),
            'temperature': row.get('temperature'),
            'dew_point': row.get('dew_point')
        }
        adjusted_time = adjust_time_for_race(row['event_name'], parsed_time, course_details, row['gender'])
    else:
        adjusted_time = parsed_time

    event_dist = None
    if row['event_name'].endswith('m'):
        event_dist = float(row['event_name'].replace('m', '').strip())
    elif row['event_name'].endswith('mi'):
        event_dist = float(row['event_name'].replace('mi', '').strip()) * 1609.34

    if event_dist:
        if row['gender'] == 'F':
            normalized_time = adjusted_time * (6000 / event_dist)**1.08
        elif row['gender'] == 'M':
            normalized_time = adjusted_time * (8000 / event_dist)**1.055
        else:
            normalized_time = adjusted_time
    else:
        normalized_time = adjusted_time
        
    return format_parsed_time(normalized_time)

# Process and normalize times
cross_country_athletes['final_adjusted_time'] = cross_country_athletes.apply(normalize_and_adjust_time, axis=1)
cross_country_athletes['final_parsed_adjusted_time'] = cross_country_athletes['final_adjusted_time'].apply(parse_time)

# def plot_performance_over_time(data, gender):
#     """Plot performance over time for a specific gender."""
#     plt.figure(figsize=(12, 6))
#     sns.lineplot(data=data, x='start_date', y='final_parsed_adjusted_time', 
#                 hue='athlete_id', marker='o', palette='viridis', legend=None)
#     plt.title(f'Athlete Performance Over Season ({gender}-{"8K" if gender == "Men" else "6K"})')
#     plt.xlabel('Date')
#     plt.ylabel('Adjusted Time (seconds)')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show(block=False)

# Separate male and female athletes
male_athletes = cross_country_athletes[cross_country_athletes['gender'] == 'M']
female_athletes = cross_country_athletes[cross_country_athletes['gender'] == 'F']

# Plot performance over time
# plot_performance_over_time(male_athletes, 'Men')
# plot_performance_over_time(female_athletes, 'Women')

def calculate_improvement_rate(data):
    """Calculate improvement rate using linear regression."""
    slopes = []
    for athlete_id in data['athlete_id'].unique():
        athlete_df = data[data['athlete_id'] == athlete_id].sort_values('start_date')
        if len(athlete_df) > 1:
            X = np.array(athlete_df['start_date'].map(pd.Timestamp.toordinal)).reshape(-1, 1)
            y = athlete_df['final_parsed_adjusted_time'].values
            
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            slopes.append(slope)
    
    return np.mean(slopes)

# Calculate improvement rates
male_improvement = calculate_improvement_rate(male_athletes)
female_improvement = calculate_improvement_rate(female_athletes)

print(f"Average Improvement Rate (Men): {male_improvement:.2f} seconds per day")
print(f"Average Improvement Rate (Women): {female_improvement:.2f} seconds per day")

# Map full state names and abbreviations to abbreviations
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}
# Create a new dictionary with reverse mappings and common misspellings
state_abbreviations_with_reverse = state_abbreviations.copy()
for abbr in state_abbreviations.values():
    state_abbreviations_with_reverse[abbr] = abbr
# Add common misspellings
state_abbreviations_with_reverse.update({
    'Pennslyvania': 'PA',
    'Pennsylania': 'PA',
    'Masssachusetts': 'MA',
    'Wiaconsin': 'WI',
    'Viriginia': 'VA',
    'Flordia': 'FL',
    'D.C.': 'DC',
    'DC': 'DC'
})

def normalize_state(state):
    """Normalize state names to their two-letter abbreviations."""
    state = str(state).strip()
    # Handle special cases
    if state.lower() in ['hill', 'university of florida']:
        return None
    # Try direct lookup first
    if state in state_abbreviations_with_reverse:
        return state_abbreviations_with_reverse[state]
    # Try case-insensitive lookup
    state_lower = state.lower()
    for key in state_abbreviations_with_reverse:
        if key.lower() == state_lower:
            return state_abbreviations_with_reverse[key]
    # If no match found, return None
    return None

# Apply normalization and filter out None values
meet_df['normalized_state'] = meet_df['meet_state'].apply(normalize_state)
meet_df = meet_df.dropna(subset=['normalized_state'])

# def create_state_choropleth(meet_df):
#     """Create a choropleth map of meets by state."""
#     results_by_state = meet_df['normalized_state'].value_counts().to_dict()
#     state_data = pd.DataFrame(list(results_by_state.items()),
#                             columns=['State', 'Results'])
#     state_data['text'] = state_data['Results'].astype(str)
#     fig = go.Figure(data=go.Choropleth(
#         locations=state_data['State'],
#         z=state_data['Results'],
#         locationmode='USA-states',
#         colorscale='Viridis',
#         colorbar_title="Number of Meets",
#         text=state_data['text'],
#         hovertemplate='<b>%{location}</b><br>Meets: %{z}<extra></extra>'
#     ))
#     fig.update_layout(
#         title_text='Number of NRCD Meets by State',
#         geo=dict(
#             scope='usa',
#             projection=go.layout.geo.Projection(type='albers usa'),
#         ),
#     )
#     fig.show()

# # Create state choropleth
# create_state_choropleth(meet_df)

def analyze_gender_distribution_by_year(sport_df, meet_df, result_df, athlete_df, year):
    """Analyze gender distribution for a specific year."""
    # Get cross country sport ID
    cross_country_sport_id = sport_df[sport_df['sport_name'].str.contains('Cross Country', case=False, na=False)]['sport_id'].values[0]
    
    # Define date range for the year
    start_date = datetime(year, 8, 1)
    end_date = datetime(year + 1, 5, 1)
    
    # Filter meets by date range and sport
    cross_country_meets = meet_df[
        (meet_df['sport_id'] == cross_country_sport_id) &
        (meet_df['start_date'] >= start_date) &
        (meet_df['start_date'] <= end_date)
    ]
    
    # Filter results for cross country meets
    cross_country_results = result_df[result_df['meet_id'].isin(cross_country_meets['meet_id'])]
    
    # Merge with athlete data
    cross_country_athletes = pd.merge(cross_country_results, athlete_df, on='athlete_id')
    
    # Count unique athletes by gender
    gender_counts = cross_country_athletes['gender'].value_counts()
    
    # Calculate meets per athlete by gender
    meets_per_athlete = cross_country_athletes.groupby(['athlete_id', 'gender'])['meet_id'].nunique()
    meets_by_gender = meets_per_athlete.groupby('gender')
    
    # Perform t-test
    male_meets = meets_per_athlete[meets_per_athlete.index.get_level_values('gender') == 'M']
    female_meets = meets_per_athlete[meets_per_athlete.index.get_level_values('gender') == 'F']
    t_stat, p_value = stats.ttest_ind(male_meets, female_meets)
    
    return gender_counts, meets_by_gender, t_stat, p_value

# Analyze gender distribution for 2024 and 2023
for year in [2024, 2023]:
    print(f"\n{year} Cross Country: Gender vs # of Races")
    gender_counts, meets_by_gender, t_stat, p_value = analyze_gender_distribution_by_year(sport_df, meet_df, result_df, athlete_df, year)
    
    print(f"Number of men: {gender_counts.get('M', 0)}")
    print(f"Number of women: {gender_counts.get('F', 0)}")
    
    print("\nDistribution of number of Cross Country meets for men:")
    print(meets_by_gender.get_group('M').describe())
    
    print("\nDistribution of number of Cross Country meets for women:")
    print(meets_by_gender.get_group('F').describe())
    
    print(f"\nT-statistic: {t_stat}")
    print(f"P-value: {p_value}")
    
    # Create visualization
    # plt.figure(figsize=(12, 6))
    
    # Plot gender distribution
    # plt.subplot(1, 2, 1)
    # gender_counts.plot(kind='bar')
    # plt.title(f'Gender Distribution ({year})')
    # plt.xlabel('Gender')
    # plt.ylabel('Number of Athletes')
    
    # Plot meets per athlete distribution
    # plt.subplot(1, 2, 2)
    # meets_by_gender.plot(kind='box')
    # plt.title(f'Meets per Athlete by Gender ({year})')
    # plt.xlabel('Gender')
    # plt.ylabel('Number of Meets')
    
    # plt.tight_layout()
    # plt.show(block=False)

    # Calculate basic summary statistics
summary_data = {
    'Metric': ['Total Results', 'Total Meets', 'Total Course Details', 'Total Athletes'],
    'Count': [len(result_df), len(meet_df), len(course_details_df), len(athlete_df)]
}
summary_df = pd.DataFrame(summary_data)
print("Summary Statistics:")
display(summary_df.style.hide(axis='index'))

# Calculate results by gender, sport, and school year
merged_df = pd.merge(result_df, athlete_df[['athlete_id', 'gender']], on='athlete_id')
merged_df = pd.merge(merged_df, meet_df[['meet_id', 'start_date', 'sport_id']], on='meet_id')
merged_df = pd.merge(merged_df, sport_df[['sport_id', 'sport_name']], on='sport_id')

# Add school year column (starting August 1st)
merged_df['school_year'] = merged_df['start_date'].apply(
    lambda x: f"{x.year}-{x.year + 1}" if x.month >= 8 else f"{x.year - 1}-{x.year}"
)

# Create separate dataframes for each gender and total
male_results = merged_df[merged_df['gender'] == 'M'].groupby(['school_year', 'sport_name']).size().reset_index(name='M')
female_results = merged_df[merged_df['gender'] == 'F'].groupby(['school_year', 'sport_name']).size().reset_index(name='F')
total_results = merged_df.groupby(['school_year', 'sport_name']).size().reset_index(name='Total')

# Merge the results
results_by_year = pd.merge(male_results, female_results, on=['school_year', 'sport_name'], how='outer')
results_by_year = pd.merge(results_by_year, total_results, on=['school_year', 'sport_name'], how='outer')
results_by_year = results_by_year.fillna(0)

# Convert to integers
results_by_year['M'] = results_by_year['M'].astype(int)
results_by_year['F'] = results_by_year['F'].astype(int)
results_by_year['Total'] = results_by_year['Total'].astype(int)

# Create pivot tables for each metric
male_pivot = results_by_year.pivot(index='sport_name', columns='school_year', values='M').fillna(0).astype(int)
female_pivot = results_by_year.pivot(index='sport_name', columns='school_year', values='F').fillna(0).astype(int)
total_pivot = results_by_year.pivot(index='sport_name', columns='school_year', values='Total').fillna(0).astype(int)

# Add total columns
male_pivot['All Years'] = male_pivot.sum(axis=1)
female_pivot['All Years'] = female_pivot.sum(axis=1)
total_pivot['All Years'] = total_pivot.sum(axis=1)

# Add total rows
male_pivot.loc['All Sports'] = male_pivot.sum()
female_pivot.loc['All Sports'] = female_pivot.sum()
total_pivot.loc['All Sports'] = total_pivot.sum()

print("\nResults by Gender, Sport, and School Year:")
print("\nMale Results:")
display(male_pivot)
print("\nFemale Results:")
display(female_pivot)
print("\nTotal Results:")
display(total_pivot)

print(male_results.columns)
print(athlete_df.columns)
print(result_df.columns)
setOfEventTypes = set()
for i in range(0, len(result_df), 1):
    setOfEventTypes.add(result_df.loc[i, "event_type"])
print(setOfEventTypes)
print(result_df["event_type"])
print(min(result_df[result_df["athlete_id"] == 1].result_time))

print(cross_country_athletes.columns)
print("However")
print(cross_country_athletes[cross_country_athletes["last_name"] == "Fryer"].result_time)

jonathanKarr = cross_country_athletes[cross_country_athletes["athlete_id"] == 1]
print(jonathanKarr)
times = jonathanKarr[jonathanKarr["event_name"] == "8000m"]
print((min(times.result_time)))

# Trying to determine the difference in 8K's over time

def analyze_cross_country_performance2023(sport_df, meet_df, result_df, athlete_df, running_event_df):
    """Analyze cross country performance data."""
    # Get cross country sport ID
    cross_country_sport_id = sport_df[sport_df['sport_name'].str.contains('Cross Country', case=False, na=False)]['sport_id'].values[0]
    
    # Define date range
    start_date = datetime(2023, 8, 1)
    end_date = datetime(2024, 5, 1)
    
    # Filter meets by date range and sport
    cross_country_meets = meet_df[
        (meet_df['sport_id'] == cross_country_sport_id) &
        (meet_df['start_date'] >= start_date) &
        (meet_df['start_date'] <= end_date)
    ]
    
    # Filter results for cross country meets
    cross_country_results = result_df[result_df['meet_id'].isin(cross_country_meets['meet_id'])]
    
    # Merge with athlete data
    cross_country_athletes = pd.merge(cross_country_results, athlete_df, on='athlete_id')
    
    # Merge with running event data to get event names
    cross_country_athletes = pd.merge(cross_country_athletes, running_event_df[['running_event_id', 'event_name']], 
                                    on='running_event_id', how='left')
    
    # Merge with meet data to get start_date
    cross_country_athletes = pd.merge(cross_country_athletes, cross_country_meets[['meet_id', 'start_date']], 
                                    on='meet_id', how='left')
    
    return cross_country_athletes

XCAthletes2023 = analyze_cross_country_performance2023(sport_df, meet_df, result_df, athlete_df, running_event_df)
XCAthletes2024 = analyze_cross_country_performance(sport_df, meet_df, result_df, athlete_df, running_event_df)

# Setting up data frames for male and female athletes in 2023 / 2024
XCMales2023 = XCAthletes2023[XCAthletes2023["gender"] == "M"]
XCMales2023 = XCMales2023[XCMales2023["event_name"] == "8000m"]
XCMales2023 = XCMales2023.reset_index(drop=True)

XCFemales2023 = XCAthletes2023[XCAthletes2023["gender"] == "F"]
XCFemales2023 = XCFemales2023[XCFemales2023["event_name"] == "6000m"]
XCFemales2023 = XCFemales2023.reset_index(drop=True)

# 2024
XCMales2024 = XCAthletes2024[XCAthletes2024["gender"] == "M"]
XCMales2024 = XCMales2024[XCMales2024["event_name"] == "8000m"]
XCMales2024 = XCMales2024.reset_index(drop=True)

XCFemales2024 = XCAthletes2024[XCAthletes2024["gender"] == "F"]
XCFemales2024 = XCFemales2024[XCFemales2024["event_name"] == "6000m"]
XCFemales2024 = XCFemales2024.reset_index(drop=True)
# Filtering out times that are greater than 1 hour.

print(XCMales2023[XCMales2023["result_time"] < "25:20.70"].result_time)

eightKTimes = [datetime.strptime(argument, "%M:%S.%f").time() for argument in XCMales2023["result_time"] if len(argument) <= 8]
for time in eightKTimes:
    if time < datetime.strptime("25:00.00", "%M:%S.%f").time():
        # print(time)
        pass

print(XCMales2023.result_time)
print(XCMales2023)
#These athlete ids are from people that have at least one result over 2023 and 2024 XC seasons.
mens_athlete_id_2023_2025 = set()
mens_athlete_id_2023 = set()
mens_athlete_id_2024 = set()
for i in range(0, len(XCMales2023), 1):
    mens_athlete_id_2023.add(XCMales2023.loc[i, "athlete_id"])
for i in range(0, len(XCMales2024), 1):
    mens_athlete_id_2024.add(XCMales2024.loc[i, "athlete_id"])
for athlete_id in mens_athlete_id_2023:
    if athlete_id in mens_athlete_id_2024:
        mens_athlete_id_2023_2025.add(athlete_id)
# Use .loc here to get the whole row into the new variable
# for i in range(0, len(XCMales2023), 1):
#     if len(XCMales2023.loc[i, "result_time"]) > 0:
#         XCMales2023MultipleRaces = pd.concat([XCMales2023MultipleRaces, XCMales2023.loc[i]], ignore_index=True)
# for i in range(0, len(XCMales2024), 1):
#     if len(XCMales2024.loc[i, "result_time"]) > 0:
#         XCMales2024MultipleRaces = pd.concat([XCMales2024MultipleRaces, XCMales2024.loc[i]], ignore_index=True)
# XCMales2023MultipleRaces = XCMales2023[len(XCMales2023["result_time"]) > 0]
# XCMales2024MultipleRaces = XCMales2024[len(XCMales2024["result_time"]) > 0]
# for athlete_number in XCMales2023MultipleRaces.athlete_id:
#     if athlete_number in XCMales2024MultipleRaces.athlete_id:
#         mens_athlete_id_2023_2025.add(athlete_number)

print(len(mens_athlete_id_2023_2025))
print(len(XCMales2023.athlete_id))
print(len(XCMales2024.athlete_id))

positiveDeltaCounts = 0
negativeDelatCounts = 0
# print(XCMales2024[XCMales2024["last_name"] == "Fryer"].result_time)
# print(XCMales2024[XCMales2024["athlete_id"] == 3])
mens_net_sum = timedelta()
womens_net_sum = timedelta()
positiveNetSumMen = timedelta()
negativeNetSumMen = timedelta()
positiveNetSumWomen = timedelta()
negativeNetSumWomen = timedelta()
for id_number in mens_athlete_id_2023_2025:
    check = True
    difference = None
    # print((XCMales2023[XCMales2023["athlete_id"] == id_number]).result_time)
    # print("2024:")
    # print((XCMales2024[XCMales2024["athlete_id"] == id_number]).result_time)
    dataFrame1 = (XCMales2023[XCMales2023["athlete_id"] == id_number]).result_time
    dataFrame2 = (XCMales2024[XCMales2024["athlete_id"] == id_number]).result_time
    for item in dataFrame1:
        # print("Time 1:")
        # print(item)
        if len(item) > 8:
            check = False
    # print(XCMales2024)
    # print(dataFrame2)
    for item in dataFrame2: 
        # print("Time 2:")
        # print(item)
        if len(item) > 8:
            check = False
    if len(dataFrame1) == 0 or len(dataFrame2) == 0:
        check = False
    if check:
        time2023 = datetime.strptime(min((XCMales2023[XCMales2023["athlete_id"] == id_number]).result_time), "%M:%S.%f")
        time2024 = datetime.strptime(min((XCMales2024[XCMales2024["athlete_id"] == id_number]).result_time), "%M:%S.%f")

    # differenceMinutes = time2024.minute - time2023.minute
    # differenceSeconds = time2024.second - time2023
        difference = time2024 - time2023
        if difference > timedelta(0, 0):
            positiveDeltaCounts += 1
            positiveNetSumMen += difference
        if difference < timedelta(0, 0):
            negativeDelatCounts += 1
            negativeNetSumMen += difference
        else:
            print("Holy cow!")

    if difference:
        # print("Difference:")
        print(difference)
        mens_net_sum += difference
        # print("Net Sum: " + str(mens_net_sum))
        # print(mens_net_sum / len(mens_athlete_id_2023_2025))
        # print(len(mens_athlete_id_2023_2025))
    else:
        print("Nuh unh")


print("Men's calculations: ", end="\n\n")
print(len(mens_athlete_id_2023))
print(len(mens_athlete_id_2024))
print(len(mens_athlete_id_2023_2025))
print("Positive Delta Counts : " + str(positiveDeltaCounts))
print("Negative Delta Counts : " + str(negativeDelatCounts))
print("Positive Net Sum Average: " + str(positiveNetSumMen / len(mens_athlete_id_2023_2025)))
print("Negative Net Sum Average : " + str(negativeNetSumMen / len(mens_athlete_id_2023_2025)))

# Women's calculation

womens_athlete_id_2023 = set()
womens_athlete_id_2024 = set()
womens_athlete_id_2023_2025 = set()

for i in range(0, len(XCFemales2023), 1):
    womens_athlete_id_2023.add(XCFemales2023.loc[i, "athlete_id"])

for i in range(0, len(XCFemales2024), 1):
    womens_athlete_id_2024.add(XCFemales2024.loc[i, "athlete_id"])

for athlete_id in womens_athlete_id_2023:
    if athlete_id in womens_athlete_id_2024:
        womens_athlete_id_2023_2025.add(athlete_id)

# print(len(womens_athlete_id_2023))
# print(len(womens_athlete_id_2024))
# print(len(womens_athlete_id_2023_2025))

# Perform data preprocessing on women's results. Will still check for result times greater than 1 hour just in case, probably shouldn't be many though
womensPositiveNetCounts = 0
womensNegativeNetCounts = 0
for athlete_id_number in womens_athlete_id_2023_2025:
    check = True
    difference = None
    # print((XCFemales2023[XCFemales2023["athlete_id"] == athlete_id_number]).result_time)
    # print("2024:")
    # print((XCFemales2024[XCFemales2024["athlete_id"] == athlete_id_number]).result_time)
    dataFrame1 = (XCFemales2023[XCFemales2023["athlete_id"] == athlete_id_number]).result_time
    dataFrame2 = (XCFemales2024[XCFemales2024["athlete_id"] == athlete_id_number]).result_time
    for item in dataFrame1:
        # print("Time 1:")
        # print(item)
        if len(item) > 8:
            check = False
    # print(XCMales2024)
    # print(dataFrame2)
    for item in dataFrame2: 
        # print("Time 2:")
        # print(item)
        if len(item) > 8:
            check = False
    if len(dataFrame1) == 0 or len(dataFrame2) == 0:
        check = False
    if check:
        time2023 = datetime.strptime(min((XCFemales2023[XCFemales2023["athlete_id"] == athlete_id_number]).result_time), "%M:%S.%f")
        time2024 = datetime.strptime(min((XCFemales2024[XCFemales2024["athlete_id"] == athlete_id_number]).result_time), "%M:%S.%f")

        difference = time2024 - time2023
        if (difference > timedelta(0, 0)):
            womensPositiveNetCounts += 1
            positiveNetSumWomen += difference
            womens_net_sum += difference
            # print("Difference : " + str(difference))
        
        if difference < timedelta(0, 0):
            womensNegativeNetCounts += 1
            negativeNetSumWomen += difference
            womens_net_sum += difference
            # print("Difference :" + str(difference))

print("Women's numbers: ", end="\n\n")
print("Net Sum Average: " + str(womens_net_sum / len(womens_athlete_id_2023_2025)))
print(womensPositiveNetCounts)
print("Positive Net Sum Average: " + str(positiveNetSumWomen / len(womens_athlete_id_2023_2025)))
print(womensNegativeNetCounts)
print("Negative Net Sum Averaage:" + str(negativeNetSumWomen / len(womens_athlete_id_2023_2025)))

print(len(womens_athlete_id_2023))
print(len(womens_athlete_id_2024))
print(len(womens_athlete_id_2023_2025))
# print(mens_athlete_id_2023)
# print("\n\n")
# print(mens_athlete_id_2024)
# print(result_df.columns)
# print(XCMales2024)