import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all CSV files from the data directory with error handling."""
    directory_path = 'data/nrcd/data'
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

def get_race_results_by_state_by_year(meet_df, result_df, year):
    """Get race results by state for a specific year."""
    # Define date range for the year
    start_date = datetime(year, 8, 1)
    end_date = datetime(year + 1, 5, 1)
    
    # Filter meets by date range
    year_meets = meet_df[
        (meet_df['start_date'] >= start_date) &
        (meet_df['start_date'] <= end_date)
    ]
    
    # Filter results for meets in this year
    year_results = result_df[result_df['meet_id'].isin(year_meets['meet_id'])]
    
    # Merge with meet data to get state information
    results_with_state = pd.merge(year_results, year_meets[['meet_id', 'normalized_state']], on='meet_id')
    
    # Count results by state
    results_by_state = results_with_state['normalized_state'].value_counts().to_dict()
    
    return results_by_state

def create_state_choropleth_map(results_by_state_2023, results_by_state_2024):
    """Create a choropleth map of race results by state for 2023 and 2024 combined."""
    
    # Create DataFrames for both years
    state_data_2023 = pd.DataFrame(list(results_by_state_2023.items()),
                                  columns=['State', 'Results_2023'])
    state_data_2024 = pd.DataFrame(list(results_by_state_2024.items()),
                                  columns=['State', 'Results_2024'])
    
    # Create a complete list of all US states and DC
    all_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    ]
    
    # Create complete state data with all states
    complete_state_data = pd.DataFrame({'State': all_states})
    
    # Merge with actual results data
    state_data = pd.merge(complete_state_data, state_data_2023, on='State', how='left').fillna(0)
    state_data = pd.merge(state_data, state_data_2024, on='State', how='left').fillna(0)
    
    # Convert to integers
    state_data['Results_2023'] = state_data['Results_2023'].astype(int)
    state_data['Results_2024'] = state_data['Results_2024'].astype(int)
    
    # Calculate total results for both years
    state_data['Total_Results'] = state_data['Results_2023'] + state_data['Results_2024']
    
    # Create text for hover showing both years
    state_data['text'] = state_data.apply(
        lambda row: f"2023: {row['Results_2023']}<br>2024: {row['Results_2024']}<br>Total: {row['Total_Results']}", 
        axis=1
    )
    
    # Create single choropleth with combined data
    fig = go.Figure(data=go.Choropleth(
        locations=state_data['State'],
        z=state_data['Total_Results'],
        locationmode='USA-states',
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text="Total<br>Race Results",
                font=dict(size=28, color='black')
            ),
            tickfont=dict(size=28, color='black'),
            thickness=30,
            len=0.8
        ),
        text=state_data['text'],
        hovertemplate='<b>%{location}</b><br>%{text}<extra></extra>',
        zmid=0,
        zmin=0,
        zmax=state_data['Total_Results'].max()
    ))
    
    # Add gray overlay for states with zero results
    zero_states = state_data[state_data['Total_Results'] == 0]['State'].tolist()
    if zero_states:
        fig.add_trace(go.Choropleth(
            locations=zero_states,
            z=[0] * len(zero_states),
            locationmode='USA-states',
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            showscale=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>Number of Cross Country Race Results (2023-2024)</b>',
            x=0.5,
            xanchor='center',
            y=0.98,
            font=dict(size=28, color='black')
        ),
        geo=dict(
            scope='usa',
            projection=go.layout.geo.Projection(type='albers usa'),
            showland=False,
            showocean=False,
            showlakes=False,
            showrivers=False,
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

# Get race results by state for 2023 and 2024
print("Calculating race results by state for 2023...")
results_by_state_2023 = get_race_results_by_state_by_year(meet_df, result_df, 2023)
print("Calculating race results by state for 2024...")
results_by_state_2024 = get_race_results_by_state_by_year(meet_df, result_df, 2024)

# Create the choropleth map
print("Creating choropleth map...")
fig = create_state_choropleth_map(results_by_state_2023, results_by_state_2024)

# Save as PDF
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'race_results_by_state_2023_2024.pdf')
print(f"Saving map to {output_file}...")
fig.write_image(output_file, format='pdf', width=1600, height=1000, scale=2)

print("Map saved successfully!")
print(f"Total race results 2023: {sum(results_by_state_2023.values())}")
print(f"Total race results 2024: {sum(results_by_state_2024.values())}")

# Print summary statistics
print("\nTop 10 states by race results in 2023:")
sorted_2023 = sorted(results_by_state_2023.items(), key=lambda x: x[1], reverse=True)
for state, count in sorted_2023[:10]:
    print(f"{state}: {count}")

print("\nTop 10 states by race results in 2024:")
sorted_2024 = sorted(results_by_state_2024.items(), key=lambda x: x[1], reverse=True)
for state, count in sorted_2024[:10]:
    print(f"{state}: {count}") 