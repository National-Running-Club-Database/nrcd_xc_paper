# Main question here: 

# How many races is the right number of races to have in a season?

# Steps: 

# Filter Males and Females from both years into groups based upon how many races they did in the season. (Full season is fine because we are looking
# at single season here)

# Look at the same analysis of max and min times from the same season based upon how many races they competed in.

# Could also bucket runners into groups of 2-3 races, 4-5 races, and 6+ races.

# Data preprocessing:

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
        meet_df = pd.read_csv(os.path.join(directory_path, 'meet.csv'))
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

# Loading in athletes from 2024: 

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

# Loading in athletes from 2023:

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

# XCAthletes2023 = analyze_cross_country_performance2023(sport_df, meet_df, result_df, athlete_df, running_event_df)
# XCAthletes2024 = analyze_cross_country_performance(sport_df, meet_df, result_df, athlete_df, running_event_df)

XCAthletes2023 = analyze_cross_country_performance2023(sport_df, meet_df, result_df, athlete_df, running_event_df)
XCAthletes2024 = analyze_cross_country_performance(sport_df, meet_df, result_df, athlete_df, running_event_df)

# Divide based on gender

XCMales2023 = XCAthletes2023[XCAthletes2023["gender"] == "M"]
XCMales2023 = XCMales2023[XCMales2023["event_type"] == "8000m"]
XCMales2023 = XCMales2023.reset_index(drop=True)

XCFemales2023 = XCAthletes2023[XCAthletes2023["gender"] == "F"]
XCFemales2023 = XCFemales2023[XCFemales2023["event_type"] == "6000m"]
XCFemales2023 = XCFemales2023.reset_index(drop=True)

# Can filter for number of races based upon the length of how many result times they have in their dataframe for that season.

# Now to filter based upon how many races they completed in each year

# Will do this after I talk with Ingrid / get in my run for the evening.