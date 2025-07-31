import pandas as pd
import numpy as np
import os
from utils import standardize_convert_exclude_nationals_df

output_dir = 'output/TeamRaceParticipation'
os.makedirs(output_dir, exist_ok=True)

def analyze_team_participation():
    """Analyze teams with at least one runner competing in 4+ races per season"""
    
    # Load data
    df = standardize_convert_exclude_nationals_df()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Load team and athlete-team association data
    athlete_team_df = pd.read_csv('data/jss_data/athlete_team_association.csv')
    team_df = pd.read_csv('data/jss_data/team.csv')
    athlete_df = pd.read_csv('data/jss_data/athlete.csv')
    
    # Merge athlete gender information
    athlete_df = athlete_df[['athlete_id', 'gender']]
    athlete_team_df = athlete_team_df.merge(athlete_df, on='athlete_id', how='left')
    
    # Filter out missing data
    df = df.dropna(subset=['athlete_id', 'start_date', 'gender'])
    athlete_team_df = athlete_team_df.dropna(subset=['athlete_id', 'team_id', 'gender'])
    
    results = {}
    
    for year in [2023, 2024]:
        start = pd.Timestamp(year=year, month=8, day=1)
        end = pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59)
        
        # Filter data for the year
        year_df = df[(df['start_date'] >= start) & (df['start_date'] <= end)].copy()
        
        # Count races per athlete for the year
        athlete_race_counts = year_df.groupby('athlete_id').size().reset_index(name='race_count')
        
        # Merge with athlete-team associations
        athlete_team_races = athlete_race_counts.merge(
            athlete_team_df[['athlete_id', 'team_id', 'gender']], 
            on='athlete_id', 
            how='left'
        )
        
        # Filter out athletes with missing team or gender info
        athlete_team_races = athlete_team_races.dropna(subset=['team_id', 'gender'])
        
        # For each gender, find teams with at least one athlete who ran 4+ races
        for gender in ['M', 'F']:
            gender_data = athlete_team_races[athlete_team_races['gender'] == gender]
            
            # Find athletes with 4+ races
            athletes_4plus = gender_data[gender_data['race_count'] >= 4]['athlete_id'].unique()
            
            # Find teams that have at least one such athlete
            teams_with_4plus_athlete = gender_data[
                gender_data['athlete_id'].isin(athletes_4plus)
            ]['team_id'].unique()
            
            # Total teams for this gender
            total_teams = gender_data['team_id'].nunique()
            
            # Teams with at least one athlete running 4+ races
            teams_with_4plus = len(teams_with_4plus_athlete)
            
            # Calculate percentage
            percentage = (teams_with_4plus / total_teams * 100) if total_teams > 0 else 0
            
            results[(year, gender)] = {
                'total_teams': total_teams,
                'teams_with_4plus_athlete': teams_with_4plus,
                'percentage': percentage
            }
    
    return results

def print_results(results):
    """Print the results in a formatted way"""
    print("Team Participation Analysis: Teams with at least one runner competing in 4+ races")
    print("=" * 80)
    
    for year in [2023, 2024]:
        print(f"\n{year} Season:")
        print("-" * 40)
        
        for gender in ['M', 'F']:
            gender_label = "Men" if gender == 'M' else "Women"
            data = results[(year, gender)]
            
            print(f"{gender_label}: {data['teams_with_4plus_athlete']}/{data['total_teams']} teams ({data['percentage']:.1f}%)")

def save_results_to_csv(results):
    """Save results to CSV file"""
    rows = []
    for year in [2023, 2024]:
        for gender in ['M', 'F']:
            gender_label = "Men" if gender == 'M' else "Women"
            data = results[(year, gender)]
            
            rows.append({
                'Year': year,
                'Gender': gender_label,
                'Total_Teams': data['total_teams'],
                'Teams_with_4plus_Athlete': data['teams_with_4plus_athlete'],
                'Percentage': data['percentage']
            })
    
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'team_participation_analysis.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    return results_df

def main():
    print("Analyzing team participation...")
    results = analyze_team_participation()
    
    print_results(results)
    save_results_to_csv(results)

if __name__ == '__main__':
    main() 