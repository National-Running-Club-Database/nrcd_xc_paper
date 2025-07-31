import pandas as pd
import numpy as np
import os
from utils import standardize_convert_exclude_nationals_df
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2_contingency

output_dir = 'output/TeamRaceParticipation'
os.makedirs(output_dir, exist_ok=True)

def get_teams_with_4plus_athletes(year, gender):
    """Get list of teams that have at least one athlete competing in 4+ races"""
    
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
    
    # Filter for the specific year and gender
    start = pd.Timestamp(year=year, month=8, day=1)
    end = pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59)
    
    year_df = df[(df['start_date'] >= start) & (df['start_date'] <= end)].copy()
    year_df = year_df[year_df['gender'] == gender]
    
    # Count races per athlete for the year
    athlete_race_counts = year_df.groupby('athlete_id').size().reset_index(name='race_count')
    
    # Merge with athlete-team associations
    athlete_team_races = athlete_race_counts.merge(
        athlete_team_df[['athlete_id', 'team_id', 'gender']], 
        on='athlete_id', 
        how='left'
    )
    
    # Filter for the specific gender
    gender_data = athlete_team_races[athlete_team_races['gender'] == gender]
    
    # Find athletes with 4+ races
    athletes_4plus = gender_data[gender_data['race_count'] >= 4]['athlete_id'].unique()
    
    # Find teams that have at least one such athlete
    teams_with_4plus_athlete = gender_data[
        gender_data['athlete_id'].isin(athletes_4plus)
    ]['team_id'].unique()
    
    return teams_with_4plus_athlete

def analyze_2024_mens_overlap():
    """Analyze overlap for 2024 men's teams"""
    
    # Get teams with 4+ athletes
    teams_4plus = get_teams_with_4plus_athletes(2024, 'M')
    
    # 2024 Men's Nationals Top 15 teams
    nationals_teams_2024_mens = [
        "Virginia Tech",
        "California Polytechnic State University", 
        "University of Michigan",
        "University of Virginia",
        "Georgia Tech",
        "University of Wisconsin",
        "University of Illinois",
        "Penn State University",
        "Ohio State University",
        "University of Maryland",
        "Cornell University",
        "Purdue University",
        "University of California, Davis",
        "Stanford University",
        "Michigan State University"
    ]
    
    # Load team data to get team names
    team_df = pd.read_csv('data/jss_data/team.csv')
    
    # Get team IDs for nationals teams (filter to avoid duplicates)
    nationals_team_ids = []
    for team_name in nationals_teams_2024_mens:
        matches = team_df[team_df['team_name'].str.contains(team_name, case=False, na=False)]
        if not matches.empty:
            # Take only the first match to avoid duplicates
            nationals_team_ids.append(matches['team_id'].iloc[0])
    
    # Find overlap
    overlap_teams = set(teams_4plus) & set(nationals_team_ids)
    
    # Get team names for overlap teams
    overlap_team_names = []
    for team_id in overlap_teams:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        overlap_team_names.append(team_name)
    
    # Get team names for all 4+ teams
    all_4plus_team_names = []
    for team_id in teams_4plus:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        all_4plus_team_names.append(team_name)
    
    # Get team names for all nationals teams
    all_nationals_team_names = []
    for team_id in nationals_team_ids:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        all_nationals_team_names.append(team_name)
    
    return {
        'teams_4plus': teams_4plus,
        'teams_4plus_names': all_4plus_team_names,
        'nationals_teams': nationals_team_ids,
        'nationals_teams_names': all_nationals_team_names,
        'overlap_teams': overlap_teams,
        'overlap_team_names': overlap_team_names,
        'overlap_count': len(overlap_teams),
        'total_4plus': len(teams_4plus),
        'total_nationals': len(nationals_team_ids),
        'overlap_percentage_4plus': len(overlap_teams) / len(teams_4plus) * 100 if len(teams_4plus) > 0 else 0,
        'overlap_percentage_nationals': len(overlap_teams) / len(nationals_team_ids) * 100 if len(nationals_team_ids) > 0 else 0
    }

def print_2024_mens_results(results):
    """Print the 2024 men's overlap results"""
    print("2024 Men's Nationals Overlap Analysis")
    print("=" * 50)
    print(f"Teams with at least one runner competing in 4+ races: {results['total_4plus']}")
    print(f"Top 15 teams at nationals: {results['total_nationals']}")
    print(f"Overlap: {results['overlap_count']} teams")
    print()
    print(f"Percentage of 4+ teams that were top 15 at nationals: {results['overlap_percentage_4plus']:.1f}%")
    print(f"Percentage of top 15 nationals teams that had 4+ athletes: {results['overlap_percentage_nationals']:.1f}%")
    print()
    print("Teams in both categories:")
    print("-" * 30)
    for team_name in sorted(results['overlap_team_names']):
        print(f"• {team_name}")
    print()
    print("Teams with 4+ athletes but NOT in top 15:")
    print("-" * 40)
    non_overlap_4plus = set(results['teams_4plus_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_4plus):
        print(f"• {team_name}")
    print()
    print("Top 15 teams that did NOT have 4+ athletes:")
    print("-" * 45)
    non_overlap_nationals = set(results['nationals_teams_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_nationals):
        print(f"• {team_name}")

def analyze_2023_mens_overlap():
    """Analyze overlap for 2023 men's teams"""
    
    # Get teams with 4+ athletes
    teams_4plus = get_teams_with_4plus_athletes(2023, 'M')
    
    # 2023 Men's Nationals Top 15 teams
    nationals_teams_2023_mens = [
        "University of Michigan",
        "Virginia Tech",
        "California Polytechnic State University",
        "Ohio State University",
        "University of Wisconsin",
        "Georgia Tech",
        "Northeastern University",
        "University of Illinois",
        "Purdue University",
        "University of Maryland",
        "University of Notre Dame",
        "Michigan State University",
        "North Carolina State University",
        "Northwestern University",
        "Boston College"
    ]
    
    # Load team data to get team names
    team_df = pd.read_csv('data/jss_data/team.csv')
    
    # Get team IDs for nationals teams (filter to avoid duplicates)
    nationals_team_ids = []
    for team_name in nationals_teams_2023_mens:
        matches = team_df[team_df['team_name'].str.contains(team_name, case=False, na=False)]
        if not matches.empty:
            # Take only the first match to avoid duplicates
            nationals_team_ids.append(matches['team_id'].iloc[0])
    
    # Find overlap
    overlap_teams = set(teams_4plus) & set(nationals_team_ids)
    
    # Get team names for overlap teams
    overlap_team_names = []
    for team_id in overlap_teams:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        overlap_team_names.append(team_name)
    
    # Get team names for all 4+ teams
    all_4plus_team_names = []
    for team_id in teams_4plus:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        all_4plus_team_names.append(team_name)
    
    # Get team names for all nationals teams
    all_nationals_team_names = []
    for team_id in nationals_team_ids:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        all_nationals_team_names.append(team_name)
    
    return {
        'teams_4plus': teams_4plus,
        'teams_4plus_names': all_4plus_team_names,
        'nationals_teams': nationals_team_ids,
        'nationals_teams_names': all_nationals_team_names,
        'overlap_teams': overlap_teams,
        'overlap_team_names': overlap_team_names,
        'overlap_count': len(overlap_teams),
        'total_4plus': len(teams_4plus),
        'total_nationals': len(nationals_team_ids),
        'overlap_percentage_4plus': len(overlap_teams) / len(teams_4plus) * 100 if len(teams_4plus) > 0 else 0,
        'overlap_percentage_nationals': len(overlap_teams) / len(nationals_team_ids) * 100 if len(nationals_team_ids) > 0 else 0
    }

def print_2023_mens_results(results):
    """Print the 2023 men's overlap results"""
    print("\n2023 Men's Nationals Overlap Analysis")
    print("=" * 50)
    print(f"Teams with at least one runner competing in 4+ races: {results['total_4plus']}")
    print(f"Top 15 teams at nationals: {results['total_nationals']}")
    print(f"Overlap: {results['overlap_count']} teams")
    print()
    print(f"Percentage of 4+ teams that were top 15 at nationals: {results['overlap_percentage_4plus']:.1f}%")
    print(f"Percentage of top 15 nationals teams that had 4+ athletes: {results['overlap_percentage_nationals']:.1f}%")
    print()
    print("Teams in both categories:")
    print("-" * 30)
    for team_name in sorted(results['overlap_team_names']):
        print(f"• {team_name}")
    print()
    print("Teams with 4+ athletes but NOT in top 15:")
    print("-" * 40)
    non_overlap_4plus = set(results['teams_4plus_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_4plus):
        print(f"• {team_name}")
    print()
    print("Top 15 teams that did NOT have 4+ athletes:")
    print("-" * 45)
    non_overlap_nationals = set(results['nationals_teams_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_nationals):
        print(f"• {team_name}")

def analyze_2023_womens_overlap():
    """Analyze overlap for 2023 women's teams"""
    
    # Get teams with 4+ athletes
    teams_4plus = get_teams_with_4plus_athletes(2023, 'F')
    
    # 2023 Women's Nationals Top 15 teams
    nationals_teams_2023_womens = [
        "University of Wisconsin",
        "University of Michigan",
        "Virginia Tech",
        "Penn State University",
        "Purdue University",
        "Ohio State University",
        "University of Illinois",
        "University of Notre Dame",
        "Northeastern University",
        "University of Texas - Austin",
        "University of Virginia",
        "University of Maryland",
        "University of Minnesota",
        "Princeton University",
        "Georgetown University"
    ]
    
    # Load team data to get team names
    team_df = pd.read_csv('data/jss_data/team.csv')
    
    # Get team IDs for nationals teams (filter to avoid duplicates)
    nationals_team_ids = []
    for team_name in nationals_teams_2023_womens:
        matches = team_df[team_df['team_name'].str.contains(team_name, case=False, na=False)]
        if not matches.empty:
            # Take only the first match to avoid duplicates
            nationals_team_ids.append(matches['team_id'].iloc[0])
    
    # Find overlap
    overlap_teams = set(teams_4plus) & set(nationals_team_ids)
    
    # Get team names for overlap teams
    overlap_team_names = []
    for team_id in overlap_teams:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        overlap_team_names.append(team_name)
    
    # Get team names for all 4+ teams
    all_4plus_team_names = []
    for team_id in teams_4plus:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        all_4plus_team_names.append(team_name)
    
    # Get team names for all nationals teams
    all_nationals_team_names = []
    for team_id in nationals_team_ids:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        all_nationals_team_names.append(team_name)
    
    return {
        'teams_4plus': teams_4plus,
        'teams_4plus_names': all_4plus_team_names,
        'nationals_teams': nationals_team_ids,
        'nationals_teams_names': all_nationals_team_names,
        'overlap_teams': overlap_teams,
        'overlap_team_names': overlap_team_names,
        'overlap_count': len(overlap_teams),
        'total_4plus': len(teams_4plus),
        'total_nationals': len(nationals_team_ids),
        'overlap_percentage_4plus': len(overlap_teams) / len(teams_4plus) * 100 if len(teams_4plus) > 0 else 0,
        'overlap_percentage_nationals': len(overlap_teams) / len(nationals_team_ids) * 100 if len(nationals_team_ids) > 0 else 0
    }

def print_2023_womens_results(results):
    """Print the 2023 women's overlap results"""
    print("\n2023 Women's Nationals Overlap Analysis")
    print("=" * 50)
    print(f"Teams with at least one runner competing in 4+ races: {results['total_4plus']}")
    print(f"Top 15 teams at nationals: {results['total_nationals']}")
    print(f"Overlap: {results['overlap_count']} teams")
    print()
    print(f"Percentage of 4+ teams that were top 15 at nationals: {results['overlap_percentage_4plus']:.1f}%")
    print(f"Percentage of top 15 nationals teams that had 4+ athletes: {results['overlap_percentage_nationals']:.1f}%")
    print()
    print("Teams in both categories:")
    print("-" * 30)
    for team_name in sorted(results['overlap_team_names']):
        print(f"• {team_name}")
    print()
    print("Teams with 4+ athletes but NOT in top 15:")
    print("-" * 40)
    non_overlap_4plus = set(results['teams_4plus_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_4plus):
        print(f"• {team_name}")
    print()
    print("Top 15 teams that did NOT have 4+ athletes:")
    print("-" * 45)
    non_overlap_nationals = set(results['nationals_teams_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_nationals):
        print(f"• {team_name}")

def analyze_2024_womens_overlap():
    """Analyze overlap for 2024 women's teams"""
    
    # Get teams with 4+ athletes
    teams_4plus = get_teams_with_4plus_athletes(2024, 'F')
    
    # 2024 Women's Nationals Top 15 teams
    nationals_teams_2024_womens = [
        "University of Michigan",
        "University of Wisconsin",
        "University of Virginia",
        "Cornell University",
        "Virginia Tech",
        "California Polytechnic State University",
        "Northeastern University",
        "Ohio State University",
        "University of Tennessee",
        "Boston College",
        "University of Maryland",
        "Brown University",
        "Michigan State University",
        "Northwestern University",
        "University of Central Florida"
    ]
    
    # Load team data to get team names
    team_df = pd.read_csv('data/jss_data/team.csv')
    
    # Get team IDs for nationals teams (filter to avoid duplicates)
    nationals_team_ids = []
    for team_name in nationals_teams_2024_womens:
        matches = team_df[team_df['team_name'].str.contains(team_name, case=False, na=False)]
        if not matches.empty:
            # Take only the first match to avoid duplicates
            nationals_team_ids.append(matches['team_id'].iloc[0])
    
    # Find overlap
    overlap_teams = set(teams_4plus) & set(nationals_team_ids)
    
    # Get team names for overlap teams
    overlap_team_names = []
    for team_id in overlap_teams:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        overlap_team_names.append(team_name)
    
    # Get team names for all 4+ teams
    all_4plus_team_names = []
    for team_id in teams_4plus:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        all_4plus_team_names.append(team_name)
    
    # Get team names for all nationals teams
    all_nationals_team_names = []
    for team_id in nationals_team_ids:
        team_name = team_df[team_df['team_id'] == team_id]['team_name'].iloc[0]
        all_nationals_team_names.append(team_name)
    
    return {
        'teams_4plus': teams_4plus,
        'teams_4plus_names': all_4plus_team_names,
        'nationals_teams': nationals_team_ids,
        'nationals_teams_names': all_nationals_team_names,
        'overlap_teams': overlap_teams,
        'overlap_team_names': overlap_team_names,
        'overlap_count': len(overlap_teams),
        'total_4plus': len(teams_4plus),
        'total_nationals': len(nationals_team_ids),
        'overlap_percentage_4plus': len(overlap_teams) / len(teams_4plus) * 100 if len(teams_4plus) > 0 else 0,
        'overlap_percentage_nationals': len(overlap_teams) / len(nationals_team_ids) * 100 if len(nationals_team_ids) > 0 else 0
    }

def print_2024_womens_results(results):
    """Print the 2024 women's overlap results"""
    print("\n2024 Women's Nationals Overlap Analysis")
    print("=" * 50)
    print(f"Teams with at least one runner competing in 4+ races: {results['total_4plus']}")
    print(f"Top 15 teams at nationals: {results['total_nationals']}")
    print(f"Overlap: {results['overlap_count']} teams")
    print()
    print(f"Percentage of 4+ teams that were top 15 at nationals: {results['overlap_percentage_4plus']:.1f}%")
    print(f"Percentage of top 15 nationals teams that had 4+ athletes: {results['overlap_percentage_nationals']:.1f}%")
    print()
    print("Teams in both categories:")
    print("-" * 30)
    for team_name in sorted(results['overlap_team_names']):
        print(f"• {team_name}")
    print()
    print("Teams with 4+ athletes but NOT in top 15:")
    print("-" * 40)
    non_overlap_4plus = set(results['teams_4plus_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_4plus):
        print(f"• {team_name}")
    print()
    print("Top 15 teams that did NOT have 4+ athletes:")
    print("-" * 45)
    non_overlap_nationals = set(results['nationals_teams_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_nationals):
        print(f"• {team_name}")

def save_2024_mens_results(results):
    """Save 2024 men's results to CSV"""
    # Create detailed results
    rows = []
    
    # Teams in overlap
    for team_name in results['overlap_team_names']:
        rows.append({
            'Team_Name': team_name,
            'Category': 'Both (4+ athletes AND top 15 nationals)'
        })
    
    # Teams with 4+ but not top 15
    non_overlap_4plus = set(results['teams_4plus_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_4plus):
        rows.append({
            'Team_Name': team_name,
            'Category': '4+ athletes only'
        })
    
    # Top 15 teams without 4+ athletes
    non_overlap_nationals = set(results['nationals_teams_names']) - set(results['overlap_team_names'])
    for team_name in sorted(non_overlap_nationals):
        rows.append({
            'Team_Name': team_name,
            'Category': 'Top 15 nationals only'
        })
    
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, '2024_mens_nationals_overlap.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    
    # Create summary
    summary_data = {
        'Metric': [
            'Total teams with 4+ athletes',
            'Total top 15 nationals teams', 
            'Overlap count',
            'Overlap percentage (of 4+ teams)',
            'Overlap percentage (of nationals teams)'
        ],
        'Value': [
            results['total_4plus'],
            results['total_nationals'],
            results['overlap_count'],
            f"{results['overlap_percentage_4plus']:.1f}%",
            f"{results['overlap_percentage_nationals']:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, '2024_mens_nationals_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

def calculate_p_values(summary_stats):
    """Calculate p-values for chi-square tests for each category"""
    p_values = {}
    
    for stat in summary_stats:
        category = stat['Category']
        
        # Create 2x2 contingency table for this category:
        #                    Made Top 15    Not in Top 15
        # Teams with 4+     Overlap        TeamsWith4Plus - Overlap
        # Teams without 4+  TotalTop15 - Overlap  TotalTeams - TeamsWith4Plus - (TotalTop15 - Overlap)
        
        # Teams with 4+ that made top 15
        a = stat['Overlap']
        # Teams with 4+ that didn't make top 15
        b = stat['TeamsWith4Plus'] - stat['Overlap']
        # Teams without 4+ that made top 15
        c = stat['TotalTop15'] - stat['Overlap']
        # Teams without 4+ that didn't make top 15
        d = stat['TotalTeams'] - stat['TeamsWith4Plus'] - (stat['TotalTop15'] - stat['Overlap'])
        
        contingency_table = [[a, b], [c, d]]
        
        # Chi-square test for this category
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            p_values[category] = p_value
        except:
            p_values[category] = 1.0
    
    return p_values

def create_summary_table_pdf(summary_stats, output_path):
    """Create a PDF table summarizing all four categories"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    # Calculate p-values
    p_values = calculate_p_values(summary_stats)

    columns = [
        "Category",
        "Total Teams",
        "Total in Top 15",
        "Teams with 4+ Athletes",
        "% of Total Teams with 4+ Athletes",
        "% of Top 15 with 4+ Athletes",
        "% of 4+ Teams in Top 15",
        "Overlap Count",
        "P-value (χ² test)"
    ]
    data = []
    for stat in summary_stats:
        data.append([
            stat['Category'],
            stat['TotalTeams'],
            stat['TotalTop15'],
            stat['TeamsWith4Plus'],
            f"{stat['PctTotal4Plus']:.1f}%",
            f"{stat['PctTop15With4Plus']:.1f}%",
            f"{stat['Pct4PlusInTop15']:.1f}%",
            stat['Overlap'],
            f"{p_values[stat['Category']]:.3f}"
        ])

    fig, ax = plt.subplots(figsize=(10, 2 + 0.5*len(data)))
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.title('Nationals Overlap Summary (2023-2024)', fontsize=16, pad=20)
    plt.tight_layout()
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def calculate_overall_stats():
    """Calculate overall statistics across all categories"""
    print("\n" + "="*60)
    print("OVERALL SUMMARY STATISTICS")
    print("="*60)
    
    # Get baseline stats from the team participation analysis
    baseline_stats = {
        '2023_M': {'total_teams': 94, 'teams_4plus': 36, 'percentage': 38.3},
        '2023_F': {'total_teams': 94, 'teams_4plus': 27, 'percentage': 28.7},
        '2024_M': {'total_teams': 115, 'teams_4plus': 37, 'percentage': 32.2},
        '2024_F': {'total_teams': 101, 'teams_4plus': 28, 'percentage': 27.7}
    }
    
    # 2024 Men's overlap (from our analysis)
    mens_2024_overlap = 11
    mens_2024_nationals = 15  # Top 15 teams
    
    print(f"2024 Men's Teams:")
    print(f"  • Total teams in dataset: {baseline_stats['2024_M']['total_teams']}")
    print(f"  • Teams with 4+ athletes: {baseline_stats['2024_M']['teams_4plus']} ({baseline_stats['2024_M']['percentage']:.1f}%)")
    print(f"  • Top 15 at nationals: {mens_2024_nationals}")
    print(f"  • Overlap: {mens_2024_overlap} teams")
    print(f"  • Overlap rate: {mens_2024_overlap/mens_2024_nationals*100:.1f}% of nationals teams had 4+ athletes")
    print(f"  • Success rate: {mens_2024_overlap/baseline_stats['2024_M']['teams_4plus']*100:.1f}% of 4+ teams made top 15")
    
    # Overall statistics
    total_teams_4plus = sum([stats['teams_4plus'] for stats in baseline_stats.values()])
    total_teams = sum([stats['total_teams'] for stats in baseline_stats.values()])
    overall_percentage = total_teams_4plus / total_teams * 100
    
    print(f"\nOverall Statistics (All Categories):")
    print(f"  • Total teams across all categories: {total_teams}")
    print(f"  • Teams with 4+ athletes: {total_teams_4plus}")
    print(f"  • Overall percentage: {overall_percentage:.1f}%")
    print(f"  • 2024 Men's nationals success: {mens_2024_overlap/mens_2024_nationals*100:.1f}%")

def main():
    print("Analyzing Nationals overlap across years...")
    
    # 2023 Men's analysis
    mens_2023_results = analyze_2023_mens_overlap()
    print_2023_mens_results(mens_2023_results)
    
    # 2023 Women's analysis
    womens_2023_results = analyze_2023_womens_overlap()
    print_2023_womens_results(womens_2023_results)
    
    # 2024 Men's analysis
    mens_2024_results = analyze_2024_mens_overlap()
    print_2024_mens_results(mens_2024_results)
    save_2024_mens_results(mens_2024_results)
    
    # 2024 Women's analysis
    womens_2024_results = analyze_2024_womens_overlap()
    print_2024_womens_results(womens_2024_results)
    
    calculate_overall_stats()

    # Collect summary stats for table
    summary_stats = [
        {
            'Category': '2023 Men',
            'TotalTeams': 94,
            'TotalTop15': mens_2023_results['total_nationals'],
            'TeamsWith4Plus': mens_2023_results['total_4plus'],
            'PctTotal4Plus': 36/94*100,
            'PctTop15With4Plus': mens_2023_results['overlap_count']/mens_2023_results['total_nationals']*100 if mens_2023_results['total_nationals'] else 0,
            'Pct4PlusInTop15': mens_2023_results['overlap_count']/mens_2023_results['total_4plus']*100 if mens_2023_results['total_4plus'] else 0,
            'Overlap': mens_2023_results['overlap_count']
        },
        {
            'Category': '2023 Women',
            'TotalTeams': 94,
            'TotalTop15': womens_2023_results['total_nationals'],
            'TeamsWith4Plus': womens_2023_results['total_4plus'],
            'PctTotal4Plus': 27/94*100,
            'PctTop15With4Plus': womens_2023_results['overlap_count']/womens_2023_results['total_nationals']*100 if womens_2023_results['total_nationals'] else 0,
            'Pct4PlusInTop15': womens_2023_results['overlap_count']/womens_2023_results['total_4plus']*100 if womens_2023_results['total_4plus'] else 0,
            'Overlap': womens_2023_results['overlap_count']
        },
        {
            'Category': '2024 Men',
            'TotalTeams': 115,
            'TotalTop15': mens_2024_results['total_nationals'],
            'TeamsWith4Plus': mens_2024_results['total_4plus'],
            'PctTotal4Plus': 37/115*100,
            'PctTop15With4Plus': mens_2024_results['overlap_count']/mens_2024_results['total_nationals']*100 if mens_2024_results['total_nationals'] else 0,
            'Pct4PlusInTop15': mens_2024_results['overlap_count']/mens_2024_results['total_4plus']*100 if mens_2024_results['total_4plus'] else 0,
            'Overlap': mens_2024_results['overlap_count']
        },
        {
            'Category': '2024 Women',
            'TotalTeams': 101,
            'TotalTop15': womens_2024_results['total_nationals'],
            'TeamsWith4Plus': womens_2024_results['total_4plus'],
            'PctTotal4Plus': 28/101*100,
            'PctTop15With4Plus': womens_2024_results['overlap_count']/womens_2024_results['total_nationals']*100 if womens_2024_results['total_nationals'] else 0,
            'Pct4PlusInTop15': womens_2024_results['overlap_count']/womens_2024_results['total_4plus']*100 if womens_2024_results['total_4plus'] else 0,
            'Overlap': womens_2024_results['overlap_count']
        }
    ]
    output_path = os.path.join(output_dir, 'nationals_overlap_summary.pdf')
    create_summary_table_pdf(summary_stats, output_path)
    print(f"\nSummary table PDF saved to: {output_path}")
    
    # Print p-values to console for verification
    p_values = calculate_p_values(summary_stats)
    print(f"\nStatistical Significance (Chi-square tests for each category):")
    print(f"Testing relationship: Teams with 4+ athletes vs. Making Top 15")
    
    for stat in summary_stats:
        category = stat['Category']
        p_value = p_values[category]
        print(f"  • {category}: p = {p_value:.3f}")
        if p_value < 0.05:
            print(f"    → Significant relationship (p < 0.05)")
        else:
            print(f"    → No significant relationship (p ≥ 0.05)")
    
    print(f"\nContingency Tables for each category:")
    for stat in summary_stats:
        category = stat['Category']
        print(f"\n{category}:")
        print(f"  Teams with 4+ athletes: {stat['TeamsWith4Plus']}")
        print(f"  Teams in Top 15: {stat['TotalTop15']}")
        print(f"  Overlap: {stat['Overlap']}")
        print(f"  % of 4+ teams in Top 15: {stat['Pct4PlusInTop15']:.1f}%")
        print(f"  % of Top 15 with 4+ athletes: {stat['PctTop15With4Plus']:.1f}%")

if __name__ == '__main__':
    main() 