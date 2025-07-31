import pandas as pd
import numpy as np
import os
from utils import standardize_convert_exclude_nationals_df
from scipy.stats import chi2_contingency

output_dir = 'output/GenderAnalysis'
os.makedirs(output_dir, exist_ok=True)

def analyze_gender_race_participation():
    """Analyze gender differences in race participation for 2023 and 2024"""
    
    # Load data (gender is already included from the utils function)
    df = standardize_convert_exclude_nationals_df()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter out missing data
    df = df.dropna(subset=['athlete_id', 'start_date', 'gender'])
    
    results = {}
    
    for year in [2023, 2024]:
        start = pd.Timestamp(year=year, month=8, day=1)
        end = pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59)
        
        # Filter data for the year
        year_df = df[(df['start_date'] >= start) & (df['start_date'] <= end)].copy()
        
        # Count unique athletes by gender
        gender_counts = year_df.groupby('gender')['athlete_id'].nunique()
        
        # Count total races by gender
        race_counts = year_df.groupby('gender').size()
        
        # Get counts
        men_athletes = gender_counts.get('M', 0)
        women_athletes = gender_counts.get('F', 0)
        men_races = race_counts.get('M', 0)
        women_races = race_counts.get('F', 0)
        
        # Calculate average races per athlete by gender
        men_avg_races = men_races / men_athletes if men_athletes > 0 else 0
        women_avg_races = women_races / women_athletes if women_athletes > 0 else 0
        
        # Chi-square test for athlete counts (comparing observed vs expected 50/50 split)
        total_athletes = men_athletes + women_athletes
        expected_men = total_athletes / 2
        expected_women = total_athletes / 2
        contingency_athletes = [[men_athletes, women_athletes], [expected_men, expected_women]]
        try:
            chi2_athletes, p_athletes, dof_athletes, expected_athletes = chi2_contingency(contingency_athletes)
        except:
            chi2_athletes, p_athletes = 0, 1.0
        
        # Chi-square test for race counts (comparing observed vs expected 50/50 split)
        total_races = men_races + women_races
        expected_men_races = total_races / 2
        expected_women_races = total_races / 2
        contingency_races = [[men_races, women_races], [expected_men_races, expected_women_races]]
        try:
            chi2_races, p_races, dof_races, expected_races = chi2_contingency(contingency_races)
        except:
            chi2_races, p_races = 0, 1.0
        
        results[year] = {
            'men_athletes': men_athletes,
            'women_athletes': women_athletes,
            'men_races': men_races,
            'women_races': women_races,
            'men_avg_races': men_avg_races,
            'women_avg_races': women_avg_races,
            'p_athletes': p_athletes,
            'p_races': p_races,
            'total_athletes': men_athletes + women_athletes,
            'total_races': men_races + women_races
        }
    
    return results

def print_gender_analysis(results):
    """Print the gender analysis results"""
    print("Gender Race Participation Analysis")
    print("=" * 50)
    
    for year in [2023, 2024]:
        data = results[year]
        print(f"\n{year} Season:")
        print("-" * 30)
        print(f"Men:")
        print(f"  • Athletes: {data['men_athletes']:,}")
        print(f"  • Races: {data['men_races']:,}")
        print(f"  • Avg races per athlete: {data['men_avg_races']:.2f}")
        print(f"  • % of total athletes: {data['men_athletes']/data['total_athletes']*100:.1f}%")
        print(f"  • % of total races: {data['men_races']/data['total_races']*100:.1f}%")
        
        print(f"\nWomen:")
        print(f"  • Athletes: {data['women_athletes']:,}")
        print(f"  • Races: {data['women_races']:,}")
        print(f"  • Avg races per athlete: {data['women_avg_races']:.2f}")
        print(f"  • % of total athletes: {data['women_athletes']/data['total_athletes']*100:.1f}%")
        print(f"  • % of total races: {data['women_races']/data['total_races']*100:.1f}%")
        
        print(f"\nStatistical Tests:")
        print(f"  • P-value (athlete count): {data['p_athletes']:.3f}")
        print(f"  • P-value (race count): {data['p_races']:.3f}")
        
        if data['p_athletes'] < 0.05:
            print(f"  → Significant difference in athlete counts (p < 0.05)")
        else:
            print(f"  → No significant difference in athlete counts (p ≥ 0.05)")
            
        if data['p_races'] < 0.05:
            print(f"  → Significant difference in race counts (p < 0.05)")
        else:
            print(f"  → No significant difference in race counts (p ≥ 0.05)")

def save_gender_analysis(results):
    """Save gender analysis results to CSV"""
    rows = []
    for year in [2023, 2024]:
        data = results[year]
        rows.append({
            'Year': year,
            'Gender': 'Men',
            'Athletes': data['men_athletes'],
            'Races': data['men_races'],
            'Avg_Races_Per_Athlete': data['men_avg_races'],
            'Pct_Total_Athletes': data['men_athletes']/data['total_athletes']*100,
            'Pct_Total_Races': data['men_races']/data['total_races']*100
        })
        rows.append({
            'Year': year,
            'Gender': 'Women',
            'Athletes': data['women_athletes'],
            'Races': data['women_races'],
            'Avg_Races_Per_Athlete': data['women_avg_races'],
            'Pct_Total_Athletes': data['women_athletes']/data['total_athletes']*100,
            'Pct_Total_Races': data['women_races']/data['total_races']*100
        })
    
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'gender_race_participation.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Create summary with p-values
    summary_rows = []
    for year in [2023, 2024]:
        data = results[year]
        summary_rows.append({
            'Year': year,
            'Men_Athletes': data['men_athletes'],
            'Women_Athletes': data['women_athletes'],
            'Men_Races': data['men_races'],
            'Women_Races': data['women_races'],
            'P_value_Athletes': data['p_athletes'],
            'P_value_Races': data['p_races'],
            'Significant_Athletes': data['p_athletes'] < 0.05,
            'Significant_Races': data['p_races'] < 0.05
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'gender_analysis_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary with p-values saved to: {summary_path}")

def main():
    print("Analyzing gender differences in race participation...")
    results = analyze_gender_race_participation()
    
    print_gender_analysis(results)
    save_gender_analysis(results)

if __name__ == '__main__':
    main() 