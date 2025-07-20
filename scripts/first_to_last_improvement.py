import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = 'data/jss_data'
output_dir = 'output/NumberOfRacesQuestion'
os.makedirs(output_dir, exist_ok=True)

# Load data
def load_data():
    athlete = pd.read_csv(os.path.join(data_dir, 'athlete.csv'))
    result = pd.read_csv(os.path.join(data_dir, 'result.csv'))
    meet = pd.read_csv(os.path.join(data_dir, 'meet.csv'))
    running_event = pd.read_csv(os.path.join(data_dir, 'running_event.csv'))
    return athlete, result, meet, running_event

def parse_time(time_str):
    if pd.isna(time_str):
        return np.nan
    parts = str(time_str).split(':')
    try:
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h*3600 + m*60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m*60 + s
        else:
            return float(time_str)
    except Exception:
        return np.nan

def analyze_season(season_year, athlete, result, meet, running_event):
    # Exclude nationals
    meet = meet[~meet['nationals'].astype(bool)]
    # Define season window: Aug 1 - Nov 28 of the given year
    start = pd.Timestamp(year=season_year, month=8, day=1)
    end = pd.Timestamp(year=season_year, month=11, day=28, hour=23, minute=59, second=59)
    # Merge result with meet to get date
    result_season = result.merge(meet[['meet_id', 'start_date']], on='meet_id', how='inner')
    result_season['start_date'] = pd.to_datetime(result_season['start_date'], errors='coerce')
    result_season = result_season[(result_season['start_date'] >= start) & (result_season['start_date'] <= end)]
    # Merge with athlete to get gender
    result_season = result_season.merge(athlete[['athlete_id', 'gender']], on='athlete_id', how='left')
    # Merge with running_event to get event_name
    result_season = result_season.merge(running_event[['running_event_id', 'event_name']], on='running_event_id', how='left')
    # Only keep 8000m for men and 6000m for women
    men = result_season[(result_season['gender'] == 'M') & (result_season['event_name'] == '8000m')]
    women = result_season[(result_season['gender'] == 'F') & (result_season['event_name'] == '6000m')]
    # Parse times
    men['parsed_time'] = men['result_time'].apply(parse_time)
    women['parsed_time'] = women['result_time'].apply(parse_time)
    # Remove missing times or dates
    men = men.dropna(subset=['parsed_time', 'start_date'])
    women = women.dropna(subset=['parsed_time', 'start_date'])
    # Group by athlete and count races
    men_counts = men.groupby('athlete_id').size()
    women_counts = women.groupby('athlete_id').size()
    # For each group (2,3,4,5), calculate first-to-last improvement
    results = {'M': {}, 'F': {}}
    for gender, df, counts in [('M', men, men_counts), ('F', women, women_counts)]:
        for n in [2,3,4,5]:
            ids = counts[counts == n].index
            improvements = []
            for aid in ids:
                races = df[df['athlete_id'] == aid].sort_values('start_date')
                if len(races) < 2:
                    continue
                first = races.iloc[0]['parsed_time']
                last = races.iloc[-1]['parsed_time']
                if pd.notna(first) and pd.notna(last):
                    improvements.append(first - last)  # positive = improvement
            if improvements:
                results[gender][n] = (np.mean(improvements), np.std(improvements), len(improvements))
            else:
                results[gender][n] = (np.nan, np.nan, 0)
    return results

def plot_results(results, season_year):
    fig, ax = plt.subplots(figsize=(8,6))
    x = np.array([2,3,4,5])
    for gender, color, label in [('M', 'blue', 'Men'), ('F', 'red', 'Women')]:
        means = [results[gender][n][0] for n in x]
        stds = [results[gender][n][1] for n in x]
        counts = [results[gender][n][2] for n in x]
        ax.errorbar(x, means, yerr=stds, fmt='o-', color=color, label=f'{label} (N={counts})')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Number of Races')
    ax.set_ylabel('First-to-Last Improvement (seconds)')
    ax.set_title(f'First-to-Last Race Improvement (Mean ± SD, Excluding Nationals) - {season_year}')
    ax.legend()
    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'first_to_last_improvement_{season_year}.pdf')
    plt.savefig(pdf_path)
    print('Saved plot to', pdf_path)
    plt.close()

def main():
    athlete, result, meet, running_event = load_data()
    for season_year in [2023, 2024]:
        results = analyze_season(season_year, athlete, result, meet, running_event)
        plot_results(results, season_year)

if __name__ == '__main__':
    main() 