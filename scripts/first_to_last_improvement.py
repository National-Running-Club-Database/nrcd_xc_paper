import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import standardize_convert_exclude_nationals_df, convert_exclude_nationals

output_dir = 'output/NumberOfRacesQuestion'
os.makedirs(output_dir, exist_ok=True)

def analyze_all(df):
    # Remove missing times or dates
    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df = df.dropna(subset=['standardized_to_target', 'start_date', 'gender'])
    # Split by gender
    men = df[df['gender'] == 'M']
    women = df[df['gender'] == 'F']
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
                first = races.iloc[0]['standardized_to_target']
                last = races.iloc[-1]['standardized_to_target']
                if pd.notna(first) and pd.notna(last):
                    improvements.append(first - last)  # positive = improvement
            if improvements:
                results[gender][n] = (np.mean(improvements), np.std(improvements), len(improvements))
            else:
                results[gender][n] = (np.nan, np.nan, 0)
    return results

def plot_results(results, year, subtitle, filename):
    fig, ax = plt.subplots(figsize=(8,6))
    x = np.array([2,3,4,5])
    width = 0.15
    offsets = {'M': -width/2, 'F': width/2}
    colors = {'M': 'blue', 'F': 'red'}
    labels = {'M': 'Men', 'F': 'Women'}
    for gender in ['M', 'F']:
        means = [results[gender][n][0] for n in x]
        stds = [results[gender][n][1] for n in x]
        counts = [results[gender][n][2] for n in x]
        # Offset x positions for men and women so error bars don't overlap
        x_offset = x + offsets[gender]
        ax.errorbar(x_offset, means, yerr=stds, fmt='o', color=colors[gender], label=f'{labels[gender]} (N={counts})', capsize=5, elinewidth=2)
        # Draw horizontal lines for error bars
        for xi, mean, std in zip(x_offset, means, stds):
            ax.hlines(y=mean, xmin=xi-width/2, xmax=xi+width/2, color=colors[gender], alpha=0.5, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Number of Races')
    ax.set_ylabel('First-to-Last Improvement (seconds, standardized to 6K/8K)')
    ax.set_title(f'First-to-Last Race Improvement (Mean ± SD, Excluding Nationals) - {year}')
    ax.set_xticks(x)
    ax.legend()
    # Add subtitle
    plt.suptitle(subtitle, fontsize=10, y=0.93)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf_path = os.path.join(output_dir, filename)
    plt.savefig(pdf_path)
    print('Saved plot to', pdf_path)
    plt.close()

def plot_comparison_grid(results_dict):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    x = np.array([2,3,4,5])
    width = 0.15
    offsets = {'M': -width/2, 'F': width/2}
    colors = {'M': 'blue', 'F': 'red'}
    labels = {'M': 'Men', 'F': 'Women'}
    titles = {
        (2023, 'standardized'): '2023: Standardized (weather, terrain, etc.)',
        (2023, 'converted'): '2023: Converted Only (distance)',
        (2024, 'standardized'): '2024: Standardized (weather, terrain, etc.)',
        (2024, 'converted'): '2024: Converted Only (distance)',
    }
    for i, year in enumerate([2023, 2024]):
        for j, mode in enumerate(['standardized', 'converted']):
            ax = axes[i, j]
            results = results_dict[(year, mode)]
            for gender in ['M', 'F']:
                means = [results[gender][n][0] for n in x]
                stds = [results[gender][n][1] for n in x]
                counts = [results[gender][n][2] for n in x]
                x_offset = x + offsets[gender]
                ax.errorbar(x_offset, means, yerr=stds, fmt='o', color=colors[gender], label=f'{labels[gender]} (N={counts})', capsize=5, elinewidth=2)
                for xi, mean, std in zip(x_offset, means, stds):
                    ax.hlines(y=mean, xmin=xi-width/2, xmax=xi+width/2, color=colors[gender], alpha=0.5, linewidth=2)
            ax.axhline(0, color='gray', linestyle='--')
            ax.set_xlabel('Number of Races')
            ax.set_ylabel('First-to-Last Improvement (seconds, standardized to 6K/8K)')
            ax.set_title(titles[(year, mode)])
            ax.set_xticks(x)
            ax.legend(loc='upper right')
    plt.suptitle('First-to-Last Race Improvement Comparison\nAll times standardized and converted: Women to 6K, Men to 8K\nStandardized = weather/terrain/course adjustment; Converted = distance only', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_path = os.path.join(output_dir, 'first_to_last_improvement_comparison.pdf')
    plt.savefig(pdf_path)
    print('Saved comparison grid to', pdf_path)
    plt.close()

def main():
    # 1. Full standardization (weather, terrain, etc.)
    df_std = standardize_convert_exclude_nationals_df()
    df_std['start_date'] = pd.to_datetime(df_std['start_date'], errors='coerce')
    # 2. Conversion only (no course details adjustment)
    df_conv = convert_exclude_nationals()
    df_conv['start_date'] = pd.to_datetime(df_conv['start_date'], errors='coerce')
    results_dict = {}
    for year in [2023, 2024]:
        start = pd.Timestamp(year=year, month=8, day=1)
        end = pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59)
        # Full standardization
        df_std_year = df_std[(df_std['start_date'] >= start) & (df_std['start_date'] <= end)].copy()
        results_std = analyze_all(df_std_year)
        plot_results(results_std, year, 'All times standardized and converted: Women to 6K, Men to 8K', f'first_to_last_improvement_{year}_standardized.pdf')
        # Conversion only
        df_conv_year = df_conv[(df_conv['start_date'] >= start) & (df_conv['start_date'] <= end)].copy()
        results_conv = analyze_all(df_conv_year)
        plot_results(results_conv, year, 'No weather/terrain adjustment: Only converted to 6K/8K', f'first_to_last_improvement_{year}_converted.pdf')
        results_dict[(year, 'standardized')] = results_std
        results_dict[(year, 'converted')] = results_conv
    plot_comparison_grid(results_dict)

if __name__ == '__main__':
    main() 