"""
RQ2 Multi-Season Analysis: Analyze athletes across all seasons and calculate improvement metrics.

This script performs the multi-season athlete analysis for RQ2:
- Calculates improvement metrics (2023→2024, 2024→2025, 2023→2025)
- Creates summary statistics by gender
- Generates visualizations

All outputs are saved to output/rq2/

Run from main directory: python scripts/multi_season_analysis_rq2.py
Or import and call from rq2.py
"""

import os
import sys

# Setup paths for imports
from _setup_paths import setup_paths
setup_paths()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_improvement_plot(results_df, output_dir):
    """Create visualization of improvement across seasons."""
    print("\nCreating improvement visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, gender in enumerate(['M', 'F']):
        gender_label = 'Men' if gender == 'M' else 'Women'
        gender_data = results_df[results_df['gender'] == gender]
        
        if len(gender_data) == 0:
            continue
        
        ax = axes[idx]
        
        # Plot improvement distribution
        improvements = gender_data['improvement_2023_to_2025_minutes'].dropna()
        if len(improvements) > 0:
            ax.hist(improvements, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(improvements.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {improvements.mean():.2f} min')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.set_xlabel('Improvement (minutes, negative = faster)')
            ax.set_ylabel('Number of Athletes')
            ax.set_title(f'{gender_label} - Improvement Distribution\n(n={len(improvements)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/athletes_all_seasons_improvement.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_dir}/athletes_all_seasons_improvement.pdf")

def create_rq2_comprehensive_plots(results_df, output_dir='output/rq2'):
    """
    Create comprehensive RQ2 visualization with 3 charts:
    1. Bar chart of mean improvement for 2023-2024 and 2024-2025
    2. Improvement between 2023-2024 and 2024-2025 with error bars
    3. Distribution of fastest times across years
    """
    print("\nCreating comprehensive RQ2 visualization...")
    
    for gender in ['M', 'F']:
        gender_label = 'Men' if gender == 'M' else 'Women'
        gender_data = results_df[results_df['gender'] == gender].copy()
        
        if len(gender_data) == 0:
            continue
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Chart 1: Bar chart of mean improvement for 2023-2024 and 2024-2025
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Calculate means and standard errors
        imp_2023_2024 = gender_data['improvement_2023_to_2024_minutes'].dropna()
        imp_2024_2025 = gender_data['improvement_2024_to_2025_minutes'].dropna()
        
        if len(imp_2023_2024) > 0 and len(imp_2024_2025) > 0:
            mean_2023_2024 = imp_2023_2024.mean()
            mean_2024_2025 = imp_2024_2025.mean()
            se_2023_2024 = imp_2023_2024.sem() if len(imp_2023_2024) > 1 else 0
            se_2024_2025 = imp_2024_2025.sem() if len(imp_2024_2025) > 1 else 0
            
            periods = ['2023-2024', '2024-2025']
            means = [mean_2023_2024, mean_2024_2025]
            errors = [se_2023_2024, se_2024_2025]
            
            bars = ax1.bar(periods, means, yerr=errors, capsize=5, alpha=0.7, 
                          color=['#2E86AB', '#A23B72'], edgecolor='black', linewidth=1.5)
            ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax1.set_ylabel('Mean Improvement (minutes)\n(negative = faster)', fontsize=11)
            ax1.set_title(f'{gender_label} - Mean Improvement by Period', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, mean, error) in enumerate(zip(bars, means, errors)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + (error if height >= 0 else -error),
                        f'{mean:.2f}\n(n={len(imp_2023_2024) if i == 0 else len(imp_2024_2025)})',
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Chart 2: Improvement between 2023-2024 and 2024-2025 with error bars (box plot + bar comparison)
        ax2 = fig.add_subplot(gs[0, 1])
        
        if len(imp_2023_2024) > 0 and len(imp_2024_2025) > 0:
            # Create box plot for distribution
            box_data = [imp_2023_2024, imp_2024_2025]
            bp = ax2.boxplot(box_data, labels=['2023→2024', '2024→2025'], patch_artist=True,
                            widths=0.6, showmeans=True, meanline=True)
            
            # Color the boxes
            colors = ['#2E86AB', '#A23B72']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add mean bars with error bars on top
            x_pos = [1, 2]
            means = [mean_2023_2024, mean_2024_2025]
            errors = [se_2023_2024 * 1.96, se_2024_2025 * 1.96]  # 95% CI
            
            # Overlay error bars on the means
            ax2.errorbar(x_pos, means, yerr=errors, fmt='o', color='red', 
                        markersize=8, capsize=8, capthick=2, label='Mean ± 95% CI',
                        zorder=10)
            
            ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax2.set_ylabel('Improvement (minutes)\n(negative = faster)', fontsize=11)
            ax2.set_title(f'{gender_label} - Improvement Distribution with Error Bars', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend(fontsize=9)
            
            # Add sample size annotations
            for i, (x, mean, n) in enumerate(zip(x_pos, means, [len(imp_2023_2024), len(imp_2024_2025)])):
                ax2.text(x, ax2.get_ylim()[1] * 0.95, f'n={n}', ha='center', va='top', 
                        fontsize=9, fontweight='bold')
        
        # Chart 3: Distribution of fastest times across years
        ax3 = fig.add_subplot(gs[1, :])
        
        # Get best times for each year
        times_2023 = gender_data['best_time_2023_minutes'].dropna()
        times_2024 = gender_data['best_time_2024_minutes'].dropna()
        times_2025 = gender_data['best_time_2025_minutes'].dropna()
        
        if len(times_2023) > 0 or len(times_2024) > 0 or len(times_2025) > 0:
            # Create histogram for each year
            all_times = [times_2023, times_2024, times_2025]
            labels = ['2023', '2024', '2025']
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            # Determine bins based on all data
            all_values = pd.concat([t for t in all_times if len(t) > 0])
            bins = np.linspace(all_values.min(), all_values.max(), 30)
            
            for times, label, color in zip(all_times, labels, colors):
                if len(times) > 0:
                    ax3.hist(times, bins=bins, alpha=0.6, label=f'{label} (n={len(times)}, μ={times.mean():.2f})',
                            color=color, edgecolor='black', linewidth=0.5)
            
            ax3.set_xlabel('Best Time (minutes)', fontsize=11)
            ax3.set_ylabel('Number of Athletes', fontsize=11)
            ax3.set_title(f'{gender_label} - Distribution of Fastest Times by Year', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Add overall title
        fig.suptitle(f'RQ2 Analysis: {gender_label} - Multi-Season Performance', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure
        gender_lower = gender_label.lower()
        output_path = f'{output_dir}/rq2_comprehensive_{gender_lower}.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive RQ2 plot for {gender_label} to {output_path}")

def create_rq2_gender_comparison_plot(results_df, output_dir='output/rq2'):
    """
    Create a side-by-side comparison of men and women for general discussion.
    Shows key metrics and trends across both genders.
    """
    print("\nCreating gender comparison plot for general discussion...")
    
    # Prepare data
    men_data = results_df[results_df['gender'] == 'M'].copy()
    women_data = results_df[results_df['gender'] == 'F'].copy()
    
    if len(men_data) == 0 or len(women_data) == 0:
        print("Insufficient data for gender comparison")
        return
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Chart 1: Mean improvement comparison (2023-2024 and 2024-2025)
    ax1 = fig.add_subplot(gs[0, 0])
    
    men_imp_2023_2024 = men_data['improvement_2023_to_2024_minutes'].dropna()
    men_imp_2024_2025 = men_data['improvement_2024_to_2025_minutes'].dropna()
    women_imp_2023_2024 = women_data['improvement_2023_to_2024_minutes'].dropna()
    women_imp_2024_2025 = women_data['improvement_2024_to_2025_minutes'].dropna()
    
    x = np.arange(2)
    width = 0.35
    
    men_means = [men_imp_2023_2024.mean() if len(men_imp_2023_2024) > 0 else 0,
                 men_imp_2024_2025.mean() if len(men_imp_2024_2025) > 0 else 0]
    men_errors = [men_imp_2023_2024.sem() * 1.96 if len(men_imp_2023_2024) > 1 else 0,
                  men_imp_2024_2025.sem() * 1.96 if len(men_imp_2024_2025) > 1 else 0]
    women_means = [women_imp_2023_2024.mean() if len(women_imp_2023_2024) > 0 else 0,
                   women_imp_2024_2025.mean() if len(women_imp_2024_2025) > 0 else 0]
    women_errors = [women_imp_2023_2024.sem() * 1.96 if len(women_imp_2023_2024) > 1 else 0,
                    women_imp_2024_2025.sem() * 1.96 if len(women_imp_2024_2025) > 1 else 0]
    
    bars1 = ax1.bar(x - width/2, men_means, width, yerr=men_errors, capsize=5,
                   label='Men', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, women_means, width, yerr=women_errors, capsize=5,
                   label='Women', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Period', fontsize=11)
    ax1.set_ylabel('Mean Improvement (minutes)\n(negative = faster)', fontsize=11)
    ax1.set_title('Mean Improvement by Gender and Period', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['2023→2024', '2024→2025'])
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Chart 2: Sample sizes comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    men_n = [len(men_imp_2023_2024), len(men_imp_2024_2025)]
    women_n = [len(women_imp_2023_2024), len(women_imp_2024_2025)]
    
    bars1 = ax2.bar(x - width/2, men_n, width, label='Men', color='#2E86AB', 
                   alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, women_n, width, label='Women', color='#A23B72', 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Period', fontsize=11)
    ax2.set_ylabel('Number of Athletes', fontsize=11)
    ax2.set_title('Sample Size by Gender and Period', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['2023→2024', '2024→2025'])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Chart 3: Improvement rate distribution comparison
    ax3 = fig.add_subplot(gs[0, 2])
    
    men_imp_rate = men_data['improvement_rate_per_year_minutes'].dropna()
    women_imp_rate = women_data['improvement_rate_per_year_minutes'].dropna()
    
    if len(men_imp_rate) > 0 and len(women_imp_rate) > 0:
        ax3.hist(men_imp_rate, bins=25, alpha=0.6, label=f'Men (n={len(men_imp_rate)}, μ={men_imp_rate.mean():.2f})',
                color='#2E86AB', edgecolor='black', linewidth=0.5)
        ax3.hist(women_imp_rate, bins=25, alpha=0.6, label=f'Women (n={len(women_imp_rate)}, μ={women_imp_rate.mean():.2f})',
                color='#A23B72', edgecolor='black', linewidth=0.5)
        ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Improvement Rate (minutes/year)\n(negative = faster)', fontsize=11)
        ax3.set_ylabel('Number of Athletes', fontsize=11)
        ax3.set_title('Improvement Rate Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Chart 4: Best times comparison across years
    ax4 = fig.add_subplot(gs[1, :])
    
    men_times_2023 = men_data['best_time_2023_minutes'].dropna()
    men_times_2024 = men_data['best_time_2024_minutes'].dropna()
    men_times_2025 = men_data['best_time_2025_minutes'].dropna()
    women_times_2023 = women_data['best_time_2023_minutes'].dropna()
    women_times_2024 = women_data['best_time_2024_minutes'].dropna()
    women_times_2025 = women_data['best_time_2025_minutes'].dropna()
    
    years = [2023, 2024, 2025]
    men_means = [men_times_2023.mean() if len(men_times_2023) > 0 else np.nan,
                 men_times_2024.mean() if len(men_times_2024) > 0 else np.nan,
                 men_times_2025.mean() if len(men_times_2025) > 0 else np.nan]
    men_stds = [men_times_2023.std() if len(men_times_2023) > 1 else 0,
                men_times_2024.std() if len(men_times_2024) > 1 else 0,
                men_times_2025.std() if len(men_times_2025) > 1 else 0]
    women_means = [women_times_2023.mean() if len(women_times_2023) > 0 else np.nan,
                   women_times_2024.mean() if len(women_times_2024) > 0 else np.nan,
                   women_times_2025.mean() if len(women_times_2025) > 0 else np.nan]
    women_stds = [women_times_2023.std() if len(women_times_2023) > 1 else 0,
                  women_times_2024.std() if len(women_times_2024) > 1 else 0,
                  women_times_2025.std() if len(women_times_2025) > 1 else 0]
    
    ax4.errorbar(years, men_means, yerr=men_stds, marker='o', linewidth=2, 
                markersize=8, capsize=5, label='Men', color='#2E86AB', alpha=0.8)
    ax4.errorbar(years, women_means, yerr=women_stds, marker='s', linewidth=2,
                markersize=8, capsize=5, label='Women', color='#A23B72', alpha=0.8)
    
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Mean Best Time (minutes)', fontsize=11)
    ax4.set_title('Mean Best Times Across Years (with Standard Deviation)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(years)
    
    # Chart 5: Percent improved comparison
    ax5 = fig.add_subplot(gs[2, 0])
    
    men_improved = (men_data['improvement_2023_to_2025_minutes'].dropna() < 0).sum()
    men_total = len(men_data['improvement_2023_to_2025_minutes'].dropna())
    men_pct = (men_improved / men_total * 100) if men_total > 0 else 0
    
    women_improved = (women_data['improvement_2023_to_2025_minutes'].dropna() < 0).sum()
    women_total = len(women_data['improvement_2023_to_2025_minutes'].dropna())
    women_pct = (women_improved / women_total * 100) if women_total > 0 else 0
    
    categories = ['Men', 'Women']
    pcts = [men_pct, women_pct]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax5.bar(categories, pcts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Percent Improved (%)', fontsize=11)
    ax5.set_title('Percentage of Athletes Who Improved\n(2023→2025)', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, pct, n_imp, n_tot in zip(bars, pcts, [men_improved, women_improved], [men_total, women_total]):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{pct:.1f}%\n({n_imp}/{n_tot})', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Chart 6: Summary statistics table
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    
    # Men statistics
    if len(men_imp_2023_2024) > 0:
        summary_data.append(['Men', '2023→2024', f'{men_imp_2023_2024.mean():.2f}', 
                           f'±{men_imp_2023_2024.sem()*1.96:.2f}', len(men_imp_2023_2024)])
    if len(men_imp_2024_2025) > 0:
        summary_data.append(['Men', '2024→2025', f'{men_imp_2024_2025.mean():.2f}',
                           f'±{men_imp_2024_2025.sem()*1.96:.2f}', len(men_imp_2024_2025)])
    if len(men_imp_rate) > 0:
        summary_data.append(['Men', '2023→2025 Rate', f'{men_imp_rate.mean():.2f}',
                           f'±{men_imp_rate.sem()*1.96:.2f}', len(men_imp_rate)])
    
    # Women statistics
    if len(women_imp_2023_2024) > 0:
        summary_data.append(['Women', '2023→2024', f'{women_imp_2023_2024.mean():.2f}',
                           f'±{women_imp_2023_2024.sem()*1.96:.2f}', len(women_imp_2023_2024)])
    if len(women_imp_2024_2025) > 0:
        summary_data.append(['Women', '2024→2025', f'{women_imp_2024_2025.mean():.2f}',
                           f'±{women_imp_2024_2025.sem()*1.96:.2f}', len(women_imp_2024_2025)])
    if len(women_imp_rate) > 0:
        summary_data.append(['Women', '2023→2025 Rate', f'{women_imp_rate.mean():.2f}',
                           f'±{women_imp_rate.sem()*1.96:.2f}', len(women_imp_rate)])
    
    if summary_data:
        table = ax6.table(cellText=summary_data,
                         colLabels=['Gender', 'Period', 'Mean (min)', '95% CI', 'N'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.25, 0.2, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(5):
            table[(0, i)].set_facecolor('#4A90E2')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows
        for i in range(1, len(summary_data) + 1):
            if summary_data[i-1][0] == 'Men':
                color = '#E3F2FD'
            else:
                color = '#FCE4EC'
            for j in range(5):
                table[(i, j)].set_facecolor(color)
        
        ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('RQ2: Gender Comparison - Multi-Season Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = f'{output_dir}/rq2_gender_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved gender comparison plot to {output_path}")

def analyze_athletes_all_seasons(df, valid_athlete_ids, output_dir='output/rq2'):
    """
    Analyze athletes across all seasons and calculate improvement metrics.
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    valid_athlete_ids : set
        Set of athlete IDs that passed the race count filter
    output_dir : str
        Output directory for results
    """
    print("\n" + "="*60)
    print("ANALYZING ATHLETES ACROSS ALL SEASONS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to valid athletes
    df_filtered = df[df['athlete_id'].isin(valid_athlete_ids)].copy()
    df_filtered['year'] = pd.to_datetime(df_filtered['start_date']).dt.year
    
    # Get athlete and gender info (only merge if gender column doesn't exist)
    if 'gender' not in df_filtered.columns:
        athlete_df = pd.read_csv('data/athlete.csv')
        df_filtered = df_filtered.merge(athlete_df[['athlete_id', 'gender']], on='athlete_id', how='left')
    
    # Ensure gender column exists
    if 'gender' not in df_filtered.columns:
        raise ValueError("Gender column not found in dataframe and could not be merged")
    
    # Analyze each athlete across seasons
    athlete_results = []
    
    for athlete_id in valid_athlete_ids:
        athlete_data = df_filtered[df_filtered['athlete_id'] == athlete_id].copy()
        
        if len(athlete_data) == 0:
            continue
        
        gender = athlete_data['gender'].iloc[0] if 'gender' in athlete_data.columns else 'Unknown'
        gender_label = 'Men' if gender == 'M' else 'Women' if gender == 'F' else 'Unknown'
        
        # Get best time for each year
        best_times = {}
        for year in [2023, 2024, 2025]:
            year_data = athlete_data[athlete_data['year'] == year]
            if len(year_data) > 0:
                best_times[year] = year_data['standardized_to_target'].min()
            else:
                best_times[year] = np.nan
        
        # Calculate improvements
        improvements = {}
        if not pd.isna(best_times.get(2023)) and not pd.isna(best_times.get(2024)):
            improvements['2023_to_2024'] = best_times[2024] - best_times[2023]  # negative = faster
        else:
            improvements['2023_to_2024'] = np.nan
            
        if not pd.isna(best_times.get(2024)) and not pd.isna(best_times.get(2025)):
            improvements['2024_to_2025'] = best_times[2025] - best_times[2024]
        else:
            improvements['2024_to_2025'] = np.nan
            
        if not pd.isna(best_times.get(2023)) and not pd.isna(best_times.get(2025)):
            improvements['2023_to_2025'] = best_times[2025] - best_times[2023]
        else:
            improvements['2023_to_2025'] = np.nan
        
        # Calculate improvement rate per year (for 2023 to 2025)
        if not pd.isna(improvements['2023_to_2025']):
            improvement_rate = improvements['2023_to_2025'] / 2  # 2 years
        else:
            improvement_rate = np.nan
        
        athlete_results.append({
            'athlete_id': athlete_id,
            'gender': gender,
            'gender_label': gender_label,
            'best_time_2023': best_times.get(2023, np.nan),
            'best_time_2024': best_times.get(2024, np.nan),
            'best_time_2025': best_times.get(2025, np.nan),
            'best_time_2023_minutes': best_times.get(2023, np.nan) / 60 if not pd.isna(best_times.get(2023)) else np.nan,
            'best_time_2024_minutes': best_times.get(2024, np.nan) / 60 if not pd.isna(best_times.get(2024)) else np.nan,
            'best_time_2025_minutes': best_times.get(2025, np.nan) / 60 if not pd.isna(best_times.get(2025)) else np.nan,
            'improvement_2023_to_2024_seconds': improvements.get('2023_to_2024', np.nan),
            'improvement_2023_to_2024_minutes': improvements.get('2023_to_2024', np.nan) / 60 if not pd.isna(improvements.get('2023_to_2024')) else np.nan,
            'improvement_2024_to_2025_seconds': improvements.get('2024_to_2025', np.nan),
            'improvement_2024_to_2025_minutes': improvements.get('2024_to_2025', np.nan) / 60 if not pd.isna(improvements.get('2024_to_2025')) else np.nan,
            'improvement_2023_to_2025_seconds': improvements.get('2023_to_2025', np.nan),
            'improvement_2023_to_2025_minutes': improvements.get('2023_to_2025', np.nan) / 60 if not pd.isna(improvements.get('2023_to_2025')) else np.nan,
            'improvement_rate_per_year_seconds': improvement_rate,
            'improvement_rate_per_year_minutes': improvement_rate / 60 if not pd.isna(improvement_rate) else np.nan,
        })
    
    results_df = pd.DataFrame(athlete_results)
    
    # Save detailed results
    results_df.to_csv(f'{output_dir}/athletes_all_seasons_detailed.csv', index=False)
    print(f"Saved detailed results to {output_dir}/athletes_all_seasons_detailed.csv")
    
    # Create summary by gender
    summary_rows = []
    for gender in ['M', 'F']:
        gender_label = 'Men' if gender == 'M' else 'Women'
        gender_data = results_df[results_df['gender'] == gender]
        
        if len(gender_data) == 0:
            continue
        
        # Filter out NaN values for calculations
        valid_2023_2024 = gender_data['improvement_2023_to_2024_minutes'].dropna()
        valid_2024_2025 = gender_data['improvement_2024_to_2025_minutes'].dropna()
        valid_2023_2025 = gender_data['improvement_2023_to_2025_minutes'].dropna()
        
        summary_rows.append({
            'gender': gender_label,
            'num_athletes': len(gender_data),
            'mean_improvement_2023_to_2024_minutes': valid_2023_2024.mean() if len(valid_2023_2024) > 0 else np.nan,
            'mean_improvement_2024_to_2025_minutes': valid_2024_2025.mean() if len(valid_2024_2025) > 0 else np.nan,
            'mean_improvement_2023_to_2025_minutes': valid_2023_2025.mean() if len(valid_2023_2025) > 0 else np.nan,
            'median_improvement_2023_to_2025_minutes': valid_2023_2025.median() if len(valid_2023_2025) > 0 else np.nan,
            'std_improvement_2023_to_2025_minutes': valid_2023_2025.std() if len(valid_2023_2025) > 0 else np.nan,
            'percent_improved_2023_to_2025': (valid_2023_2025 < 0).sum() / len(valid_2023_2025) * 100 if len(valid_2023_2025) > 0 else np.nan,
            'percent_declined_2023_to_2025': (valid_2023_2025 > 0).sum() / len(valid_2023_2025) * 100 if len(valid_2023_2025) > 0 else np.nan,
            'mean_2023_best_time_minutes': gender_data['best_time_2023_minutes'].dropna().mean() if len(gender_data['best_time_2023_minutes'].dropna()) > 0 else np.nan,
            'mean_2025_best_time_minutes': gender_data['best_time_2025_minutes'].dropna().mean() if len(gender_data['best_time_2025_minutes'].dropna()) > 0 else np.nan,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{output_dir}/athletes_all_seasons_summary.csv', index=False)
    print(f"Saved summary to {output_dir}/athletes_all_seasons_summary.csv")
    
    # Create visualization
    create_improvement_plot(results_df, output_dir)
    
    # Create comprehensive RQ2 plots with 3 charts
    create_rq2_comprehensive_plots(results_df, output_dir)
    
    # Create gender comparison plot for general discussion
    create_rq2_gender_comparison_plot(results_df, output_dir)
    
    return results_df, summary_df

def main():
    """Standalone execution for testing."""
    print("="*60)
    print("RQ2 MULTI-SEASON ANALYSIS - STANDALONE EXECUTION")
    print("="*60)
    print("\nNote: This script is typically called from rq2.py")
    print("For standalone execution, you need to provide filtered data.")
    print("\nRun: python scripts/rq2.py (instead)")

if __name__ == "__main__":
    main()

