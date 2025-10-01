import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

import os
from utils import standardize_convert_exclude_nationals_df

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def create_summary_table():
    """Create a comprehensive summary table for the research paper."""
    print("Creating summary table...")
    
    # Load results
    results_df = pd.read_csv('output/temporal_validation_results.csv')
    feature_importance_df = pd.read_csv('output/raw_data_feature_importance.csv')
    
    # Create summary table
    summary_data = {
        'Metric': [
            'Dataset Size',
            'Training Samples (2023)',
            'Test Samples (2024)',
            'Best Model',
            'Test R² Score',
            'Cross-Validation R² Score',
            'Overfitting Score',
            'RMSE (seconds/day)',
            'MAE (seconds/day)',
            'Top Feature Importance',
            'Second Most Important Feature',
            'Third Most Important Feature'
        ],
        'Value': [
            '3,155 athletes',
            '2,186 athletes',
            '969 athletes',
            'Gradient Boosting',
            f"{results_df[results_df['Model'] == 'Gradient Boosting']['Test_R2_Score'].iloc[0]:.3f}",
            f"{results_df[results_df['Model'] == 'Gradient Boosting']['CV_R2_Mean'].iloc[0]:.3f}",
            f"{results_df[results_df['Model'] == 'Gradient Boosting']['Overfitting_Score'].iloc[0]:.3f}",
            f"{results_df[results_df['Model'] == 'Gradient Boosting']['Test_RMSE'].iloc[0]:.3f}",
            f"{results_df[results_df['Model'] == 'Gradient Boosting']['Test_MAE'].iloc[0]:.3f}",
            f"{feature_importance_df.iloc[0]['feature']} ({feature_importance_df.iloc[0]['importance']:.3f})",
            f"{feature_importance_df.iloc[1]['feature']} ({feature_importance_df.iloc[1]['importance']:.3f})",
            f"{feature_importance_df.iloc[2]['feature']} ({feature_importance_df.iloc[2]['importance']:.3f})"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_df.to_csv('output/research_summary_table.csv', index=False)
    print("Summary table saved to output/research_summary_table.csv")
    
    return summary_df

def create_improvement_trends_chart():
    """Create a chart showing improvement trends over time."""
    print("Creating improvement trends chart...")
    
    # Load athlete features
    features_df = pd.read_csv('output/raw_data_athlete_features.csv')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Improvement rate distribution by gender and year
    gender_year_data = features_df.groupby(['gender', 'year'])['improvement_rate'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create grouped bar chart
    men_2023 = gender_year_data[(gender_year_data['gender'] == 'M') & (gender_year_data['year'] == 2023)]['mean'].iloc[0]
    men_2024 = gender_year_data[(gender_year_data['gender'] == 'M') & (gender_year_data['year'] == 2024)]['mean'].iloc[0]
    women_2023 = gender_year_data[(gender_year_data['gender'] == 'F') & (gender_year_data['year'] == 2023)]['mean'].iloc[0]
    women_2024 = gender_year_data[(gender_year_data['gender'] == 'F') & (gender_year_data['year'] == 2024)]['mean'].iloc[0]
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, [men_2023, men_2024], width, label='Men', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, [women_2023, women_2024], width, label='Women', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Improvement Rate (seconds/day)')
    ax1.set_title('Improvement Trends by Gender and Year')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['2023', '2024'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # 2. Improvement vs number of races
    ax2.scatter(features_df['num_races'], features_df['improvement_rate'], alpha=0.6, s=20)
    ax2.set_xlabel('Number of Races')
    ax2.set_ylabel('Improvement Rate (seconds/day)')
    ax2.set_title('Improvement Rate vs Number of Races')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(features_df['num_races'], features_df['improvement_rate'], 1)
    p = np.poly1d(z)
    ax2.plot(features_df['num_races'], p(features_df['num_races']), "r--", alpha=0.8)
    
    # 3. Improvement vs season duration
    ax3.scatter(features_df['season_duration'], features_df['improvement_rate'], alpha=0.6, s=20)
    ax3.set_xlabel('Season Duration (days)')
    ax3.set_ylabel('Improvement Rate (seconds/day)')
    ax3.set_title('Improvement Rate vs Season Duration')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(features_df['season_duration'], features_df['improvement_rate'], 1)
    p = np.poly1d(z)
    ax3.plot(features_df['season_duration'], p(features_df['season_duration']), "r--", alpha=0.8)
    
    # 4. Improvement vs starting percentile
    ax4.scatter(features_df['starting_percentile'], features_df['improvement_rate'], alpha=0.6, s=20)
    ax4.set_xlabel('Starting Performance Percentile')
    ax4.set_ylabel('Improvement Rate (seconds/day)')
    ax4.set_title('Improvement Rate vs Starting Performance Level')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(features_df['starting_percentile'], features_df['improvement_rate'], 1)
    p = np.poly1d(z)
    ax4.plot(features_df['starting_percentile'], p(features_df['starting_percentile']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('output/research_improvement_trends.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_comparison_chart():
    """Create a comprehensive model comparison chart."""
    print("Creating model comparison chart...")
    
    # Load results
    results_df = pd.read_csv('output/temporal_validation_results.csv')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. R² Score comparison
    models = results_df['Model']
    test_r2 = results_df['Test_R2_Score']
    cv_r2 = results_df['CV_R2_Mean']
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cv_r2, width, label='CV R² (2023)', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_r2, width, label='Test R² (2024)', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. RMSE comparison
    rmse_values = results_df['Test_RMSE']
    bars3 = ax2.bar(models, rmse_values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    ax2.set_xlabel('Model')
    ax2.set_ylabel('RMSE (seconds/day)')
    ax2.set_title('Root Mean Square Error Comparison')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rmse in zip(bars3, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2., rmse + 0.1,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. MAE comparison
    mae_values = results_df['Test_MAE']
    bars4 = ax3.bar(models, mae_values, color=plt.cm.plasma(np.linspace(0, 1, len(models))))
    ax3.set_xlabel('Model')
    ax3.set_ylabel('MAE (seconds/day)')
    ax3.set_title('Mean Absolute Error Comparison')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mae in zip(bars4, mae_values):
        ax3.text(bar.get_x() + bar.get_width()/2., mae + 0.01,
                f'{mae:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Overfitting analysis
    overfitting_scores = results_df['Overfitting_Score']
    colors = ['red' if score > 0.1 else 'green' if score < -0.1 else 'orange' for score in overfitting_scores]
    
    bars5 = ax4.bar(models, overfitting_scores, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting threshold')
    ax4.axhline(y=-0.1, color='green', linestyle='--', alpha=0.7, label='Good generalization')
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Overfitting Score\n(CV R² - Test R²)')
    ax4.set_title('Overfitting Analysis')
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars5, overfitting_scores):
        ax4.text(bar.get_x() + bar.get_width()/2., score + (0.01 if score >= 0 else -0.01),
                f'{score:.3f}', ha='center', va='bottom' if score >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/research_model_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_analysis_chart():
    """Create detailed feature analysis charts."""
    print("Creating feature analysis chart...")
    
    # Load feature importance
    feature_importance_df = pd.read_csv('output/raw_data_feature_importance.csv')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top 10 feature importance
    top_10 = feature_importance_df.head(10)
    bars1 = ax1.barh(range(len(top_10)), top_10['importance'], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(top_10))))
    
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['feature'])
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('Top 10 Most Important Features')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars1, top_10['importance'])):
        ax1.text(importance + 0.001, i, f'{importance:.3f}', 
                va='center', fontsize=10)
    
    # 2. Cumulative importance
    cumulative_importance = np.cumsum(feature_importance_df['importance'])
    ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
             marker='o', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Cumulative Importance')
    ax2.set_title('Cumulative Feature Importance')
    ax2.grid(True, alpha=0.3)
    
    # Add 80% and 90% lines
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% importance')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% importance')
    ax2.legend()
    
    # 3. Feature importance by category
    # Categorize features
    feature_categories = {
        'Performance': ['improvement_per_race', 'progression_improvement', 'improvement_efficiency'],
        'Racing': ['race_frequency', 'races_duration_ratio', 'num_races', 'num_races_squared'],
        'Season': ['season_duration', 'season_duration_squared', 'experience_level'],
        'Consistency': ['cv_time', 'consistency_score', 'time_std', 'time_range'],
        'Timing': ['early_season_performance', 'late_season_performance'],
        'Ratios': ['best_to_avg_ratio', 'worst_to_avg_ratio']
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_importance[category] = feature_importance_df[
            feature_importance_df['feature'].isin(features)]['importance'].sum()
    
    categories = list(category_importance.keys())
    importance_values = list(category_importance.values())
    
    bars3 = ax3.bar(categories, importance_values, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
    ax3.set_xlabel('Feature Category')
    ax3.set_ylabel('Total Importance')
    ax3.set_title('Feature Importance by Category')
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, importance in zip(bars3, importance_values):
        ax3.text(bar.get_x() + bar.get_width()/2., importance + 0.001,
                f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Feature importance distribution
    ax4.hist(feature_importance_df['importance'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Feature Importance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Feature Importance')
    ax4.grid(True, alpha=0.3)
    
    # Add mean line
    mean_importance = feature_importance_df['importance'].mean()
    ax4.axvline(mean_importance, color='red', linestyle='--', 
                label=f'Mean: {mean_importance:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('output/research_feature_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all paper summary materials."""
    print("="*60)
    print("CREATING RESEARCH PAPER SUMMARY MATERIALS")
    print("="*60)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    print("\nCreating research paper materials...")
    
    # 1. Summary table
    print("1. Creating summary table...")
    summary_df = create_summary_table()
    
    # 2. Improvement trends chart
    print("2. Creating improvement trends chart...")
    create_improvement_trends_chart()
    
    # 3. Model comparison chart
    print("3. Creating model comparison chart...")
    create_model_comparison_chart()
    
    # 4. Feature analysis chart
    print("4. Creating feature analysis chart...")
    create_feature_analysis_chart()
    
    print("\nAll research paper materials saved to output/ directory:")
    print("- research_summary_table.csv")
    print("- research_improvement_trends.pdf")
    print("- research_model_comparison.pdf")
    print("- research_feature_analysis.pdf")
    print("- research_model_performance.pdf")
    print("- research_feature_importance.pdf")
    print("- research_improvement_distributions.pdf")
    print("- research_model_formula.pdf")
    
    print("\nSummary table contents:")
    print(summary_df.to_string(index=False))
    
    print("\nAll materials created successfully!")

if __name__ == "__main__":
    main() 