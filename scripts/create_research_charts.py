import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
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

def load_and_prepare_data():
    """Load raw data and prepare features."""
    print("Loading data for research charts...")
    
    df = standardize_convert_exclude_nationals_df()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df = df.dropna(subset=['standardized_to_target', 'start_date', 'gender', 'athlete_id'])
    
    return df

def calculate_athlete_features(df):
    """Calculate features for each athlete."""
    print("Calculating athlete features...")
    
    athlete_features = []
    athletes = df['athlete_id'].unique()
    
    for i, athlete_id in enumerate(athletes):
        if i % 1000 == 0:
            print(f"  Processing athlete {i+1}/{len(athletes)}")
        
        athlete_races = df[df['athlete_id'] == athlete_id].sort_values('start_date')
        
        if len(athlete_races) < 2:
            continue
        
        first_time = athlete_races.iloc[0]['standardized_to_target']
        last_time = athlete_races.iloc[-1]['standardized_to_target']
        first_date = athlete_races.iloc[0]['start_date']
        last_date = athlete_races.iloc[-1]['start_date']
        
        if pd.isna(first_time) or pd.isna(last_time):
            continue
        
        total_improvement = last_time - first_time
        days_diff = (last_date - first_date).days
        
        if days_diff <= 0:
            continue
        
        improvement_rate = total_improvement / days_diff
        
        num_races = len(athlete_races)
        season_duration = days_diff
        times = athlete_races['standardized_to_target'].values
        best_time = np.min(times)
        worst_time = np.max(times)
        avg_time = np.mean(times)
        time_std = np.std(times)
        time_range = worst_time - best_time
        cv_time = time_std / avg_time if avg_time > 0 else 0
        
        if len(times) >= 3:
            X = np.arange(len(times)).reshape(-1, 1)
            y = times
            slope_model = LinearRegression()
            slope_model.fit(X, y)
            slope = slope_model.coef_[0]
        else:
            slope = improvement_rate
        
        race_frequency = num_races / season_duration if season_duration > 0 else 0
        
        if len(times) >= 3:
            mid_point = len(times) // 2
            first_half_avg = np.mean(times[:mid_point])
            second_half_avg = np.mean(times[mid_point:])
            progression_improvement = first_half_avg - second_half_avg
        else:
            progression_improvement = total_improvement
        
        gender = athlete_races.iloc[0]['gender']
        year = athlete_races.iloc[0]['start_date'].year
        
        year_gender_df = df[(df['start_date'].dt.year == year) & (df['gender'] == gender)]
        if len(year_gender_df) > 0:
            starting_percentile = (year_gender_df['standardized_to_target'] <= first_time).mean() * 100
        else:
            starting_percentile = 50
        
        athlete_features.append({
            'athlete_id': athlete_id,
            'gender': gender,
            'year': year,
            'num_races': num_races,
            'season_duration': season_duration,
            'first_time': first_time,
            'last_time': last_time,
            'best_time': best_time,
            'worst_time': worst_time,
            'avg_time': avg_time,
            'time_std': time_std,
            'time_range': time_range,
            'cv_time': cv_time,
            'total_improvement': total_improvement,
            'improvement_rate': improvement_rate,
            'slope': slope,
            'race_frequency': race_frequency,
            'progression_improvement': progression_improvement,
            'starting_percentile': starting_percentile
        })
    
    return pd.DataFrame(athlete_features)

def create_features(athlete_df):
    """Create advanced features."""
    features_df = athlete_df.copy()
    
    le_gender = LabelEncoder()
    features_df['gender_encoded'] = le_gender.fit_transform(features_df['gender'])
    
    features_df['gender_year'] = features_df['gender_encoded'] * features_df['year']
    features_df['races_duration_ratio'] = features_df['num_races'] / features_df['season_duration']
    features_df['improvement_per_race'] = features_df['total_improvement'] / features_df['num_races']
    
    features_df['starting_percentile_squared'] = features_df['starting_percentile'] ** 2
    features_df['num_races_squared'] = features_df['num_races'] ** 2
    features_df['season_duration_squared'] = features_df['season_duration'] ** 2
    
    features_df['best_to_avg_ratio'] = features_df['best_time'] / features_df['avg_time']
    features_df['worst_to_avg_ratio'] = features_df['worst_time'] / features_df['avg_time']
    
    features_df['improvement_efficiency'] = features_df['total_improvement'] / features_df['time_range']
    features_df['consistency_score'] = 1 / (1 + features_df['cv_time'])
    
    features_df['early_season_performance'] = features_df['first_time']
    features_df['late_season_performance'] = features_df['last_time']
    
    features_df['experience_level'] = features_df['num_races'] * features_df['season_duration']
    
    return features_df

def create_model_performance_chart(results_df):
    """Create model performance comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² Score comparison
    models = results_df['Model']
    r2_scores = results_df['Test_R2_Score']
    cv_r2_scores = results_df['CV_R2_Mean']
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cv_r2_scores, width, label='CV R² (2023)', 
                     color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, r2_scores, width, label='Test R² (2024)', 
                     color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Overfitting analysis
    overfitting_scores = results_df['Overfitting_Score']
    colors = ['red' if score > 0.1 else 'green' if score < -0.1 else 'orange' for score in overfitting_scores]
    
    bars3 = ax2.bar(models, overfitting_scores, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting threshold')
    ax2.axhline(y=-0.1, color='green', linestyle='--', alpha=0.7, label='Good generalization')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Overfitting Score\n(CV R² - Test R²)')
    ax2.set_title('Overfitting Analysis')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/research_model_performance.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_chart(feature_importance_df):
    """Create feature importance chart."""
    plt.figure(figsize=(12, 8))
    
    # Select top 15 features
    top_features = feature_importance_df.head(15)
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(top_features)), top_features['importance'], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features for Improvement Prediction')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(importance + 0.001, i, f'{importance:.3f}', 
                va='center', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('output/research_feature_importance.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_prediction_scatter_plot(results, best_model_name):
    """Create prediction vs actual scatter plot."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot actual vs predicted
        ax.scatter(result['y_test'], result['y_pred'], alpha=0.6, s=30)
        
        # Add perfect prediction line
        min_val = min(result['y_test'].min(), result['y_pred'].min())
        max_val = max(result['y_test'].max(), result['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Actual Improvement Rate (seconds/day)')
        ax.set_ylabel('Predicted Improvement Rate (seconds/day)')
        ax.set_title(f'{name}\nR² = {result["r2"]:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Add R² text box
        ax.text(0.05, 0.95, f'R² = {result["r2"]:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('output/research_prediction_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_distribution_chart(features_df):
    """Create improvement distribution charts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall improvement distribution
    ax1.hist(features_df['improvement_rate'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(features_df['improvement_rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {features_df["improvement_rate"].mean():.2f}')
    ax1.set_xlabel('Improvement Rate (seconds/day)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Improvement Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement by gender
    gender_data = [features_df[features_df['gender'] == 'M']['improvement_rate'],
                   features_df[features_df['gender'] == 'F']['improvement_rate']]
    ax2.boxplot(gender_data, labels=['Men', 'Women'])
    ax2.set_ylabel('Improvement Rate (seconds/day)')
    ax2.set_title('Improvement Rates by Gender')
    ax2.grid(True, alpha=0.3)
    
    # Improvement by year
    year_data = [features_df[features_df['year'] == 2023]['improvement_rate'],
                 features_df[features_df['year'] == 2024]['improvement_rate']]
    ax3.boxplot(year_data, labels=['2023', '2024'])
    ax3.set_ylabel('Improvement Rate (seconds/day)')
    ax3.set_title('Improvement Rates by Year')
    ax3.grid(True, alpha=0.3)
    
    # Improvement by starting percentile
    percentile_bins = [0, 25, 50, 75, 100]
    percentile_means = []
    percentile_labels = []
    
    for i in range(len(percentile_bins)-1):
        mask = (features_df['starting_percentile'] >= percentile_bins[i]) & (features_df['starting_percentile'] < percentile_bins[i+1])
        subset = features_df[mask]
        if len(subset) > 0:
            percentile_means.append(subset['improvement_rate'].mean())
            percentile_labels.append(f'{percentile_bins[i]}-{percentile_bins[i+1]}%')
    
    bars = ax4.bar(percentile_labels, percentile_means, color=plt.cm.viridis(np.linspace(0, 1, len(percentile_means))))
    ax4.set_ylabel('Average Improvement Rate (seconds/day)')
    ax4.set_title('Improvement by Starting Performance Level')
    ax4.set_xticklabels(percentile_labels, rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean_val in zip(bars, percentile_means):
        ax4.text(bar.get_x() + bar.get_width()/2., mean_val + (0.01 if mean_val >= 0 else -0.01),
                f'{mean_val:.2f}', ha='center', va='bottom' if mean_val >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('output/research_improvement_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_formula_visualization():
    """Create a visualization of the model formula."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define the formula components
    components = [
        'Baseline (-0.6)',
        'Improvement per Race (0.40)',
        'Race Frequency (0.11)',
        'Season Duration (0.10)',
        'Races/Duration Ratio (0.10)',
        'Season Duration² (0.09)',
        'Experience Level (0.08)',
        'Progression (0.03)',
        'Efficiency (0.02)',
        'CV Time (0.02)',
        'Consistency (0.02)'
    ]
    
    weights = [0.6, 0.40, 0.11, 0.10, 0.10, 0.09, 0.08, 0.03, 0.02, 0.02, 0.02]
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(components)), weights, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(components))))
    
    plt.yticks(range(len(components)), components)
    plt.xlabel('Feature Weight in Model')
    plt.title('Model Formula Components\nImprovement Rate = Σ(Weight × Feature)')
    
    # Add value labels
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        plt.text(weight + 0.001, i, f'{weight:.2f}', 
                va='center', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('output/research_model_formula.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all research charts."""
    print("="*60)
    print("CREATING RESEARCH CHARTS")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    athlete_df = calculate_athlete_features(df)
    features_df = create_features(athlete_df)
    
    # Load temporal validation results
    results_df = pd.read_csv('output/temporal_validation_results.csv')
    
    # Load feature importance
    feature_importance_df = pd.read_csv('output/raw_data_feature_importance.csv')
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    print("\nCreating research charts...")
    
    # 1. Model performance comparison
    print("1. Creating model performance chart...")
    create_model_performance_chart(results_df)
    
    # 2. Feature importance
    print("2. Creating feature importance chart...")
    create_feature_importance_chart(feature_importance_df)
    
    # 3. Prediction scatter plots
    print("3. Creating prediction scatter plots...")
    # We need to recreate the results for scatter plots
    # For now, let's create the distribution charts
    
    # 4. Improvement distributions
    print("4. Creating improvement distribution charts...")
    create_improvement_distribution_chart(features_df)
    
    # 5. Model formula visualization
    print("5. Creating model formula visualization...")
    create_formula_visualization()
    
    print("\nAll research charts saved to output/ directory:")
    print("- research_model_performance.pdf")
    print("- research_feature_importance.pdf") 
    print("- research_improvement_distributions.pdf")
    print("- research_model_formula.pdf")
    
    print("\nCharts created successfully!")

if __name__ == "__main__":
    main() 