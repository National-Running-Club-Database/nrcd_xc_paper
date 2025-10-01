import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

import os
from utils import standardize_convert_exclude_nationals_df, convert_exclude_nationals

def load_raw_data():
    """Load raw athlete data for improvement prediction."""
    print("Loading raw data...")
    
    # Load standardized data (this includes all the raw athlete records)
    df = standardize_convert_exclude_nationals_df()
    
    # Ensure dates are datetime
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter for valid data
    df = df.dropna(subset=['standardized_to_target', 'start_date', 'gender', 'athlete_id'])
    
    print(f"Loaded {len(df)} raw athlete records")
    return df

def calculate_athlete_features(df):
    """Calculate features for each athlete based on their race history."""
    print("Calculating athlete features...")
    
    athlete_features = []
    
    # Get unique athletes
    athletes = df['athlete_id'].unique()
    print(f"Processing {len(athletes)} athletes...")
    
    for i, athlete_id in enumerate(athletes):
        if i % 1000 == 0:
            print(f"  Processing athlete {i+1}/{len(athletes)}")
        
        # Get all races for this athlete
        athlete_races = df[df['athlete_id'] == athlete_id].sort_values('start_date')
        
        if len(athlete_races) < 2:
            continue
        
        # Calculate improvement metrics
        first_time = athlete_races.iloc[0]['standardized_to_target']
        last_time = athlete_races.iloc[-1]['standardized_to_target']
        first_date = athlete_races.iloc[0]['start_date']
        last_date = athlete_races.iloc[-1]['start_date']
        
        if pd.isna(first_time) or pd.isna(last_time):
            continue
        
        # Calculate improvement
        total_improvement = last_time - first_time  # negative = getting faster
        days_diff = (last_date - first_date).days
        
        if days_diff <= 0:
            continue
        
        improvement_rate = total_improvement / days_diff  # seconds per day
        
        # Calculate additional features
        num_races = len(athlete_races)
        season_duration = days_diff
        
        # Calculate performance statistics
        times = athlete_races['standardized_to_target'].values
        best_time = np.min(times)
        worst_time = np.max(times)
        avg_time = np.mean(times)
        time_std = np.std(times)
        
        # Calculate consistency metrics
        time_range = worst_time - best_time
        cv_time = time_std / avg_time if avg_time > 0 else 0
        
        # Calculate improvement pattern (linear regression slope)
        if len(times) >= 3:
            # Use all races for slope calculation
            X = np.arange(len(times)).reshape(-1, 1)
            y = times
            slope_model = LinearRegression()
            slope_model.fit(X, y)
            slope = slope_model.coef_[0]
        else:
            slope = improvement_rate
        
        # Calculate race frequency
        race_frequency = num_races / season_duration if season_duration > 0 else 0
        
        # Calculate performance progression
        if len(times) >= 3:
            # Calculate improvement from first half to second half
            mid_point = len(times) // 2
            first_half_avg = np.mean(times[:mid_point])
            second_half_avg = np.mean(times[mid_point:])
            progression_improvement = first_half_avg - second_half_avg
        else:
            progression_improvement = total_improvement
        
        # Extract athlete metadata
        gender = athlete_races.iloc[0]['gender']
        year = athlete_races.iloc[0]['start_date'].year
        
        # Calculate percentile of starting performance within gender/year
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
    
    print(f"Calculated features for {len(athlete_features)} athletes")
    return pd.DataFrame(athlete_features)

def create_advanced_features(athlete_df):
    """Create advanced features for the machine learning model."""
    print("Creating advanced features...")
    
    features_df = athlete_df.copy()
    
    # Create categorical encodings
    le_gender = LabelEncoder()
    features_df['gender_encoded'] = le_gender.fit_transform(features_df['gender'])
    
    # Create interaction features
    features_df['gender_year'] = features_df['gender_encoded'] * features_df['year']
    features_df['races_duration_ratio'] = features_df['num_races'] / features_df['season_duration']
    features_df['improvement_per_race'] = features_df['total_improvement'] / features_df['num_races']
    
    # Create polynomial features for key variables
    features_df['starting_percentile_squared'] = features_df['starting_percentile'] ** 2
    features_df['num_races_squared'] = features_df['num_races'] ** 2
    features_df['season_duration_squared'] = features_df['season_duration'] ** 2
    
    # Create performance ratio features
    features_df['best_to_avg_ratio'] = features_df['best_time'] / features_df['avg_time']
    features_df['worst_to_avg_ratio'] = features_df['worst_time'] / features_df['avg_time']
    
    # Create improvement efficiency features
    features_df['improvement_efficiency'] = features_df['total_improvement'] / features_df['time_range']
    features_df['consistency_score'] = 1 / (1 + features_df['cv_time'])
    
    # Create season timing features
    features_df['early_season_performance'] = features_df['first_time']
    features_df['late_season_performance'] = features_df['last_time']
    
    # Create experience features
    features_df['experience_level'] = features_df['num_races'] * features_df['season_duration']
    
    return features_df

def prepare_model_data(features_df):
    """Prepare data for machine learning models."""
    print("Preparing model data...")
    
    # Select features for the model
    feature_columns = [
        'gender_encoded', 'year', 'num_races', 'season_duration', 
        'first_time', 'last_time', 'best_time', 'worst_time', 'avg_time',
        'time_std', 'time_range', 'cv_time', 'race_frequency',
        'progression_improvement', 'starting_percentile', 'gender_year',
        'races_duration_ratio', 'improvement_per_race', 'starting_percentile_squared',
        'num_races_squared', 'season_duration_squared', 'best_to_avg_ratio',
        'worst_to_avg_ratio', 'improvement_efficiency', 'consistency_score',
        'early_season_performance', 'late_season_performance', 'experience_level'
    ]
    
    # Target variable: improvement_rate (seconds per day)
    X = features_df[feature_columns].copy()
    y = features_df['improvement_rate'].copy()
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"Prepared {len(X)} samples with {len(X.columns)} features")
    print(f"Target variable range: {y.min():.3f} to {y.max():.3f} seconds/day")
    
    return X, y

def train_models(X, y):
    """Train multiple machine learning models and compare performance."""
    print("Training machine learning models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline with scaling for models that need it
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('model', model)
            ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        
        results[name] = {
            'model': pipeline,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print()
    
    return results, X_test, y_test

def hyperparameter_tuning(X, y, best_model_name):
    """Perform hyperparameter tuning for the best performing model."""
    print(f"Performing hyperparameter tuning for {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42)
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 0.9, 1.0]
        }
        base_model = GradientBoostingRegressor(random_state=42)
    else:
        print("Hyperparameter tuning not implemented for this model type.")
        return None
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()) if best_model_name not in ['Random Forest', 'Gradient Boosting'] else ('scaler', 'passthrough'),
        ('model', base_model)
    ])
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance for tree-based models."""
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance_df)
        
        # Plot feature importance
        plt.figure(figsize=(14, 10))
        sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature')
        plt.title('Top 15 Most Important Features for Improvement Prediction')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('output/raw_data_feature_importance.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    else:
        print("Feature importance not available for this model type.")
        return None

def plot_predictions(results, output_dir='output'):
    """Plot actual vs predicted values for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot actual vs predicted
        ax.scatter(result['y_test'], result['y_pred'], alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(result['y_test'].min(), result['y_pred'].min())
        max_val = max(result['y_test'].max(), result['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_xlabel('Actual Improvement Rate (seconds/day)')
        ax.set_ylabel('Predicted Improvement Rate (seconds/day)')
        ax.set_title(f'{name}\nR² = {result["r2"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/raw_data_model_predictions.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_insights(features_df, best_model, feature_names):
    """Generate insights about what predicts improvement."""
    print("\n" + "="*60)
    print("IMPROVEMENT PREDICTION INSIGHTS FROM RAW DATA")
    print("="*60)
    
    # Analyze the data
    print("\n1. Overall Improvement Patterns:")
    print(f"   - Average improvement rate: {features_df['improvement_rate'].mean():.3f} seconds/day")
    print(f"   - Standard deviation: {features_df['improvement_rate'].std():.3f} seconds/day")
    print(f"   - Range: {features_df['improvement_rate'].min():.3f} to {features_df['improvement_rate'].max():.3f} seconds/day")
    
    # Gender differences
    gender_analysis = features_df.groupby('gender')['improvement_rate'].agg(['mean', 'std', 'count'])
    print(f"\n2. Gender Differences:")
    for gender in gender_analysis.index:
        print(f"   - {gender}: {gender_analysis.loc[gender, 'mean']:.3f} ± {gender_analysis.loc[gender, 'std']:.3f} seconds/day (n={gender_analysis.loc[gender, 'count']})")
    
    # Year differences
    year_analysis = features_df.groupby('year')['improvement_rate'].agg(['mean', 'std', 'count'])
    print(f"\n3. Year Differences:")
    for year in year_analysis.index:
        print(f"   - {year}: {year_analysis.loc[year, 'mean']:.3f} ± {year_analysis.loc[year, 'std']:.3f} seconds/day (n={year_analysis.loc[year, 'count']})")
    
    # Performance level analysis
    print(f"\n4. Improvement by Starting Performance Level:")
    percentile_bins = [0, 25, 50, 75, 100]
    for i in range(len(percentile_bins)-1):
        mask = (features_df['starting_percentile'] >= percentile_bins[i]) & (features_df['starting_percentile'] < percentile_bins[i+1])
        subset = features_df[mask]
        if len(subset) > 0:
            print(f"   - {percentile_bins[i]}-{percentile_bins[i+1]}th percentile: {subset['improvement_rate'].mean():.3f} seconds/day (n={len(subset)})")
    
    # Race frequency analysis
    print(f"\n5. Improvement by Race Frequency:")
    race_freq_bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    for i in range(len(race_freq_bins)-1):
        mask = (features_df['race_frequency'] >= race_freq_bins[i]) & (features_df['race_frequency'] < race_freq_bins[i+1])
        subset = features_df[mask]
        if len(subset) > 0:
            print(f"   - {race_freq_bins[i]:.1f}-{race_freq_bins[i+1]:.1f} races/day: {subset['improvement_rate'].mean():.3f} seconds/day (n={len(subset)})")
    
    # Feature importance insights
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n6. Most Important Predictors of Improvement:")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"   - {row['feature']}: {row['importance']:.4f}")

def save_model_results(results, features_df, feature_names, output_dir='output'):
    """Save all model results and analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model performance summary
    results_summary = []
    for name, result in results.items():
        results_summary.append({
            'Model': name,
            'R² Score': result['r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'CV R² Mean': result['cv_r2_mean'],
            'CV R² Std': result['cv_r2_std']
        })
    
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(f'{output_dir}/raw_data_model_performance.csv', index=False)
    print(f"Model performance summary saved to {output_dir}/raw_data_model_performance.csv")
    
    # Save feature importance
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        feature_importance_df.to_csv(f'{output_dir}/raw_data_feature_importance.csv', index=False)
        print(f"Feature importance saved to {output_dir}/raw_data_feature_importance.csv")
    
    # Save athlete features for further analysis
    features_df.to_csv(f'{output_dir}/raw_data_athlete_features.csv', index=False)
    print(f"Athlete features saved to {output_dir}/raw_data_athlete_features.csv")

def main():
    """Main function to run the raw data improvement prediction analysis."""
    print("="*60)
    print("RAW DATA IMPROVEMENT PREDICTION MODEL")
    print("="*60)
    
    # Load raw data
    df = load_raw_data()
    
    # Calculate athlete features
    athlete_df = calculate_athlete_features(df)
    
    # Create advanced features
    features_df = create_advanced_features(athlete_df)
    
    # Prepare model data
    X, y = prepare_model_data(features_df)
    
    # Train models
    results, X_test, y_test = train_models(X, y)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"R² Score: {results[best_model_name]['r2']:.4f}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    tuned_model = hyperparameter_tuning(X, y, best_model_name)
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    feature_names = X.columns.tolist()
    feature_importance_df = analyze_feature_importance(best_model, feature_names)
    
    # Plot predictions
    print("\nCreating prediction plots...")
    plot_predictions(results)
    
    # Generate insights
    create_improvement_insights(features_df, best_model, feature_names)
    
    # Save results
    print("\nSaving results...")
    save_model_results(results, features_df, feature_names)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 