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

def load_and_prepare_data():
    """Load raw data and prepare features with temporal split."""
    print("Loading raw data...")
    
    # Load standardized data
    df = standardize_convert_exclude_nationals_df()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df = df.dropna(subset=['standardized_to_target', 'start_date', 'gender', 'athlete_id'])
    
    print(f"Loaded {len(df)} raw athlete records")
    return df

def calculate_athlete_features(df):
    """Calculate features for each athlete based on their race history."""
    print("Calculating athlete features...")
    
    athlete_features = []
    athletes = df['athlete_id'].unique()
    print(f"Processing {len(athletes)} athletes...")
    
    for i, athlete_id in enumerate(athletes):
        if i % 1000 == 0:
            print(f"  Processing athlete {i+1}/{len(athletes)}")
        
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
        
        total_improvement = last_time - first_time
        days_diff = (last_date - first_date).days
        
        if days_diff <= 0:
            continue
        
        improvement_rate = total_improvement / days_diff
        
        # Calculate features
        num_races = len(athlete_races)
        season_duration = days_diff
        times = athlete_races['standardized_to_target'].values
        best_time = np.min(times)
        worst_time = np.max(times)
        avg_time = np.mean(times)
        time_std = np.std(times)
        time_range = worst_time - best_time
        cv_time = time_std / avg_time if avg_time > 0 else 0
        
        # Calculate slope
        if len(times) >= 3:
            X = np.arange(len(times)).reshape(-1, 1)
            y = times
            slope_model = LinearRegression()
            slope_model.fit(X, y)
            slope = slope_model.coef_[0]
        else:
            slope = improvement_rate
        
        race_frequency = num_races / season_duration if season_duration > 0 else 0
        
        # Calculate progression
        if len(times) >= 3:
            mid_point = len(times) // 2
            first_half_avg = np.mean(times[:mid_point])
            second_half_avg = np.mean(times[mid_point:])
            progression_improvement = first_half_avg - second_half_avg
        else:
            progression_improvement = total_improvement
        
        gender = athlete_races.iloc[0]['gender']
        year = athlete_races.iloc[0]['start_date'].year
        
        # Calculate percentile
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

def create_features(athlete_df):
    """Create advanced features."""
    print("Creating advanced features...")
    
    features_df = athlete_df.copy()
    
    # Create categorical encodings
    le_gender = LabelEncoder()
    features_df['gender_encoded'] = le_gender.fit_transform(features_df['gender'])
    
    # Create interaction features
    features_df['gender_year'] = features_df['gender_encoded'] * features_df['year']
    features_df['races_duration_ratio'] = features_df['num_races'] / features_df['season_duration']
    features_df['improvement_per_race'] = features_df['total_improvement'] / features_df['num_races']
    
    # Create polynomial features
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

def temporal_split_validation(features_df):
    """Perform temporal validation: train on 2023, test on 2024."""
    print("Performing temporal validation...")
    
    # Select features
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
    
    X = features_df[feature_columns].copy()
    y = features_df['improvement_rate'].copy()
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    features_df = features_df[mask]
    
    # Temporal split: train on 2023, test on 2024
    train_mask = features_df['year'] == 2023
    test_mask = features_df['year'] == 2024
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Training set (2023): {len(X_train)} samples")
    print(f"Test set (2024): {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train models and evaluate with temporal validation."""
    print("Training models with temporal validation...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('model', model)
            ])
        
        # Train on 2023 data
        pipeline.fit(X_train, y_train)
        
        # Predict on 2024 data
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation on training data only
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
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
        
        print(f"  Test R² Score (2024): {r2:.4f}")
        print(f"  Test RMSE: {rmse:.4f}")
        print(f"  Test MAE: {mae:.4f}")
        print(f"  CV R² (2023): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print()
    
    return results

def analyze_overfitting(results):
    """Analyze potential overfitting by comparing CV and test scores."""
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    
    print("\nModel Performance Comparison:")
    print("Model                    | CV R² (2023) | Test R² (2024) | Overfitting Score")
    print("-" * 75)
    
    for name, result in results.items():
        cv_r2 = result['cv_r2_mean']
        test_r2 = result['r2']
        overfitting_score = cv_r2 - test_r2
        
        print(f"{name:<25} | {cv_r2:>11.4f} | {test_r2:>13.4f} | {overfitting_score:>15.4f}")
        
        if overfitting_score > 0.1:
            print(f"  ⚠️  {name} shows potential overfitting (difference > 0.1)")
        elif overfitting_score < -0.1:
            print(f"  ✅ {name} generalizes well to new data")
        else:
            print(f"  ➖ {name} shows moderate generalization")

def get_model_formula(results, feature_names):
    """Extract the formula for the best model."""
    print("\n" + "="*60)
    print("MODEL FORMULA")
    print("="*60)
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        # For linear models, we can extract coefficients
        coefficients = best_model.named_steps['model'].coef_
        intercept = best_model.named_steps['model'].intercept_
        
        print(f"\n{best_model_name} Formula:")
        print(f"Improvement Rate = {intercept:.6f}")
        
        # Sort features by absolute coefficient value
        feature_coefs = list(zip(feature_names, coefficients))
        feature_coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, coef in feature_coefs:
            if abs(coef) > 0.001:  # Only show significant coefficients
                sign = "+" if coef >= 0 else ""
                print(f"  {sign}{coef:.6f} × {feature}")
        
        print(f"\nR² Score: {results[best_model_name]['r2']:.4f}")
        
    elif best_model_name in ['Random Forest', 'Gradient Boosting']:
        # For tree-based models, show feature importance
        importances = best_model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n{best_model_name} - Top 10 Most Important Features:")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nR² Score: {results[best_model_name]['r2']:.4f}")
        print("Note: Tree-based models don't have simple linear formulas.")
        print("They use decision trees to make predictions based on feature splits.")
    
    else:
        print(f"\n{best_model_name} - Complex model without simple formula")
        print(f"R² Score: {results[best_model_name]['r2']:.4f}")

def main():
    """Main function for temporal validation."""
    print("="*60)
    print("TEMPORAL VALIDATION MODEL")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    athlete_df = calculate_athlete_features(df)
    features_df = create_features(athlete_df)
    
    # Perform temporal split
    X_train, X_test, y_train, y_test = temporal_split_validation(features_df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Analyze overfitting
    analyze_overfitting(results)
    
    # Get model formula
    feature_names = X_train.columns.tolist()
    get_model_formula(results, feature_names)
    
    # Save results
    print("\nSaving results...")
    os.makedirs('output', exist_ok=True)
    
    results_summary = []
    for name, result in results.items():
        results_summary.append({
            'Model': name,
            'Test_R2_Score': result['r2'],
            'Test_RMSE': result['rmse'],
            'Test_MAE': result['mae'],
            'CV_R2_Mean': result['cv_r2_mean'],
            'CV_R2_Std': result['cv_r2_std'],
            'Overfitting_Score': result['cv_r2_mean'] - result['r2']
        })
    
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('output/temporal_validation_results.csv', index=False)
    print("Temporal validation results saved to output/temporal_validation_results.csv")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 