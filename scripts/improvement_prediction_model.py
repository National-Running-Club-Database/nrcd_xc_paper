import os
import sys

# Setup paths for imports (works from main directory or scripts directory)
from _setup_paths import setup_paths
setup_paths()

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
import warnings
warnings.filterwarnings('ignore')

from utils import standardize_convert_exclude_nationals_df, convert_exclude_nationals

def load_percentile_data():
    """Load all percentile analysis data and combine into a single dataset."""
    data_dir = 'output/PercentileTimeAnalysis'
    all_data = []
    
    # Load all CSV files
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            
            # Extract metadata from filename
            parts = filename.replace('.csv', '').split('_')
            year = int(parts[2])
            gender = parts[3]
            mode = parts[4] if len(parts) > 4 else 'standardized'
            
            # Add metadata columns
            df['year'] = year
            df['gender'] = gender
            df['mode'] = mode
            df['filename'] = filename
            
            all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def parse_time_range(time_range_str):
    """Parse time range string to extract min and max times in seconds."""
    if '-' not in time_range_str:
        return None, None
    
    min_time_str, max_time_str = time_range_str.split('-')
    
    def time_to_seconds(time_str):
        """Convert MM:SS.S format to seconds."""
        if ':' in time_str:
            minutes, seconds = time_str.split(':')
            return int(minutes) * 60 + float(seconds)
        return float(time_str)
    
    min_seconds = time_to_seconds(min_time_str.strip())
    max_seconds = time_to_seconds(max_time_str.strip())
    
    return min_seconds, max_seconds

def create_features(df):
    """Create features for the machine learning model."""
    features_df = df.copy()
    
    # Parse time ranges
    features_df[['min_time_seconds', 'max_time_seconds']] = features_df['starting_time_range'].apply(
        lambda x: pd.Series(parse_time_range(x))
    )
    
    # Calculate additional features
    features_df['time_range_seconds'] = features_df['max_time_seconds'] - features_df['min_time_seconds']
    features_df['avg_time_seconds'] = (features_df['min_time_seconds'] + features_df['max_time_seconds']) / 2
    
    # Extract percentile information
    features_df['percentile_lower'] = features_df['percentile_range'].str.extract(r'(\d+)').astype(int)
    features_df['percentile_upper'] = features_df['percentile_range'].str.extract(r'-(\d+)').astype(int)
    features_df['percentile_mid'] = (features_df['percentile_lower'] + features_df['percentile_upper']) / 2
    
    # Create categorical encodings
    le_gender = LabelEncoder()
    le_mode = LabelEncoder()
    
    features_df['gender_encoded'] = le_gender.fit_transform(features_df['gender'])
    features_df['mode_encoded'] = le_mode.fit_transform(features_df['mode'])
    
    # Create interaction features
    features_df['gender_year'] = features_df['gender_encoded'] * features_df['year']
    features_df['percentile_gender'] = features_df['percentile_mid'] * features_df['gender_encoded']
    
    # Create polynomial features for key numerical variables
    features_df['avg_time_squared'] = features_df['avg_time_seconds'] ** 2
    features_df['percentile_squared'] = features_df['percentile_mid'] ** 2
    
    return features_df

def prepare_model_data(features_df):
    """Prepare data for machine learning models."""
    # Select features for the model
    feature_columns = [
        'year', 'gender_encoded', 'mode_encoded', 'percentile_lower', 'percentile_upper', 
        'percentile_mid', 'min_time_seconds', 'max_time_seconds', 'time_range_seconds',
        'avg_time_seconds', 'slope_cv', 'num_athletes', 'gender_year', 'percentile_gender',
        'avg_time_squared', 'percentile_squared'
    ]
    
    # Target variable: median_slope (improvement rate)
    X = features_df[feature_columns].copy()
    y = features_df['median_slope'].copy()
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    return X, y

def train_models(X, y):
    """Train multiple machine learning models and compare performance."""
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
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
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
        
        ax.set_xlabel('Actual Improvement Rate')
        ax.set_ylabel('Predicted Improvement Rate')
        ax.set_title(f'{name}\nR² = {result["r2"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_insights(features_df, best_model, feature_names):
    """Generate insights about what predicts improvement."""
    print("\n" + "="*50)
    print("IMPROVEMENT PREDICTION INSIGHTS")
    print("="*50)
    
    # Analyze the data
    print("\n1. Overall Improvement Patterns:")
    print(f"   - Average improvement rate: {features_df['median_slope'].mean():.3f} seconds/day")
    print(f"   - Standard deviation: {features_df['median_slope'].std():.3f} seconds/day")
    print(f"   - Range: {features_df['median_slope'].min():.3f} to {features_df['median_slope'].max():.3f} seconds/day")
    
    # Gender differences
    gender_analysis = features_df.groupby('gender')['median_slope'].agg(['mean', 'std', 'count'])
    print(f"\n2. Gender Differences:")
    for gender in gender_analysis.index:
        print(f"   - {gender}: {gender_analysis.loc[gender, 'mean']:.3f} ± {gender_analysis.loc[gender, 'std']:.3f} seconds/day (n={gender_analysis.loc[gender, 'count']})")
    
    # Year differences
    year_analysis = features_df.groupby('year')['median_slope'].agg(['mean', 'std', 'count'])
    print(f"\n3. Year Differences:")
    for year in year_analysis.index:
        print(f"   - {year}: {year_analysis.loc[year, 'mean']:.3f} ± {year_analysis.loc[year, 'std']:.3f} seconds/day (n={year_analysis.loc[year, 'count']})")
    
    # Percentile analysis
    percentile_analysis = features_df.groupby('percentile_mid')['median_slope'].mean()
    print(f"\n4. Improvement by Starting Performance Level:")
    for percentile, improvement in percentile_analysis.items():
        print(f"   - {percentile:.0f}th percentile: {improvement:.3f} seconds/day")
    
    # Feature importance insights
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n5. Most Important Predictors of Improvement:")
        for i, row in feature_importance_df.head(5).iterrows():
            print(f"   - {row['feature']}: {row['importance']:.4f}")

def main():
    """Main function to run the improvement prediction analysis."""
    print("Loading percentile analysis data...")
    df = load_percentile_data()
    print(f"Loaded {len(df)} records from percentile analysis")
    
    print("\nCreating features...")
    features_df = create_features(df)
    print(f"Created features for {len(features_df)} records")
    
    print("\nPreparing model data...")
    X, y = prepare_model_data(features_df)
    print(f"Prepared {len(X)} samples with {len(X.columns)} features")
    
    print("\nTraining machine learning models...")
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
    os.makedirs('output', exist_ok=True)
    
    # Save model results summary
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
    results_df.to_csv('output/model_performance_summary.csv', index=False)
    print("Model performance summary saved to output/model_performance_summary.csv")
    
    if feature_importance_df is not None:
        feature_importance_df.to_csv('output/feature_importance.csv', index=False)
        print("Feature importance saved to output/feature_importance.csv")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 