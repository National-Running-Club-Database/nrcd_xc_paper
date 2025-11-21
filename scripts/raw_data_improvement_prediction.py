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
from scipy import stats
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not available. Install with 'pip install tqdm' for progress bars.")

import os
from utils import standardize_convert_exclude_nationals_df, convert_exclude_nationals, parse_time

def load_raw_data(mode='standardized'):
    """
    Load raw athlete data for improvement prediction.
    
    Parameters:
    -----------
    mode : str
        'standardized' - Full standardization (weather, terrain, distance)
        'converted' - Distance conversion only (no weather/terrain)
        'original' - Raw times (no adjustments)
    """
    print(f"Loading raw data ({mode})...")
    
    if mode == 'standardized':
        # Full standardization: weather, terrain, distance adjustments
        df = standardize_convert_exclude_nationals_df()
        time_col = 'standardized_to_target'
    elif mode == 'converted':
        # Distance conversion only: no weather/terrain adjustments
        df = convert_exclude_nationals()
        time_col = 'standardized_to_target'
    elif mode == 'original' or mode == 'raw':
        # Raw/Original: No adjustments at all - just parse the raw times
        # Note: This doesn't convert distances, so times from different distances aren't directly comparable
        # This demonstrates why standardization is necessary
        import os
        results_df = pd.read_csv(os.path.join('data/jss_data', 'result.csv'))
        meet_df = pd.read_csv(os.path.join('data/jss_data', 'meet.csv'))
        athlete_df = pd.read_csv(os.path.join('data/jss_data', 'athlete.csv'))
        running_event_df = pd.read_csv(os.path.join('data/jss_data', 'running_event.csv'))
        
        # Exclude nationals
        non_nationals_meets = meet_df[~meet_df['nationals'].astype(bool)]['meet_id']
        df = results_df[results_df['meet_id'].isin(non_nationals_meets)].copy()
        
        # Merge with other data
        df = df.merge(athlete_df[['athlete_id', 'gender']], on='athlete_id', how='left')
        df = df.merge(running_event_df[['running_event_id', 'event_name']], on='running_event_id', how='left')
        df = df.merge(meet_df[['meet_id', 'start_date']], on='meet_id', how='left')
        
        # Parse raw times (no distance conversion, no weather/terrain adjustments)
        df['standardized_to_target'] = df['result_time'].apply(parse_time)
        time_col = 'standardized_to_target'
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Ensure dates are datetime
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter for valid data
    df = df.dropna(subset=[time_col, 'start_date', 'gender', 'athlete_id'])
    
    print(f"Loaded {len(df)} raw athlete records ({mode})")
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
        
        # Calculate average days between races (recovery time)
        if num_races > 1:
            dates = athlete_races['start_date'].values  # Already sorted by start_date
            days_between = np.diff(dates)
            # Convert timedelta to days
            if len(days_between) > 0:
                # Handle pandas Timedelta objects
                avg_days_between_races = np.mean([d.days if hasattr(d, 'days') else float(d) / (24*3600*1e9) for d in days_between])
            else:
                avg_days_between_races = 0
        else:
            avg_days_between_races = 0
        
        # Calculate race-to-race improvement consistency
        if len(times) >= 2:
            race_to_race_improvements = np.diff(times)  # Negative = improving
            race_to_race_improvement_std = np.std(race_to_race_improvements) if len(race_to_race_improvements) > 0 else 0
            # Count "bad" races (worse than previous race)
            bad_race_count = np.sum(race_to_race_improvements > 0)  # Positive = slower = bad
        else:
            race_to_race_improvement_std = 0
            bad_race_count = 0
        
        # Calculate when best race occurred (timing of peak performance)
        best_race_idx = np.argmin(times)
        if best_race_idx == 0:
            best_race_timing = 0  # Best race was first race
        elif best_race_idx == len(times) - 1:
            best_race_timing = season_duration  # Best race was last race
        else:
            # Days from first race to best race
            best_race_date = athlete_races.iloc[best_race_idx]['start_date']
            best_race_timing = (best_race_date - first_date).days
        
        # Note: progression_improvement removed - not needed as a feature
        
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
            'starting_percentile': starting_percentile,
            # New features
            'avg_days_between_races': avg_days_between_races,
            'race_to_race_improvement_std': race_to_race_improvement_std,
            'best_race_timing': best_race_timing,
            'bad_race_count': bad_race_count
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
    # Note: races_duration_ratio removed - it's identical to race_frequency (both = num_races / season_duration)
    # Note: improvement_per_race removed - it's circular with target variable (improvement_rate)
    # Both use total_improvement, creating a data leakage issue
    
    # Create polynomial features for key variables
    # Note: Squared terms capture non-linear relationships (e.g., optimal season length)
    # We include both linear and squared terms to model potential U-shaped or inverted-U relationships
    features_df['starting_percentile_squared'] = features_df['starting_percentile'] ** 2
    features_df['num_races_squared'] = features_df['num_races'] ** 2
    features_df['season_duration_squared'] = features_df['season_duration'] ** 2
    
    # Create performance ratio features
    features_df['best_to_avg_ratio'] = features_df['best_time'] / features_df['avg_time']
    features_df['worst_to_avg_ratio'] = features_df['worst_time'] / features_df['avg_time']
    
    # Create improvement efficiency features
    # improvement_to_variability_ratio: total improvement relative to performance variability
    # Higher values = more improvement relative to variability (more consistent improvement)
    features_df['improvement_to_variability_ratio'] = features_df['total_improvement'] / features_df['time_range']
    features_df['consistency_score'] = 1 / (1 + features_df['cv_time'])
    
    # Note: early_season_performance and late_season_performance removed - duplicates of first_time and last_time
    
    # Create experience features
    features_df['experience_level'] = features_df['num_races'] * features_df['season_duration']
    
    # Create peak timing feature (normalized to season duration)
    features_df['best_race_timing_ratio'] = features_df['best_race_timing'] / features_df['season_duration']
    # Replace inf/NaN with 0 (for edge cases)
    features_df['best_race_timing_ratio'] = features_df['best_race_timing_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return features_df

def prepare_model_data(features_df):
    """Prepare data for machine learning models."""
    print("Preparing model data...")
    
    # Select features for the model
    # Note: improvement_per_race removed - it's circular with target (both use total_improvement)
    # Note: progression_improvement removed - not needed as a feature
    # Note: Removed duplicates:
    #   - early_season_performance (duplicate of first_time)
    #   - late_season_performance (duplicate of last_time)
    #   - races_duration_ratio (duplicate of race_frequency)
    # num_races is the correct feature to use (number of races in season)
    feature_columns = [
        'gender_encoded', 'year', 'num_races', 'season_duration', 
        'first_time', 'last_time', 'best_time', 'worst_time', 'avg_time',
        'time_std', 'time_range', 'cv_time', 'race_frequency',
        'starting_percentile', 'gender_year',
        'starting_percentile_squared',
        'num_races_squared', 'season_duration_squared', 'best_to_avg_ratio',
        'worst_to_avg_ratio', 'improvement_to_variability_ratio', 'consistency_score',
        'experience_level',
        # New features
        'slope',  # Improvement trajectory pattern (was calculated but not used)
        'avg_days_between_races',  # Recovery time indicator
        'race_to_race_improvement_std',  # Consistency of improvement
        'best_race_timing',  # When peak performance occurred
        'best_race_timing_ratio',  # Peak timing normalized to season duration
        'bad_race_count'  # Number of races worse than previous
    ]
    
    # Target variable: improvement_rate (seconds per day)
    X = features_df[feature_columns].copy()
    y = features_df['improvement_rate'].copy()
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    features_df_filtered = features_df[mask].copy()
    
    print(f"Prepared {len(X)} samples with {len(X.columns)} features")
    print(f"Target variable range: {y.min():.3f} to {y.max():.3f} seconds/day")
    
    return X, y, features_df_filtered

def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for a metric.
    
    This function uses bootstrap resampling to estimate confidence intervals
    for model performance metrics, providing a more robust assessment of
    model reliability than point estimates alone.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    metric_func : callable
        Function to compute the metric (e.g., r2_score)
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
    --------
    mean_score : float
        Mean metric value across bootstrap samples
    lower : float
        Lower bound of confidence interval
    upper : float
        Upper bound of confidence interval
    """
    n = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(bootstrap_scores)
    
    return mean_score, lower, upper

def train_models(X, y, features_df, use_temporal_split=True):
    """
    Train multiple machine learning models and compare performance.
    
    Parameters:
    -----------
    use_temporal_split : bool
        If True, use temporal split (train on 2023, test on 2024).
        If False, use random split (80/20).
    """
    print("Training machine learning models...")
    
    if use_temporal_split:
        # Temporal split: train on 2023, test on 2024
        print("Using temporal validation: training on 2023, testing on 2024")
        train_mask = features_df['year'] == 2023
        test_mask = features_df['year'] == 2024
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        test_indices = X_test.index
        test_metadata = features_df.loc[test_indices, ['year', 'gender']].copy()
        
        print(f"Training set (2023): {len(X_train)} samples")
        print(f"Test set (2024): {len(X_test)} samples")
    else:
        # Random split
        print("Using random split: 80% train, 20% test")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get indices for test set to track year and gender
        test_indices = X_test.index
        test_metadata = features_df.loc[test_indices, ['year', 'gender']].copy()
    
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
        
        # Bootstrap confidence intervals for R²
        r2_mean, r2_lower, r2_upper = bootstrap_confidence_interval(
            y_test, y_pred, r2_score, n_bootstrap=1000
        )
        
        # Cross-validation score (on training data only for temporal split)
        if use_temporal_split:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        
        # Calculate residuals for diagnostics
        residuals = y_test - y_pred
        
        results[name] = {
            'model': pipeline,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'r2_ci_lower': r2_lower,
            'r2_ci_upper': r2_upper,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test,
            'residuals': residuals
        }
        
        print(f"  Test R² Score: {r2:.4f} (95% CI: [{r2_lower:.4f}, {r2_upper:.4f}])")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  CV R² (train): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print()
    
    # Statistical comparison of models
    print("\n" + "="*60)
    print("STATISTICAL MODEL COMPARISON")
    print("="*60)
    
    # Compare models using paired t-test on cross-validation scores
    model_names = list(results.keys())
    if len(model_names) > 1:
        print("\nPaired t-tests comparing CV R² scores:")
        print("-" * 60)
        
        # Get CV scores for all models (need to recompute for comparison)
        cv_scores_dict = {}
        for name in model_names:
            if use_temporal_split:
                cv_scores = cross_val_score(
                    results[name]['model'], X_train, y_train, cv=5, scoring='r2'
                )
            else:
                cv_scores = cross_val_score(
                    results[name]['model'], X, y, cv=5, scoring='r2'
                )
            cv_scores_dict[name] = cv_scores
        
        # Compare each pair
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                t_stat, p_value = ttest_rel(cv_scores_dict[model1], cv_scores_dict[model2])
                mean_diff = cv_scores_dict[model1].mean() - cv_scores_dict[model2].mean()
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{model1} vs {model2}:")
                print(f"  Mean difference: {mean_diff:.4f}")
                print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f} {significance}")
                print()
    
    return results, X_test, y_test, test_metadata

def hyperparameter_tuning(X_train, y_train, best_model_name):
    """
    Perform hyperparameter tuning for the best performing model.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    best_model_name : str
        Name of the best performing model
    """
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
    if best_model_name not in ['Random Forest', 'Gradient Boosting']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
    else:
        pipeline = Pipeline([
            ('model', base_model)
        ])
    
    # Calculate total number of fits for progress bar
    from itertools import product
    param_combinations = list(product(*param_grid.values()))
    total_fits = len(param_combinations) * 5  # 5 CV folds
    
    print(f"Testing {len(param_combinations)} parameter combinations with 5-fold CV ({total_fits} total fits)...")
    
    # Grid search - suppress verbose output to avoid clutter
    # Use verbose=0 to prevent sklearn from printing each CV fold
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
    )
    
    if TQDM_AVAILABLE:
        # Show a simple progress message with tqdm
        with tqdm(total=total_fits, desc="Hyperparameter tuning", unit="fit",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            # Fit the model
            grid_search.fit(X_train, y_train)
            # Update to completion
            pbar.n = total_fits
            pbar.refresh()
    else:
        # No progress bar, just fit
        grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def analyze_feature_importance(model, feature_names, output_dir='output'):
    """Analyze feature importance for tree-based models."""
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance_df)
        
        # Plot feature importance - show top 15
        plt.figure(figsize=(14, 10))
        top_15 = feature_importance_df.head(15)
        sns.barplot(data=top_15, x='importance', y='feature')
        plt.title('Top 15 Most Important Features for Improvement Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/raw_data_feature_importance.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance_df
    else:
        print("Feature importance not available for this model type.")
        return None

def analyze_gender_specific_feature_importance(X, y, features_df, output_dir='output'):
    """
    Analyze feature importance separately for men and women.
    
    This function trains separate models for each gender to determine if
    different factors are important for predicting improvement in men vs women.
    """
    print("\n" + "="*60)
    print("GENDER-SPECIFIC FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use temporal split: train on 2023, test on 2024
    train_mask = features_df['year'] == 2023
    test_mask = features_df['year'] == 2024
    
    X_train_all = X[train_mask]
    y_train_all = y[train_mask]
    X_test_all = X[test_mask]
    y_test_all = y[test_mask]
    features_train = features_df[train_mask]
    features_test = features_df[test_mask]
    
    gender_importance_comparison = []
    
    for gender in ['M', 'F']:
        gender_label = 'Men' if gender == 'M' else 'Women'
        print(f"\nAnalyzing {gender_label}...")
        
        # Filter training and test data by gender
        train_gender_mask = features_train['gender'] == gender
        test_gender_mask = features_test['gender'] == gender
        
        if train_gender_mask.sum() < 50:  # Need minimum samples
            print(f"  Insufficient data for {gender_label} (n={train_gender_mask.sum()})")
            continue
        
        X_train_gender = X_train_all[train_gender_mask.values]
        y_train_gender = y_train_all[train_gender_mask.values]
        X_test_gender = X_test_all[test_gender_mask.values]
        y_test_gender = y_test_all[test_gender_mask.values]
        
        print(f"  Training samples: {len(X_train_gender)}, Test samples: {len(X_test_gender)}")
        
        # Train Gradient Boosting model (best overall model)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_gender, y_train_gender)
        
        # Evaluate
        y_pred_gender = model.predict(X_test_gender)
        r2_gender = r2_score(y_test_gender, y_pred_gender)
        mae_gender = mean_absolute_error(y_test_gender, y_pred_gender)
        
        print(f"  Test R²: {r2_gender:.4f}, MAE: {mae_gender:.4f}")
        
        # Get feature importance
        feature_names_list = X.columns.tolist()
        importances = model.feature_importances_
        
        # Store importance for comparison
        for i, (feature, importance) in enumerate(zip(feature_names_list, importances)):
            gender_importance_comparison.append({
                'Gender': gender_label,
                'Feature': feature,
                'Importance': importance,
                'Rank': None  # Will fill later
            })
    
    # Create comparison DataFrame
    if len(gender_importance_comparison) > 0:
        comparison_df = pd.DataFrame(gender_importance_comparison)
        
        # Calculate ranks within each gender
        for gender in comparison_df['Gender'].unique():
            mask = comparison_df['Gender'] == gender
            comparison_df.loc[mask, 'Rank'] = comparison_df.loc[mask, 'Importance'].rank(ascending=False, method='min')
        
        # Pivot for easier comparison
        pivot_df = comparison_df.pivot(index='Feature', columns='Gender', values='Importance').fillna(0)
        rank_pivot_df = comparison_df.pivot(index='Feature', columns='Gender', values='Rank').fillna(999)
        
        # Calculate difference in importance
        if 'Men' in pivot_df.columns and 'Women' in pivot_df.columns:
            pivot_df['Difference'] = pivot_df['Women'] - pivot_df['Men']
            pivot_df['Abs_Difference'] = pivot_df['Difference'].abs()
            pivot_df = pivot_df.sort_values('Abs_Difference', ascending=False)
        
        # Save comparison
        comparison_df.to_csv(f'{output_dir}/raw_data_gender_feature_importance_comparison.csv', index=False)
        pivot_df.to_csv(f'{output_dir}/raw_data_gender_feature_importance_pivot.csv')
        
        print(f"\nGender-specific feature importance saved to:")
        print(f"  - {output_dir}/raw_data_gender_feature_importance_comparison.csv")
        print(f"  - {output_dir}/raw_data_gender_feature_importance_pivot.csv")
        
        # Print top differences
        if 'Men' in pivot_df.columns and 'Women' in pivot_df.columns:
            print("\nTop 10 Features with Largest Gender Differences:")
            print("-" * 80)
            print(f"{'Feature':<30} {'Men':<12} {'Women':<12} {'Difference':<12}")
            print("-" * 80)
            for feature in pivot_df.head(10).index:
                men_imp = pivot_df.loc[feature, 'Men']
                women_imp = pivot_df.loc[feature, 'Women']
                diff = pivot_df.loc[feature, 'Difference']
                print(f"{feature:<30} {men_imp:<12.6f} {women_imp:<12.6f} {diff:<12.6f}")
        
        # Test statistical significance of feature importance differences using bootstrap
        print("\nTesting statistical significance of feature importance differences...")
        significance_results = test_feature_importance_significance(
            X_train_all, y_train_all, features_train, feature_names_list
        )
        
        # Merge significance results with pivot_df and apply multiple comparisons correction
        bonferroni_alpha = None  # Initialize for scope
        if significance_results is not None and 'Men' in pivot_df.columns and 'Women' in pivot_df.columns:
            pivot_df['P_Value'] = pivot_df.index.map(
                lambda f: significance_results.get(f, {}).get('p_value', 1.0)
            )
            
            # Apply multiple comparisons corrections
            n_tests = len(pivot_df)
            alpha = 0.05
            
            # Bonferroni correction: divide alpha by number of tests
            bonferroni_alpha = alpha / n_tests
            pivot_df['P_Value_Bonferroni'] = pivot_df['P_Value'] * n_tests  # Adjusted p-values
            pivot_df['Significant_Bonferroni'] = pivot_df['P_Value'] < bonferroni_alpha
            
            # Benjamini-Hochberg FDR correction (manual implementation)
            p_values = pivot_df['P_Value'].values
            # Manual Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            p_values_fdr = np.zeros_like(p_values)
            for i, idx in enumerate(sorted_indices):
                p_values_fdr[idx] = p_values[idx] * n_tests / (i + 1)
            # Ensure monotonicity (p-values should be non-decreasing)
            for i in range(len(p_values_fdr) - 2, -1, -1):
                if p_values_fdr[i] > p_values_fdr[i + 1]:
                    p_values_fdr[i] = p_values_fdr[i + 1]
            # Cap at 1.0
            p_values_fdr = np.minimum(p_values_fdr, 1.0)
            
            pivot_df['P_Value_FDR'] = p_values_fdr
            pivot_df['Significant_FDR'] = pivot_df['P_Value_FDR'] < alpha
            
            # Use FDR for main significance flag (less conservative than Bonferroni)
            pivot_df['Significant'] = pivot_df['Significant_FDR']
            
            # Significance level based on uncorrected p-values (for display)
            pivot_df['Significance_Level'] = pivot_df['P_Value'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            )
            
            # Add corrected significance markers
            pivot_df['Significance_Bonferroni'] = pivot_df['P_Value'].apply(
                lambda p: '***' if p < bonferroni_alpha else ''
            )
            pivot_df['Significance_FDR'] = pivot_df['P_Value_FDR'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            )
            
            print(f"\nMultiple Comparisons Correction Applied:")
            print(f"  Number of tests: {n_tests}")
            print(f"  Bonferroni alpha: {bonferroni_alpha:.6f} (0.05 / {n_tests})")
            print(f"  FDR (Benjamini-Hochberg) alpha: {alpha}")
            print(f"  Features significant (uncorrected p<0.05): {sum(pivot_df['P_Value'] < 0.05)}")
            print(f"  Features significant (Bonferroni): {sum(pivot_df['Significant_Bonferroni'])}")
            print(f"  Features significant (FDR): {sum(pivot_df['Significant_FDR'])}")
        
        # Create visualization
        if 'Men' in pivot_df.columns and 'Women' in pivot_df.columns:
            # Get top 15 features by absolute difference
            top_features = pivot_df.head(15).index
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot 1: Side-by-side comparison with significance markers
            x = np.arange(len(top_features))
            width = 0.35
            
            men_vals = [pivot_df.loc[f, 'Men'] for f in top_features]
            women_vals = [pivot_df.loc[f, 'Women'] for f in top_features]
            
            # Color bars based on significance
            men_colors = []
            women_colors = []
            for f in top_features:
                is_sig = pivot_df.loc[f, 'Significant'] if 'Significant' in pivot_df.columns else False
                if is_sig:
                    men_colors.append('#2E86AB')  # Blue for significant
                    women_colors.append('#A23B72')  # Purple for significant
                else:
                    men_colors.append('#6C757D')  # Gray for non-significant
                    women_colors.append('#6C757D')
            
            bars1 = ax1.barh(x - width/2, men_vals, width, label='Men', alpha=0.8, color=men_colors)
            bars2 = ax1.barh(x + width/2, women_vals, width, label='Women', alpha=0.8, color=women_colors)
            
            # Set x-axis limits with padding for significance markers
            max_importance = max(max(men_vals), max(women_vals))
            ax1.set_xlim(0, max_importance * 1.15)  # Add 15% padding for markers
            
            # Add significance markers
            if 'Significance_Level' in pivot_df.columns:
                for i, f in enumerate(top_features):
                    sig_level = pivot_df.loc[f, 'Significance_Level']
                    if sig_level:
                        # Position marker at the right edge of the bars with small offset
                        max_val = max(men_vals[i], women_vals[i])
                        # Use axis coordinates to position relative to plot width
                        x_pos = max_val + (max_importance * 0.02)  # Small fixed offset
                        ax1.text(x_pos, i, sig_level, 
                                fontsize=12, fontweight='bold', va='center', ha='left',
                                color='red' if pivot_df.loc[f, 'Difference'] > 0 else 'blue')
            
            ax1.set_yticks(x)
            ax1.set_yticklabels(top_features)
            ax1.set_xlabel('Feature Importance', fontsize=12)
            # Count significant features
            n_sig_fdr = sum(pivot_df.loc[top_features, 'Significant_FDR']) if 'Significant_FDR' in pivot_df.columns else 0
            n_sig_bonf = sum(pivot_df.loc[top_features, 'Significant_Bonferroni']) if 'Significant_Bonferroni' in pivot_df.columns else 0
            
            ax1.set_title(f'Feature Importance Comparison: Men vs Women\n'
                         f'(Colored = FDR Significant, *p<0.05, **p<0.01, ***p<0.001)\n'
                         f'FDR Significant: {n_sig_fdr}, Bonferroni Significant: {n_sig_bonf}', 
                         fontsize=12, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Plot 2: Difference plot with significance
            differences = [pivot_df.loc[f, 'Difference'] for f in top_features]
            sig_levels = [pivot_df.loc[f, 'Significance_Level'] if 'Significance_Level' in pivot_df.columns else '' 
                         for f in top_features]
            
            # Color based on significance
            diff_colors = []
            for i, f in enumerate(top_features):
                is_sig = pivot_df.loc[f, 'Significant'] if 'Significant' in pivot_df.columns else False
                if is_sig:
                    diff_colors.append('#E63946' if differences[i] > 0 else '#06A77D')  # Red/Green for significant
                else:
                    diff_colors.append('#6C757D')  # Gray for non-significant
            
            bars3 = ax2.barh(x, differences, color=diff_colors, alpha=0.7)
            
            # Set x-axis limits with padding for significance markers
            max_diff = max(abs(min(differences)), abs(max(differences)))
            ax2.set_xlim(min(differences) * 1.15, max(differences) * 1.15)  # Add 15% padding
            
            ax2.set_yticks(x)
            ax2.set_yticklabels(top_features)
            ax2.set_xlabel('Difference (Women - Men)', fontsize=12)
            ax2.set_title(f'Feature Importance Differences\n'
                         f'(Red/Green = FDR Significant, Gray = Not Significant)\n'
                         f'FDR: {n_sig_fdr} significant, Bonferroni: {n_sig_bonf} significant', 
                         fontsize=12, fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add significance markers to difference plot
            for i, (diff, sig) in enumerate(zip(differences, sig_levels)):
                if sig:
                    # Position marker with small fixed offset from bar end
                    if diff > 0:
                        x_pos = diff + (max_diff * 0.02)  # Small offset to the right
                        ha = 'left'
                    else:
                        x_pos = diff - (max_diff * 0.02)  # Small offset to the left
                        ha = 'right'
                    ax2.text(x_pos, i, sig, fontsize=12, fontweight='bold', 
                            va='center', ha=ha,
                            color='red' if diff > 0 else 'blue')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/raw_data_gender_feature_importance_comparison.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nVisualization saved to {output_dir}/raw_data_gender_feature_importance_comparison.pdf")
            
            # Print significant features (with corrections)
            if 'Significant' in pivot_df.columns:
                # Uncorrected
                sig_uncorrected = pivot_df[pivot_df['P_Value'] < 0.05].sort_values('Abs_Difference', ascending=False)
                print(f"\nStatistically Significant Feature Differences (Uncorrected, p < 0.05):")
                print("-" * 90)
                print(f"{'Feature':<30} {'Men':<12} {'Women':<12} {'Difference':<12} {'P-Value':<10} {'FDR':<10}")
                print("-" * 90)
                for f in sig_uncorrected.index:
                    print(f"{f:<30} {pivot_df.loc[f, 'Men']:<12.6f} {pivot_df.loc[f, 'Women']:<12.6f} "
                          f"{pivot_df.loc[f, 'Difference']:<12.6f} {pivot_df.loc[f, 'P_Value']:<10.4f} "
                          f"{pivot_df.loc[f, 'P_Value_FDR']:<10.4f}")
                
                # FDR corrected
                sig_fdr = pivot_df[pivot_df['Significant_FDR']].sort_values('Abs_Difference', ascending=False)
                if len(sig_fdr) > 0:
                    print(f"\nStatistically Significant Feature Differences (FDR Corrected, q < 0.05):")
                    print("-" * 90)
                    print(f"{'Feature':<30} {'Men':<12} {'Women':<12} {'Difference':<12} {'P-Value':<10} {'FDR':<10}")
                    print("-" * 90)
                    for f in sig_fdr.index:
                        print(f"{f:<30} {pivot_df.loc[f, 'Men']:<12.6f} {pivot_df.loc[f, 'Women']:<12.6f} "
                              f"{pivot_df.loc[f, 'Difference']:<12.6f} {pivot_df.loc[f, 'P_Value']:<10.4f} "
                              f"{pivot_df.loc[f, 'P_Value_FDR']:<10.4f}")
                else:
                    print(f"\nNo features remain significant after FDR correction.")
                
                # Bonferroni corrected
                sig_bonf = pivot_df[pivot_df['Significant_Bonferroni']].sort_values('Abs_Difference', ascending=False)
                if len(sig_bonf) > 0:
                    print(f"\nStatistically Significant Feature Differences (Bonferroni Corrected, p < {bonferroni_alpha:.6f}):")
                    print("-" * 90)
                    print(f"{'Feature':<30} {'Men':<12} {'Women':<12} {'Difference':<12} {'P-Value':<10}")
                    print("-" * 90)
                    for f in sig_bonf.index:
                        print(f"{f:<30} {pivot_df.loc[f, 'Men']:<12.6f} {pivot_df.loc[f, 'Women']:<12.6f} "
                              f"{pivot_df.loc[f, 'Difference']:<12.6f} {pivot_df.loc[f, 'P_Value']:<10.4f}")
                else:
                    print(f"\nNo features remain significant after Bonferroni correction.")
                
                # Save corrected results
                pivot_df.to_csv(f'{output_dir}/raw_data_gender_feature_importance_pivot.csv')
                print(f"\nUpdated pivot table with corrections saved to {output_dir}/raw_data_gender_feature_importance_pivot.csv")
        
        return comparison_df, pivot_df

def test_feature_importance_significance(X_train_all, y_train_all, features_train, feature_names, n_bootstrap=100):
    """
    Test statistical significance of feature importance differences between genders using bootstrap.
    
    This function uses bootstrap resampling to create distributions of feature importance
    for each gender, then tests if the distributions are significantly different.
    """
    print(f"  Running bootstrap test (n={n_bootstrap})...")
    
    # Separate data by gender
    men_mask = features_train['gender'] == 'M'
    women_mask = features_train['gender'] == 'F'
    
    X_train_men = X_train_all[men_mask.values]
    y_train_men = y_train_all[men_mask.values]
    X_train_women = X_train_all[women_mask.values]
    y_train_women = y_train_all[women_mask.values]
    
    if len(X_train_men) < 50 or len(X_train_women) < 50:
        print("  Insufficient data for bootstrap test")
        return None
    
    # Bootstrap feature importance for men
    men_importances_boot = {f: [] for f in feature_names}
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(X_train_men), len(X_train_men), replace=True)
        X_boot = X_train_men.iloc[indices]
        y_boot = y_train_men.iloc[indices]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=None)
        model.fit(X_boot, y_boot)
        
        for i, feature in enumerate(feature_names):
            men_importances_boot[feature].append(model.feature_importances_[i])
    
    # Bootstrap feature importance for women
    women_importances_boot = {f: [] for f in feature_names}
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(X_train_women), len(X_train_women), replace=True)
        X_boot = X_train_women.iloc[indices]
        y_boot = y_train_women.iloc[indices]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=None)
        model.fit(X_boot, y_boot)
        
        for i, feature in enumerate(feature_names):
            women_importances_boot[feature].append(model.feature_importances_[i])
    
    # Test significance for each feature
    significance_results = {}
    for feature in feature_names:
        men_vals = np.array(men_importances_boot[feature])
        women_vals = np.array(women_importances_boot[feature])
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(men_vals, women_vals)
        
        # Also calculate effect size (Cohen's d)
        pooled_std = np.sqrt((men_vals.std()**2 + women_vals.std()**2) / 2)
        cohens_d = (men_vals.mean() - women_vals.mean()) / pooled_std if pooled_std > 0 else 0
        
        significance_results[feature] = {
            'p_value': p_value,
            't_statistic': t_stat,
            'cohens_d': cohens_d,
            'men_mean': men_vals.mean(),
            'women_mean': women_vals.mean(),
            'men_std': men_vals.std(),
            'women_std': women_vals.std()
        }
    
    return significance_results
    
    return None, None

def plot_residual_diagnostics(results, output_dir='output'):
    """
    Create residual plots for model diagnostics.
    
    This function generates three diagnostic plots for each model:
    1. Residuals vs Predicted: Checks for heteroscedasticity and non-linear patterns
    2. Q-Q Plot: Tests normality of residuals
    3. Residual Histogram: Visualizes the distribution of residuals
    
    These diagnostics help validate model assumptions and identify potential issues.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results containing 'residuals', 'y_pred', 'y_test'
    output_dir : str, default='output'
        Directory to save the diagnostic plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, result) in enumerate(results.items()):
        residuals = result['residuals']
        y_pred = result['y_pred']
        y_test = result['y_test']
        
        # Residuals vs Predicted
        axes[idx, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[idx, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[idx, 0].set_xlabel('Predicted Values')
        axes[idx, 0].set_ylabel('Residuals')
        axes[idx, 0].set_title(f'{name}\nResiduals vs Predicted')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=axes[idx, 1])
        axes[idx, 1].set_title(f'{name}\nQ-Q Plot (Normality Check)')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[idx, 2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[idx, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[idx, 2].set_xlabel('Residuals')
        axes[idx, 2].set_ylabel('Frequency')
        axes[idx, 2].set_title(f'{name}\nResidual Distribution')
        axes[idx, 2].grid(True, alpha=0.3)
        
        # Add normality test
        from scipy.stats import shapiro
        if len(residuals) <= 5000:  # Shapiro-Wilk works for n <= 5000
            stat, p_val = shapiro(residuals)
            axes[idx, 2].text(0.05, 0.95, f'Shapiro-Wilk: p={p_val:.4f}', 
                              transform=axes[idx, 2].transAxes,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                              verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/raw_data_residual_diagnostics.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residual diagnostics saved to {output_dir}/raw_data_residual_diagnostics.pdf")

def plot_predictions(results, test_metadata, output_dir='output'):
    """Plot actual vs predicted values for all models, split by year and gender."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Reset index to ensure alignment with y_test and y_pred arrays (which are positional)
    test_metadata = test_metadata.reset_index(drop=True)
    
    # Get unique years and genders
    years = sorted(test_metadata['year'].unique())
    genders = sorted(test_metadata['gender'].unique())
    gender_labels = {'M': 'Men', 'F': 'Women'}
    
    # Create a figure for each model, with subplots for each year/gender combination
    for model_name, result in results.items():
        if len(years) == 1 and len(genders) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            axes = [[ax]]
        else:
            fig, axes = plt.subplots(len(years), len(genders), figsize=(15, 12))
            # Handle case where there's only one year or one gender
            if len(years) == 1:
                axes = axes.reshape(1, -1)
            elif len(genders) == 1:
                axes = axes.reshape(-1, 1)
        
        for i, year in enumerate(years):
            for j, gender in enumerate(genders):
                if len(years) == 1 and len(genders) == 1:
                    ax = axes[0][0]
                elif len(years) == 1:
                    ax = axes[0, j]
                elif len(genders) == 1:
                    ax = axes[i, 0]
                else:
                    ax = axes[i, j]
                
                # Filter data for this year and gender
                mask = (test_metadata['year'] == year) & (test_metadata['gender'] == gender)
                
                if mask.sum() > 0:
                    # Convert to numpy arrays for indexing
                    y_test_array = np.array(result['y_test'])
                    y_pred_array = np.array(result['y_pred'])
                    
                    y_test_filtered = y_test_array[mask.values]
                    y_pred_filtered = y_pred_array[mask.values]
                    
                    # Plot actual vs predicted
                    ax.scatter(y_test_filtered, y_pred_filtered, alpha=0.6, s=30)
                    
                    # Add perfect prediction line
                    min_val = min(y_test_filtered.min(), y_pred_filtered.min())
                    max_val = max(y_test_filtered.max(), y_pred_filtered.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                    
                    # Calculate R² for this subset
                    from sklearn.metrics import r2_score
                    r2_subset = r2_score(y_test_filtered, y_pred_filtered)
                    
                    # Bootstrap CI for subset R²
                    r2_subset_mean, r2_subset_lower, r2_subset_upper = bootstrap_confidence_interval(
                        y_test_filtered, y_pred_filtered, r2_score, n_bootstrap=500
                    )
                    
                    ax.set_xlabel('Actual Improvement Rate (seconds/day)')
                    ax.set_ylabel('Predicted Improvement Rate (seconds/day)')
                    ax.set_title(f'{year} - {gender_labels.get(gender, gender)}\nR² = {r2_subset:.3f} (95% CI: [{r2_subset_lower:.3f}, {r2_subset_upper:.3f}])\nn={len(y_test_filtered)}', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{year} - {gender_labels.get(gender, gender)}', fontweight='bold')
        
        plt.suptitle(f'{model_name} - Predictions by Year and Gender', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save individual model plots
        safe_model_name = model_name.replace(' ', '_').lower()
        plt.savefig(f'{output_dir}/raw_data_model_predictions_{safe_model_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also create the original combined plot for all models
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
        ax.set_title(f'{name}\nR² = {result["r2"]:.4f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/raw_data_model_predictions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined plots: all 6 models for men, all 6 models for women
    print("\nCreating combined gender plots...")
    gender_labels = {'M': 'Men', 'F': 'Women'}
    
    for gender in ['M', 'F']:
        gender_label = gender_labels[gender]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Filter test metadata for this gender
        gender_mask = test_metadata['gender'] == gender
        
        for i, (model_name, result) in enumerate(results.items()):
            ax = axes[i]
            
            # Filter predictions for this gender
            y_test_array = np.array(result['y_test'])
            y_pred_array = np.array(result['y_pred'])
            
            y_test_gender = y_test_array[gender_mask.values]
            y_pred_gender = y_pred_array[gender_mask.values]
            
            if len(y_test_gender) > 0:
                # Plot actual vs predicted
                ax.scatter(y_test_gender, y_pred_gender, alpha=0.6, s=30)
                
                # Add perfect prediction line
                min_val = min(y_test_gender.min(), y_pred_gender.min())
                max_val = max(y_test_gender.max(), y_pred_gender.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                
                # Calculate R² for this gender
                r2_gender = r2_score(y_test_gender, y_pred_gender)
                
                # Bootstrap CI for gender R²
                r2_gender_mean, r2_gender_lower, r2_gender_upper = bootstrap_confidence_interval(
                    y_test_gender, y_pred_gender, r2_score, n_bootstrap=500
                )
                
                ax.set_xlabel('Actual Improvement Rate (seconds/day)')
                ax.set_ylabel('Predicted Improvement Rate (seconds/day)')
                ax.set_title(f'{model_name}\nR² = {r2_gender:.3f} (95% CI: [{r2_gender_lower:.3f}, {r2_gender_upper:.3f}])\nn={len(y_test_gender)}', fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name}', fontweight='bold')
        
        plt.suptitle(f'All Models - {gender_label} Predictions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save combined gender plot
        safe_gender = gender_label.lower()
        plt.savefig(f'{output_dir}/raw_data_model_predictions_all_models_{safe_gender}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved combined plot for {gender_label}: {output_dir}/raw_data_model_predictions_all_models_{safe_gender}.pdf")
    
    print(f"Prediction plots saved to {output_dir}/")

def analyze_subgroup_differences(results, test_metadata, output_dir='output'):
    """
    Perform statistical tests for differences between subgroups (year/gender).
    
    This function analyzes whether model performance differs significantly
    across different subgroups (years and genders) using t-tests on residuals.
    This is important for assessing model fairness and generalizability.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
    test_metadata : DataFrame
        Metadata for test set containing 'year' and 'gender' columns
    output_dir : str, default='output'
        Directory to save subgroup analysis results
    """
    print("\n" + "="*60)
    print("SUBGROUP ANALYSIS: YEAR AND GENDER DIFFERENCES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get best model (highest R²)
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    result = results[best_model_name]
    
    y_test_array = np.array(result['y_test'])
    y_pred_array = np.array(result['y_pred'])
    residuals = np.array(result['residuals'])
    
    # Reset index for alignment
    test_metadata = test_metadata.reset_index(drop=True)
    
    # Analyze by year
    print("\n1. Performance Differences by Year:")
    print("-" * 60)
    years = sorted(test_metadata['year'].unique())
    year_r2_scores = {}
    year_residuals = {}
    
    for year in years:
        mask = test_metadata['year'] == year
        if mask.sum() > 0:
            y_test_year = y_test_array[mask.values]
            y_pred_year = y_pred_array[mask.values]
            r2_year = r2_score(y_test_year, y_pred_year)
            year_r2_scores[year] = r2_year
            year_residuals[year] = residuals[mask.values]
            print(f"  {year}: R² = {r2_year:.4f}, n = {mask.sum()}")
    
    # Statistical test for year differences (if 2 years)
    if len(years) == 2:
        year1, year2 = years
        if year1 in year_residuals and year2 in year_residuals:
            # Test if residuals differ between years
            t_stat, p_val = stats.ttest_ind(year_residuals[year1], year_residuals[year2])
            print(f"\n  T-test for residual differences ({year1} vs {year2}):")
            print(f"    t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                print(f"    *** Significant difference (p < 0.05)")
            else:
                print(f"    No significant difference")
    
    # Analyze by gender
    print("\n2. Performance Differences by Gender:")
    print("-" * 60)
    genders = sorted(test_metadata['gender'].unique())
    gender_labels = {'M': 'Men', 'F': 'Women'}
    gender_r2_scores = {}
    gender_residuals = {}
    
    for gender in genders:
        mask = test_metadata['gender'] == gender
        if mask.sum() > 0:
            y_test_gender = y_test_array[mask.values]
            y_pred_gender = y_pred_array[mask.values]
            r2_gender = r2_score(y_test_gender, y_pred_gender)
            gender_r2_scores[gender] = r2_gender
            gender_residuals[gender] = residuals[mask.values]
            print(f"  {gender_labels.get(gender, gender)}: R² = {r2_gender:.4f}, n = {mask.sum()}")
    
    # Statistical test for gender differences (if 2 genders)
    if len(genders) == 2:
        g1, g2 = genders
        if g1 in gender_residuals and g2 in gender_residuals:
            # Test if residuals differ between genders
            t_stat, p_val = stats.ttest_ind(gender_residuals[g1], gender_residuals[g2])
            print(f"\n  T-test for residual differences ({gender_labels.get(g1, g1)} vs {gender_labels.get(g2, g2)}):")
            print(f"    t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                print(f"    *** Significant difference (p < 0.05)")
            else:
                print(f"    No significant difference")
    
    # Analyze year × gender interactions
    print("\n3. Year × Gender Interactions:")
    print("-" * 60)
    subgroup_stats = []
    
    for year in years:
        for gender in genders:
            mask = (test_metadata['year'] == year) & (test_metadata['gender'] == gender)
            if mask.sum() > 0:
                y_test_sub = y_test_array[mask.values]
                y_pred_sub = y_pred_array[mask.values]
                r2_sub = r2_score(y_test_sub, y_pred_sub)
                mae_sub = mean_absolute_error(y_test_sub, y_pred_sub)
                
                subgroup_stats.append({
                    'Year': year,
                    'Gender': gender_labels.get(gender, gender),
                    'R²': r2_sub,
                    'MAE': mae_sub,
                    'n': mask.sum()
                })
                print(f"  {year} - {gender_labels.get(gender, gender)}: R² = {r2_sub:.4f}, MAE = {mae_sub:.4f}, n = {mask.sum()}")
    
    # Save subgroup statistics
    subgroup_df = pd.DataFrame(subgroup_stats)
    subgroup_df.to_csv(f'{output_dir}/raw_data_subgroup_analysis.csv', index=False)
    print(f"\nSubgroup statistics saved to {output_dir}/raw_data_subgroup_analysis.csv")

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
            'R² CI Lower': result.get('r2_ci_lower', np.nan),
            'R² CI Upper': result.get('r2_ci_upper', np.nan),
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

def save_important_statistics(results, test_metadata, features_df, best_model, feature_names, output_dir='output'):
    """
    Compile and save all important statistics to a comprehensive CSV file.
    
    This function collects all key statistics from the analysis including:
    - Model performance metrics with confidence intervals
    - Subgroup analysis (year/gender)
    - Statistical test results
    - Dataset summary statistics
    - Feature importance summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = []
    
    # 1. Model Performance Statistics
    print("\nCompiling important statistics...")
    for name, result in results.items():
        all_stats.append({
            'Category': 'Model Performance',
            'Metric': 'R² Score',
            'Model': name,
            'Value': result['r2'],
            'CI_Lower': result.get('r2_ci_lower', np.nan),
            'CI_Upper': result.get('r2_ci_upper', np.nan),
            'Subgroup': 'Overall',
            'Year': 'All',
            'Gender': 'All',
            'N': len(result['y_test']),
            'Additional_Info': ''
        })
        
        all_stats.append({
            'Category': 'Model Performance',
            'Metric': 'RMSE',
            'Model': name,
            'Value': result['rmse'],
            'CI_Lower': np.nan,
            'CI_Upper': np.nan,
            'Subgroup': 'Overall',
            'Year': 'All',
            'Gender': 'All',
            'N': len(result['y_test']),
            'Additional_Info': ''
        })
        
        all_stats.append({
            'Category': 'Model Performance',
            'Metric': 'MAE',
            'Model': name,
            'Value': result['mae'],
            'CI_Lower': np.nan,
            'CI_Upper': np.nan,
            'Subgroup': 'Overall',
            'Year': 'All',
            'Gender': 'All',
            'N': len(result['y_test']),
            'Additional_Info': ''
        })
        
        all_stats.append({
            'Category': 'Model Performance',
            'Metric': 'CV R² Mean',
            'Model': name,
            'Value': result['cv_r2_mean'],
            'CI_Lower': result['cv_r2_mean'] - result['cv_r2_std'],
            'CI_Upper': result['cv_r2_mean'] + result['cv_r2_std'],
            'Subgroup': 'Overall',
            'Year': 'All',
            'Gender': 'All',
            'N': len(result['y_test']),
            'Additional_Info': f'Std: {result["cv_r2_std"]:.4f}'
        })
    
    # 2. Subgroup Analysis Statistics
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    result = results[best_model_name]
    y_test_array = np.array(result['y_test'])
    y_pred_array = np.array(result['y_pred'])
    test_metadata_reset = test_metadata.reset_index(drop=True)
    
    gender_labels = {'M': 'Men', 'F': 'Women'}
    years = sorted(test_metadata_reset['year'].unique())
    genders = sorted(test_metadata_reset['gender'].unique())
    
    # By Year
    for year in years:
        mask = test_metadata_reset['year'] == year
        if mask.sum() > 0:
            y_test_year = y_test_array[mask.values]
            y_pred_year = y_pred_array[mask.values]
            r2_year = r2_score(y_test_year, y_pred_year)
            mae_year = mean_absolute_error(y_test_year, y_pred_year)
            rmse_year = np.sqrt(mean_squared_error(y_test_year, y_pred_year))
            
            r2_mean, r2_lower, r2_upper = bootstrap_confidence_interval(
                y_test_year, y_pred_year, r2_score, n_bootstrap=500
            )
            
            all_stats.append({
                'Category': 'Subgroup Analysis',
                'Metric': 'R² Score',
                'Model': best_model_name,
                'Value': r2_year,
                'CI_Lower': r2_lower,
                'CI_Upper': r2_upper,
                'Subgroup': 'Year',
                'Year': year,
                'Gender': 'All',
                'N': mask.sum(),
                'Additional_Info': ''
            })
            
            all_stats.append({
                'Category': 'Subgroup Analysis',
                'Metric': 'MAE',
                'Model': best_model_name,
                'Value': mae_year,
                'CI_Lower': np.nan,
                'CI_Upper': np.nan,
                'Subgroup': 'Year',
                'Year': year,
                'Gender': 'All',
                'N': mask.sum(),
                'Additional_Info': ''
            })
    
    # By Gender
    for gender in genders:
        mask = test_metadata_reset['gender'] == gender
        if mask.sum() > 0:
            y_test_gender = y_test_array[mask.values]
            y_pred_gender = y_pred_array[mask.values]
            r2_gender = r2_score(y_test_gender, y_pred_gender)
            mae_gender = mean_absolute_error(y_test_gender, y_pred_gender)
            rmse_gender = np.sqrt(mean_squared_error(y_test_gender, y_pred_gender))
            
            r2_mean, r2_lower, r2_upper = bootstrap_confidence_interval(
                y_test_gender, y_pred_gender, r2_score, n_bootstrap=500
            )
            
            all_stats.append({
                'Category': 'Subgroup Analysis',
                'Metric': 'R² Score',
                'Model': best_model_name,
                'Value': r2_gender,
                'CI_Lower': r2_lower,
                'CI_Upper': r2_upper,
                'Subgroup': 'Gender',
                'Year': 'All',
                'Gender': gender_labels.get(gender, gender),
                'N': mask.sum(),
                'Additional_Info': ''
            })
            
            all_stats.append({
                'Category': 'Subgroup Analysis',
                'Metric': 'MAE',
                'Model': best_model_name,
                'Value': mae_gender,
                'CI_Lower': np.nan,
                'CI_Upper': np.nan,
                'Subgroup': 'Gender',
                'Year': 'All',
                'Gender': gender_labels.get(gender, gender),
                'N': mask.sum(),
                'Additional_Info': ''
            })
    
    # Year × Gender combinations
    for year in years:
        for gender in genders:
            mask = (test_metadata_reset['year'] == year) & (test_metadata_reset['gender'] == gender)
            if mask.sum() > 0:
                y_test_sub = y_test_array[mask.values]
                y_pred_sub = y_pred_array[mask.values]
                r2_sub = r2_score(y_test_sub, y_pred_sub)
                mae_sub = mean_absolute_error(y_test_sub, y_pred_sub)
                
                r2_mean, r2_lower, r2_upper = bootstrap_confidence_interval(
                    y_test_sub, y_pred_sub, r2_score, n_bootstrap=500
                )
                
                all_stats.append({
                    'Category': 'Subgroup Analysis',
                    'Metric': 'R² Score',
                    'Model': best_model_name,
                    'Value': r2_sub,
                    'CI_Lower': r2_lower,
                    'CI_Upper': r2_upper,
                    'Subgroup': 'Year × Gender',
                    'Year': year,
                    'Gender': gender_labels.get(gender, gender),
                    'N': mask.sum(),
                    'Additional_Info': ''
                })
    
    # 3. Dataset Summary Statistics
    all_stats.append({
        'Category': 'Dataset Summary',
        'Metric': 'Total Athletes',
        'Model': 'N/A',
        'Value': len(features_df),
        'CI_Lower': np.nan,
        'CI_Upper': np.nan,
        'Subgroup': 'Overall',
        'Year': 'All',
        'Gender': 'All',
        'N': len(features_df),
        'Additional_Info': ''
    })
    
    all_stats.append({
        'Category': 'Dataset Summary',
        'Metric': 'Mean Improvement Rate',
        'Model': 'N/A',
        'Value': features_df['improvement_rate'].mean(),
        'CI_Lower': np.nan,
        'CI_Upper': np.nan,
        'Subgroup': 'Overall',
        'Year': 'All',
        'Gender': 'All',
        'N': len(features_df),
        'Additional_Info': f'Std: {features_df["improvement_rate"].std():.4f}'
    })
    
    # By gender in dataset
    for gender in features_df['gender'].unique():
        gender_data = features_df[features_df['gender'] == gender]
        all_stats.append({
            'Category': 'Dataset Summary',
            'Metric': 'Mean Improvement Rate',
            'Model': 'N/A',
            'Value': gender_data['improvement_rate'].mean(),
            'CI_Lower': np.nan,
            'CI_Upper': np.nan,
            'Subgroup': 'Gender',
            'Year': 'All',
            'Gender': gender_labels.get(gender, gender),
            'N': len(gender_data),
            'Additional_Info': f'Std: {gender_data["improvement_rate"].std():.4f}'
        })
    
    # By year in dataset
    for year in features_df['year'].unique():
        year_data = features_df[features_df['year'] == year]
        all_stats.append({
            'Category': 'Dataset Summary',
            'Metric': 'Mean Improvement Rate',
            'Model': 'N/A',
            'Value': year_data['improvement_rate'].mean(),
            'CI_Lower': np.nan,
            'CI_Upper': np.nan,
            'Subgroup': 'Year',
            'Year': year,
            'Gender': 'All',
            'N': len(year_data),
            'Additional_Info': f'Std: {year_data["improvement_rate"].std():.4f}'
        })
    
    # 4. Feature Importance (Top 10)
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for rank, (idx, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            all_stats.append({
                'Category': 'Feature Importance',
                'Metric': 'Importance Score',
                'Model': best_model_name,
                'Value': row['importance'],
                'CI_Lower': np.nan,
                'CI_Upper': np.nan,
                'Subgroup': 'Feature',
                'Year': 'N/A',
                'Gender': 'N/A',
                'N': np.nan,
                'Additional_Info': f'Feature: {row["feature"]}, Rank: {rank}'
            })
    
    # Convert to DataFrame and save
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.sort_values(['Category', 'Metric', 'Model', 'Subgroup', 'Year', 'Gender'])
    
    output_path = f'{output_dir}/raw_data_important_statistics.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"\nImportant statistics saved to {output_path}")
    print(f"Total statistics compiled: {len(stats_df)}")
    
    return stats_df

def main():
    """Main function to run the raw data improvement prediction analysis."""
    print("="*60)
    print("RAW DATA IMPROVEMENT PREDICTION MODEL")
    print("="*60)
    
    # Load raw data (using standardized times - best method)
    df = load_raw_data(mode='standardized')
    
    # Calculate athlete features
    athlete_df = calculate_athlete_features(df)
    
    # Create advanced features
    features_df = create_advanced_features(athlete_df)
    
    # Prepare model data
    X, y, features_df_filtered = prepare_model_data(features_df)
    
    # Train models with temporal validation (train on 2023, test on 2024)
    print("\nUsing temporal validation: training on 2023 data, testing on 2024 data")
    results, X_test, y_test, test_metadata = train_models(X, y, features_df_filtered, use_temporal_split=True)
    
    # Extract training data for hyperparameter tuning (2023 data only)
    train_mask = features_df_filtered['year'] == 2023
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Test R² Score: {results[best_model_name]['r2']:.4f}")
    print(f"95% CI: [{results[best_model_name].get('r2_ci_lower', 'N/A'):.4f}, {results[best_model_name].get('r2_ci_upper', 'N/A'):.4f}]")
    
    # Hyperparameter tuning (optional - can be time-consuming)
    print("\nPerforming hyperparameter tuning...")
    tuned_model = hyperparameter_tuning(X_train, y_train, best_model_name)
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    feature_names = X.columns.tolist()
    feature_importance_df = analyze_feature_importance(best_model, feature_names)
    
    # Plot predictions
    print("\nCreating prediction plots...")
    plot_predictions(results, test_metadata)
    
    # Create residual diagnostics
    print("\nCreating residual diagnostics...")
    plot_residual_diagnostics(results)
    
    # Analyze subgroup differences
    print("\nAnalyzing subgroup differences...")
    analyze_subgroup_differences(results, test_metadata)
    
    # Analyze gender-specific feature importance
    print("\nAnalyzing gender-specific feature importance...")
    gender_importance_df, gender_pivot_df = analyze_gender_specific_feature_importance(
        X, y, features_df_filtered
    )
    
    # Generate insights
    create_improvement_insights(features_df, best_model, feature_names)
    
    # Save results
    print("\nSaving results...")
    save_model_results(results, features_df, feature_names)
    
    # Save important statistics
    print("\nSaving important statistics...")
    stats_df = save_important_statistics(results, test_metadata, features_df, best_model, feature_names)
    
    # Test if season_duration_squared is necessary
    print("\n" + "="*60)
    print("TESTING: Is season_duration_squared necessary?")
    print("="*60)
    test_feature_redundancy(X, y, features_df_filtered, feature_names)
    
    # Analyze race frequency by gender
    print("\n" + "="*60)
    print("ANALYZING: Race Frequency Distribution by Gender")
    print("="*60)
    analyze_race_frequency_by_gender(features_df_filtered)
    
    # Compare standardized vs converted vs original times
    print("\n" + "="*60)
    print("COMPARING: Standardized vs Converted vs Original Times")
    print("="*60)
    compare_time_standardization_methods()
    
    print("\nAnalysis complete!")

def compare_time_standardization_methods(output_dir='output'):
    """
    Compare model performance using:
    1. Standardized times (weather, terrain, distance adjustments)
    2. Converted-only times (distance only, no weather/terrain)
    3. Original times (no adjustments)
    
    This demonstrates that standardization improves model performance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nComparing three time standardization approaches:")
    print("1. Standardized: Weather + Terrain + Distance adjustments (BEST)")
    print("2. Converted: Distance conversion only (no weather/terrain) (MIDDLE)")
    print("3. Raw: No adjustments at all - original race times (WORST)")
    print("\nNote: Standardized and Converted methods convert distances to 6k (women) / 8k (men).")
    print("Raw method uses original times without distance conversion (less comparable).")
    
    comparison_results = []
    
    for mode in ['standardized', 'converted', 'raw']:
        print(f"\n{'='*60}")
        print(f"Analyzing: {mode.upper()} times")
        print('='*60)
        
        try:
            # Load data with specified mode
            df = load_raw_data(mode=mode)
            
            # Calculate athlete features
            athlete_df = calculate_athlete_features(df)
            
            if len(athlete_df) < 100:
                print(f"  Insufficient data for {mode} (n={len(athlete_df)})")
                continue
            
            # Create advanced features
            features_df = create_advanced_features(athlete_df)
            
            # Prepare model data
            X, y, features_df_filtered = prepare_model_data(features_df)
            
            # Temporal split
            train_mask = features_df_filtered['year'] == 2023
            test_mask = features_df_filtered['year'] == 2024
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Train best model (Gradient Boosting)
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Bootstrap CI
            r2_mean, r2_lower, r2_upper = bootstrap_confidence_interval(
                y_test, y_pred, r2_score, n_bootstrap=1000
            )
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            comparison_results.append({
                'Method': mode.capitalize(),
                'R² Score': r2,
                'R² CI Lower': r2_lower,
                'R² CI Upper': r2_upper,
                'RMSE': rmse,
                'MAE': mae,
                'CV R² Mean': cv_scores.mean(),
                'CV R² Std': cv_scores.std(),
                'Train n': len(X_train),
                'Test n': len(X_test)
            })
            
            print(f"  Test R²: {r2:.4f} (95% CI: [{r2_lower:.4f}, {r2_upper:.4f}])")
            print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
        except Exception as e:
            print(f"  Error analyzing {mode}: {str(e)}")
            continue
    
    # Create comparison visualization
    if len(comparison_results) >= 2:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(f'{output_dir}/time_standardization_comparison.csv', index=False)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        methods = comparison_df['Method'].values
        r2_scores = comparison_df['R² Score'].values
        r2_lower = comparison_df['R² CI Lower'].values
        r2_upper = comparison_df['R² CI Upper'].values
        rmse_scores = comparison_df['RMSE'].values
        mae_scores = comparison_df['MAE'].values
        
        # Plot 1: R² Scores with CIs
        x_pos = np.arange(len(methods))
        bars1 = axes[0].bar(x_pos, r2_scores, alpha=0.7, 
                           color=['#2E86AB', '#A23B72', '#6C757D'][:len(methods)])
        axes[0].errorbar(x_pos, r2_scores, 
                         yerr=[r2_scores - r2_lower, r2_upper - r2_scores],
                         fmt='none', color='black', capsize=5, capthick=2)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Model Performance by Time Standardization Method\n(Higher is Better)')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim(0, 1.0)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
            axes[0].text(bar.get_x() + bar.get_width()/2., score + 0.02,
                         f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: RMSE
        bars2 = axes[1].bar(x_pos, rmse_scores, alpha=0.7,
                           color=['#2E86AB', '#A23B72', '#6C757D'][:len(methods)])
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1].set_ylabel('RMSE (seconds/day)')
        axes[1].set_title('RMSE by Time Standardization Method\n(Lower is Better)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
            axes[1].text(bar.get_x() + bar.get_width()/2., score + max(rmse_scores)*0.02,
                         f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: MAE
        bars3 = axes[2].bar(x_pos, mae_scores, alpha=0.7,
                           color=['#2E86AB', '#A23B72', '#6C757D'][:len(methods)])
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(methods, rotation=45, ha='right')
        axes[2].set_ylabel('MAE (seconds/day)')
        axes[2].set_title('MAE by Time Standardization Method\n(Lower is Better)')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars3, mae_scores)):
            axes[2].text(bar.get_x() + bar.get_width()/2., score + max(mae_scores)*0.02,
                         f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_standardization_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*60}")
        print("STANDARDIZATION COMPARISON RESULTS")
        print('='*60)
        print(comparison_df.to_string(index=False))
        print(f"\nComparison saved to:")
        print(f"  - {output_dir}/time_standardization_comparison.csv")
        print(f"  - {output_dir}/time_standardization_comparison.pdf")
        
        # Statistical comparison
        if len(comparison_results) >= 2:
            print(f"\nPerformance Improvement:")
            if 'Standardized' in comparison_df['Method'].values and 'Converted' in comparison_df['Method'].values:
                std_r2 = comparison_df[comparison_df['Method'] == 'Standardized']['R² Score'].values[0]
                conv_r2 = comparison_df[comparison_df['Method'] == 'Converted']['R² Score'].values[0]
                improvement = ((std_r2 - conv_r2) / conv_r2) * 100
                print(f"  Standardized vs Converted: {improvement:+.1f}% improvement in R²")
            
            if 'Standardized' in comparison_df['Method'].values and 'Original' in comparison_df['Method'].values:
                std_r2 = comparison_df[comparison_df['Method'] == 'Standardized']['R² Score'].values[0]
                orig_r2 = comparison_df[comparison_df['Method'] == 'Original']['R² Score'].values[0]
                improvement = ((std_r2 - orig_r2) / orig_r2) * 100
                print(f"  Standardized vs Original: {improvement:+.1f}% improvement in R²")
        
        return comparison_df
    
    return None

def test_feature_redundancy(X, y, features_df, feature_names):
    """Test if season_duration_squared is redundant."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    
    # Temporal split
    train_mask = features_df['year'] == 2023
    test_mask = features_df['year'] == 2024
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    # Model with both terms
    model_both = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_both.fit(X_train, y_train)
    r2_both = r2_score(y_test, model_both.predict(X_test))
    
    # Model without squared term
    feature_cols_no_squared = [f for f in feature_names if f != 'season_duration_squared']
    X_train_no_sq = X_train[feature_cols_no_squared]
    X_test_no_sq = X_test[feature_cols_no_squared]
    
    model_no_squared = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_no_squared.fit(X_train_no_sq, y_train)
    r2_no_squared = r2_score(y_test, model_no_squared.predict(X_test_no_sq))
    
    # Model without linear term (only squared)
    feature_cols_only_squared = [f for f in feature_names if f != 'season_duration']
    X_train_only_sq = X_train[feature_cols_only_squared]
    X_test_only_sq = X_test[feature_cols_only_squared]
    
    model_only_squared = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_only_squared.fit(X_train_only_sq, y_train)
    r2_only_squared = r2_score(y_test, model_only_squared.predict(X_test_only_sq))
    
    print(f"\nModel Performance Comparison:")
    print(f"  With both terms:        R² = {r2_both:.4f}")
    print(f"  Without squared term:   R² = {r2_no_squared:.4f} (Δ = {r2_both - r2_no_squared:.4f})")
    print(f"  Without linear term:    R² = {r2_only_squared:.4f} (Δ = {r2_both - r2_only_squared:.4f})")
    
    if abs(r2_both - r2_no_squared) < 0.01:
        print(f"\n  ⚠️  Squared term adds <0.01 to R² - may be redundant")
    else:
        print(f"\n  ✅ Squared term adds {r2_both - r2_no_squared:.4f} to R² - keep both terms")
    
    if abs(r2_both - r2_only_squared) < 0.01:
        print(f"  ⚠️  Linear term adds <0.01 to R² - may be redundant")
    else:
        print(f"  ✅ Linear term adds {r2_both - r2_only_squared:.4f} to R² - keep both terms")

def analyze_race_frequency_by_gender(features_df):
    """Analyze race frequency distribution by gender to understand why it's more important for women."""
    print("\nRace Frequency Statistics by Gender:")
    print("-" * 60)
    
    for gender in ['M', 'F']:
        gender_label = 'Men' if gender == 'M' else 'Women'
        gender_data = features_df[features_df['gender'] == gender]
        
        print(f"\n{gender_label}:")
        print(f"  Mean race frequency: {gender_data['race_frequency'].mean():.4f} races/day")
        print(f"  Std race frequency:  {gender_data['race_frequency'].std():.4f}")
        print(f"  Min: {gender_data['race_frequency'].min():.4f}, Max: {gender_data['race_frequency'].max():.4f}")
        print(f"  Median: {gender_data['race_frequency'].median():.4f}")
        
        # Distribution by bins
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0]
        print(f"  Distribution:")
        for i in range(len(bins)-1):
            count = ((gender_data['race_frequency'] >= bins[i]) & 
                    (gender_data['race_frequency'] < bins[i+1])).sum()
            pct = count / len(gender_data) * 100
            print(f"    {bins[i]:.2f}-{bins[i+1]:.2f} races/day: {count} ({pct:.1f}%)")
    
    # Statistical test
    men_freq = features_df[features_df['gender'] == 'M']['race_frequency']
    women_freq = features_df[features_df['gender'] == 'F']['race_frequency']
    
    t_stat, p_val = stats.ttest_ind(men_freq, women_freq)
    print(f"\nT-test for race frequency difference (Men vs Women):")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print(f"  *** Significant difference (p < 0.05)")
        if men_freq.mean() > women_freq.mean():
            print(f"  Men race more frequently on average")
        else:
            print(f"  Women race more frequently on average")
    else:
        print(f"  No significant difference in race frequency")
    
    # Correlation with improvement
    print(f"\nCorrelation between race_frequency and improvement_rate:")
    for gender in ['M', 'F']:
        gender_label = 'Men' if gender == 'M' else 'Women'
        gender_data = features_df[features_df['gender'] == gender]
        corr = gender_data['race_frequency'].corr(gender_data['improvement_rate'])
        print(f"  {gender_label}: r = {corr:.4f}")

if __name__ == "__main__":
    main() 