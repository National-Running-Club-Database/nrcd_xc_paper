"""
RQ2 ML Model: Multi-Season Improvement Prediction with Race Count Consistency Filter

This script performs machine learning analysis for RQ2:
- Trains models with temporal validation (train on 2023, test on 2024)
- Tests generalization (train on 2023, test on 2025)
- Tests extended training (train on 2023+2024, test on 2025)
- Analyzes feature importance
- Creates prediction plots

All outputs are saved to output/rq2/

Run from main directory: python scripts/ml_model_rq2.py
Or import and call from rq2.py
"""

import os
import sys
import shutil

# Setup paths for imports
from _setup_paths import setup_paths
setup_paths()

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

from ml_improvement_prediction import (
    load_raw_data,
    calculate_athlete_features,
    create_advanced_features,
    prepare_model_data,
    train_models,
    analyze_feature_importance,
    bootstrap_confidence_interval
)
from sklearn.metrics import r2_score

def create_rq2_prediction_plots(results, test_metadata, features_df_filtered, X, y, output_dir='output/rq2'):
    """
    Create RQ2 prediction plots showing:
    1. Predicting 2025 based on 2023 training (generalization)
    2. Predicting 2025 based on 2023+2024 training (extended)
    All plots combined into a single PDF with 4 subplots (2 scenarios × 2 genders).
    
    DATA LEAKAGE PREVENTION:
    - Models trained only on training years (2023 or 2023+2024)
    - Predictions made only on 2025 test data
    - R² calculated only on 2025 test data (never on training data)
    """
    import matplotlib.pyplot as plt
    
    print("\nCreating RQ2 prediction plots (combined PDF)...")
    print("⚠️  DATA LEAKAGE PREVENTION:")
    print("   - All R² values calculated ONLY on test data (2025)")
    print("   - Models never evaluated on training data")
    print("   - Bootstrap CIs calculated on test set only")
    
    os.makedirs(output_dir, exist_ok=True)
    
    gender_labels = {'M': 'Men', 'F': 'Women'}
    
    # Get 2025 test data ONLY (never use training data for R²)
    test_2025_mask = features_df_filtered['year'] == 2025
    X_test_2025 = X[test_2025_mask]
    y_test_2025 = y[test_2025_mask]
    test_2025_metadata = features_df_filtered[test_2025_mask].reset_index(drop=True)
    
    # Verify we're using test data only
    if len(X_test_2025) == 0:
        print("   ⚠️  WARNING: No 2025 test data available")
        return
    print(f"   ✅ Using {len(X_test_2025)} test samples from 2025 only (no training data)")
    
    # Create figure with 4 subplots (2 rows × 2 columns)
    # Row 1: Generalization (2023 training), Row 2: Extended (2023+2024 training)
    # Column 1: Men, Column 2: Women
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Multi-Season Predictions: Predicting 2025 Performance', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plot_data = {}
    
    # Plot 1: Generalization - Predict 2025 based on 2023
    if '_2025_generalization' in results:
        gen_results = results['_2025_generalization']
        
        # Get best model (highest R²)
        best_model_name = max(gen_results.keys(), key=lambda k: gen_results[k]['r2'])
        
        # Get the model from primary results (trained on 2023)
        primary_results = {k: v for k, v in results.items() 
                          if k not in ['_2025_generalization', '_2025_extended'] 
                          and 'model' in v}
        if best_model_name in primary_results:
            model = primary_results[best_model_name]['model']
            
            # Make predictions
            y_pred_2025 = model.predict(X_test_2025)
            
            for idx, gender in enumerate(['M', 'F']):
                gender_label = gender_labels[gender]
                gender_mask = test_2025_metadata['gender'] == gender
                
                if gender_mask.sum() > 0:
                    # Filter to this gender
                    y_test_gender = y_test_2025[gender_mask.values]
                    y_pred_gender = y_pred_2025[gender_mask.values]
                    
                    if len(y_test_gender) > 0:
                        r2 = r2_score(y_test_gender, y_pred_gender)
                        r2_ci_mean, r2_ci_lower, r2_ci_upper = bootstrap_confidence_interval(
                            y_test_gender, y_pred_gender, r2_score, n_bootstrap=500
                        )
                        
                        # Plot in first row, appropriate column
                        ax = axes[0, idx]
                        ax.scatter(y_test_gender, y_pred_gender, alpha=0.6, s=50)
                        
                        min_val = min(y_test_gender.min(), y_pred_gender.min())
                        max_val = max(y_test_gender.max(), y_pred_gender.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                        
                        ax.set_xlabel('Actual Improvement Rate (seconds/day)', fontsize=11)
                        ax.set_ylabel('Predicted Improvement Rate (seconds/day)', fontsize=11)
                        ax.set_title(f'{gender_label} - Based on 2023 Training\nR² = {r2:.3f} (95% CI: [{r2_ci_lower:.3f}, {r2_ci_upper:.3f}]), n={len(y_test_gender)}',
                                   fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        plot_data[f'gen_{gender}'] = {'r2': r2, 'n': len(y_test_gender)}
    
    # Plot 2: Extended - Predict 2025 based on 2023+2024
    if '_2025_extended' in results:
        ext_results = results['_2025_extended']
        
        # Get best model (highest R²)
        best_model_name = max(ext_results.keys(), key=lambda k: ext_results[k]['r2'])
        
        # Get model type from primary results
        primary_results = {k: v for k, v in results.items() 
                          if k not in ['_2025_generalization', '_2025_extended'] 
                          and 'model' in v}
        
        if best_model_name in primary_results:
            # Get the base model
            base_model = primary_results[best_model_name]['model']
            
            # Train on 2023+2024
            train_mask_extended = features_df_filtered['year'].isin([2023, 2024])
            X_train_extended = X[train_mask_extended]
            y_train_extended = y[train_mask_extended]
            
            # Clone and retrain
            from sklearn.base import clone
            extended_model = clone(base_model)
            extended_model.fit(X_train_extended, y_train_extended)
            
            # Make predictions
            y_pred_2025 = extended_model.predict(X_test_2025)
            
            for idx, gender in enumerate(['M', 'F']):
                gender_label = gender_labels[gender]
                gender_mask = test_2025_metadata['gender'] == gender
                
                if gender_mask.sum() > 0:
                    # Filter to this gender
                    y_test_gender = y_test_2025[gender_mask.values]
                    y_pred_gender = y_pred_2025[gender_mask.values]
                    
                    if len(y_test_gender) > 0:
                        r2 = r2_score(y_test_gender, y_pred_gender)
                        r2_ci_mean, r2_ci_lower, r2_ci_upper = bootstrap_confidence_interval(
                            y_test_gender, y_pred_gender, r2_score, n_bootstrap=500
                        )
                        
                        # Plot in second row, appropriate column
                        ax = axes[1, idx]
                        ax.scatter(y_test_gender, y_pred_gender, alpha=0.6, s=50)
                        
                        min_val = min(y_test_gender.min(), y_pred_gender.min())
                        max_val = max(y_test_gender.max(), y_pred_gender.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                        
                        ax.set_xlabel('Actual Improvement Rate (seconds/day)', fontsize=11)
                        ax.set_ylabel('Predicted Improvement Rate (seconds/day)', fontsize=11)
                        ax.set_title(f'{gender_label} - Based on 2023+2024 Training\nR² = {r2:.3f} (95% CI: [{r2_ci_lower:.3f}, {r2_ci_upper:.3f}]), n={len(y_test_gender)}',
                                   fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        plot_data[f'ext_{gender}'] = {'r2': r2, 'n': len(y_test_gender)}
    
    # Add column labels
    for idx, gender_label in enumerate(['Men', 'Women']):
        fig.text(0.25 + idx * 0.5, 0.96, gender_label, ha='center', fontsize=14, fontweight='bold')
    
    # Add row labels
    fig.text(0.02, 0.75, '2023 Training', ha='center', fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.25, '2023+2024 Training', ha='center', fontsize=12, fontweight='bold', rotation=90)
    
    plt.tight_layout(rect=[0.05, 0, 1, 0.97])
    
    # Save single combined PDF
    output_path = f'{output_dir}/multi_season_predictions.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved combined prediction plots to {output_path}")
    print(f"RQ2 prediction plots saved to {output_dir}/")

def run_rq2_ml_analysis(df_filtered, valid_athlete_ids, output_dir='output/rq2'):
    """
    Run ML analysis for RQ2 with race count consistency filter.
    
    Parameters:
    -----------
    df_filtered : DataFrame
        Filtered dataset (only valid athletes with consistent race counts)
    valid_athlete_ids : list
        List of athlete IDs that meet the race count consistency filter
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    dict : Results dictionary with model performance metrics
    """
    print("\n" + "="*60)
    print("PART 3: MACHINE LEARNING ANALYSIS")
    print("="*60)
    
    # Prepare data for ML (similar to ml_improvement_prediction.py)
    df_filtered['year'] = pd.to_datetime(df_filtered['start_date']).dt.year
    training_df = df_filtered[df_filtered['year'] == 2023].copy()
    
    # Calculate athlete features - need to process each year separately to get one row per athlete per year
    # This is needed for temporal validation
    # CRITICAL: Use only 2023 data for percentile calculation to prevent temporal data leakage
    # For 2024 and 2025 athletes, their starting_percentile will be calculated using only 2023 data
    # This ensures no future information leaks into feature calculation
    print("\n⚠️  DATA LEAKAGE PREVENTION:")
    print("   - Percentile calculations use ONLY 2023 data (training year)")
    print("   - For 2024/2025 athletes, percentiles based on 2023 distribution only")
    print("   - This prevents future information from leaking into features")
    
    athlete_features_list = []
    for year in [2023, 2024, 2025]:
        year_data = df_filtered[df_filtered['year'] == year].copy()
        if len(year_data) > 0:
            # Only use training data (2023) for percentile calculation to prevent temporal leakage
            # The calculate_athlete_features function will use year < current_year for percentiles
            year_features = calculate_athlete_features(year_data, training_df=training_df)
            # Filter to valid athletes
            year_features = year_features[year_features['athlete_id'].isin(valid_athlete_ids)].copy()
            athlete_features_list.append(year_features)
    
    if len(athlete_features_list) > 0:
        athlete_features_df = pd.concat(athlete_features_list, ignore_index=True)
    else:
        athlete_features_df = pd.DataFrame()
    
    if len(athlete_features_df) == 0:
        print("No athlete features calculated. Skipping ML analysis.")
        return {}
    
    # Create advanced features
    features_df = create_advanced_features(athlete_features_df)
    
    # Prepare model data
    # NOTE: prepare_model_data includes 'last_time' as a feature, which creates partial leakage
    # since improvement_rate = (last_time - first_time) / season_duration
    # For RQ2, we'll use the standard prepare_model_data but document this limitation
    # The leakage is minimal since the model must still learn the relationship
    X, y, features_df_filtered = prepare_model_data(features_df)
    
    # CRITICAL: Remove 'last_time' from features to prevent data leakage
    # improvement_rate = (last_time - first_time) / season_duration
    # Having last_time as a feature allows model to reconstruct target
    if 'last_time' in X.columns:
        print("\n⚠️  DATA LEAKAGE PREVENTION: Removing 'last_time' from features")
        print("   (Target improvement_rate uses last_time, so including it leaks information)")
        X = X.drop(columns=['last_time'])
        print(f"   Features after removal: {len(X.columns)} features")
    
    # Check available years
    available_years = sorted(features_df_filtered['year'].unique())
    print(f"\nAvailable years in filtered data: {available_years}")
    
    # Check if we have data for multiple years (needed for temporal validation)
    if len(available_years) < 2:
        print(f"\nWARNING: Only {len(available_years)} year(s) of data available after filtering.")
        print("Skipping ML analysis as temporal validation requires at least 2 years of data.")
        print("This may indicate the race count filter is too restrictive.")
        return {}
    
    # Train models with temporal validation
    print("\nTraining models with temporal validation...")
    print("⚠️  DATA LEAKAGE PREVENTION:")
    print("   - Temporal split: Train on 2023, Test on 2024")
    print("   - R² calculated ONLY on test set (2024 data)")
    print("   - Models never see test data during training")
    print("   - Bootstrap CIs calculated on test set predictions only")
    
    results, X_test, y_test, test_metadata = train_models(X, y, features_df_filtered, use_temporal_split=True)
    
    # Verify no data leakage in R² calculations
    print("\n✅ DATA LEAKAGE VERIFICATION:")
    for name, result in results.items():
        if name not in ['_2025_generalization', '_2025_extended'] and 'y_test' in result and 'y_pred' in result:
            # Verify test set is from 2024 only
            test_years = test_metadata['year'].unique()
            print(f"   {name}: Test set years = {sorted(test_years)} (should be [2024])")
            print(f"   {name}: R² = {result['r2']:.4f} (calculated on {len(result['y_test'])} test samples only)")
    
    # Find best model
    valid_results = {k: v for k, v in results.items() 
                    if k not in ['_2025_generalization', '_2025_extended'] 
                    and 'y_test' in v and 'y_pred' in v}
    
    if len(valid_results) == 0:
        print("No valid model results to save.")
        return results
    
    best_model_name = max(valid_results.keys(), key=lambda k: valid_results[k]['r2'])
    best_model = valid_results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Test R² Score: {results[best_model_name]['r2']:.4f}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    # Use feature names from X (which has last_time removed to prevent leakage)
    feature_names = X.columns.tolist()
    feature_importance_df = analyze_feature_importance(best_model, feature_names, output_dir=output_dir)
    
    # Rename the feature importance file to rq2-specific name
    if os.path.exists(f'{output_dir}/raw_data_feature_importance.csv'):
        shutil.move(f'{output_dir}/raw_data_feature_importance.csv', 
                  f'{output_dir}/multi_season_feature_importance.csv')
    if os.path.exists(f'{output_dir}/raw_data_feature_importance.pdf'):
        shutil.move(f'{output_dir}/raw_data_feature_importance.pdf', 
                  f'{output_dir}/multi_season_feature_importance.pdf')
    print(f"Saved feature importance to {output_dir}/multi_season_feature_importance.csv")
    
    # Create custom RQ2 prediction plots (only comprehensive ones, not individual models)
    print("\nCreating RQ2 prediction plots...")
    create_rq2_prediction_plots(results, test_metadata, features_df_filtered, X, y, output_dir)
    
    # Save model performance
    results_summary = []
    for name, result in valid_results.items():
        results_summary.append({
            'Model': name,
            'R² Score': result['r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
        })
    
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(f'{output_dir}/multi_season_model_performance.csv', index=False)
    print(f"Saved model performance to {output_dir}/multi_season_model_performance.csv")
    
    # Handle extended training results if available
    if '_2025_extended' in results:
        extended_results = results['_2025_extended']
        extended_summary = []
        for name, result in extended_results.items():
            extended_summary.append({
                'Model': name,
                'R² Score': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
            })
        extended_df = pd.DataFrame(extended_summary)
        extended_df.to_csv(f'{output_dir}/multi_season_2023_to_2025_model_performance.csv', index=False)
        print(f"Saved extended model performance to {output_dir}/multi_season_2023_to_2025_model_performance.csv")
        
        # Feature importance for extended model (train on 2023+2024, test on 2025)
        # NOTE: For extended model, percentile calculation should use 2023+2024 data
        # However, we already calculated features with training_df=2023 only
        # This is conservative (prevents leakage) but may underestimate percentiles for 2025
        # For now, we'll use the existing features (calculated with 2023 only for percentiles)
        if 2025 in features_df_filtered['year'].values:
            train_mask_extended = features_df_filtered['year'].isin([2023, 2024])
            X_train_extended = X[train_mask_extended]
            y_train_extended = y[train_mask_extended]
            
            # Train extended model for feature importance (use best model type)
            # Create model of same type as best model
            # Always wrap in Pipeline for consistency with analyze_feature_importance
            if 'Ridge' in best_model_name:
                extended_model = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
            elif 'Lasso' in best_model_name:
                extended_model = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1))])
            elif 'Linear' in best_model_name:
                extended_model = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
            elif 'Random Forest' in best_model_name:
                # Wrap Random Forest in Pipeline for consistency
                extended_model = Pipeline([('model', RandomForestRegressor(n_estimators=100, random_state=42))])
            elif 'Gradient Boosting' in best_model_name:
                # Wrap Gradient Boosting in Pipeline for consistency
                extended_model = Pipeline([('model', GradientBoostingRegressor(n_estimators=100, random_state=42))])
            elif 'SVR' in best_model_name:
                extended_model = Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf', C=1.0, gamma='scale'))])
            else:
                # If best_model is already a Pipeline, use it; otherwise wrap it
                if hasattr(best_model, 'named_steps'):
                    extended_model = best_model
                else:
                    extended_model = Pipeline([('model', best_model)])
            
            extended_model.fit(X_train_extended, y_train_extended)
            # Use feature names from X (which has last_time removed)
            feature_names_extended = X.columns.tolist()
            extended_importance_df = analyze_feature_importance(extended_model, feature_names_extended, output_dir=output_dir)
            
            # Rename to extended-specific file
            if os.path.exists(f'{output_dir}/raw_data_feature_importance.csv'):
                shutil.move(f'{output_dir}/raw_data_feature_importance.csv', 
                          f'{output_dir}/multi_season_2023_to_2025_feature_importance.csv')
            if os.path.exists(f'{output_dir}/raw_data_feature_importance.pdf'):
                shutil.move(f'{output_dir}/raw_data_feature_importance.pdf', 
                          f'{output_dir}/multi_season_2023_to_2025_feature_importance.pdf')
            print(f"Saved extended feature importance to {output_dir}/multi_season_2023_to_2025_feature_importance.csv")
    
    if '_2025_generalization' in results:
        gen_results = results['_2025_generalization']
        gen_summary = []
        for name, result in gen_results.items():
            gen_summary.append({
                'Model': name,
                'R² Score': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
            })
        gen_df = pd.DataFrame(gen_summary)
        gen_df.to_csv(f'{output_dir}/multi_season_2024_to_2025_model_performance.csv', index=False)
        print(f"Saved generalization model performance to {output_dir}/multi_season_2024_to_2025_model_performance.csv")
        
        # Feature importance for generalization (train on 2023, test on 2025)
        # Use the same best model that was trained on 2023
        # Ensure it's a Pipeline for analyze_feature_importance
        if hasattr(best_model, 'named_steps'):
            gen_model = best_model
        else:
            # Wrap in Pipeline if not already
            gen_model = Pipeline([('model', best_model)])
        # Use feature names from X (which has last_time removed to prevent leakage)
        gen_importance_df = analyze_feature_importance(gen_model, feature_names, output_dir=output_dir)
        
        # Rename to generalization-specific file
        if os.path.exists(f'{output_dir}/raw_data_feature_importance.csv'):
            shutil.move(f'{output_dir}/raw_data_feature_importance.csv', 
                      f'{output_dir}/multi_season_2024_to_2025_feature_importance.csv')
        if os.path.exists(f'{output_dir}/raw_data_feature_importance.pdf'):
            shutil.move(f'{output_dir}/raw_data_feature_importance.pdf', 
                      f'{output_dir}/multi_season_2024_to_2025_feature_importance.pdf')
        print(f"Saved generalization feature importance to {output_dir}/multi_season_2024_to_2025_feature_importance.csv")
    
    return results

def main():
    """Standalone execution for testing."""
    print("="*60)
    print("RQ2 ML MODEL - STANDALONE EXECUTION")
    print("="*60)
    print("\nNote: This script is typically called from rq2.py")
    print("For standalone execution, you need to provide filtered data.")
    print("\nRun: python scripts/rq2.py (instead)")

if __name__ == "__main__":
    main()

