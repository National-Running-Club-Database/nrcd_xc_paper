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
import seaborn as sns
from scipy import stats
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

def create_rq2_prediction_plots(results, test_metadata, features_df_filtered, X, y, output_dir='output/rq2', best_model_name=None, best_model=None):
    """
    Create RQ2 prediction plots showing:
    1. Predicting 2025 based on 2023 training (generalization)
    2. Predicting 2025 based on 2023+2024 training (extended)
    All plots combined into a single PDF with 4 subplots (2 scenarios × 2 genders).
    
    DATA LEAKAGE PREVENTION:
    - Models trained only on training years (2023 or 2023+2024)
    - Predictions made only on 2025 test data
    - R² calculated only on 2025 test data (never on training data)
    
    Parameters:
    -----------
    best_model_name : str, optional
        Name of the best performing model
    best_model : model object, optional
        The best performing model (for feature importance extraction)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\nCreating RQ2 prediction plots (combined PDF)...")
    print("⚠️  DATA LEAKAGE PREVENTION:")
    print("   - All R² values calculated ONLY on test data (2025)")
    print("   - Models never evaluated on training data")
    print("   - Bootstrap CIs calculated on test set only")
    
    os.makedirs(output_dir, exist_ok=True)
    
    gender_labels = {'M': 'Men', 'F': 'Women'}
    
    # Determine model name from results if not provided
    if best_model_name is None:
        valid_results = {k: v for k, v in results.items() 
                        if k not in ['_2025_generalization', '_2025_extended'] 
                        and 'y_test' in v and 'y_pred' in v}
        if len(valid_results) > 0:
            best_model_name = max(valid_results.keys(), key=lambda k: valid_results[k]['r2'])
    
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.reshape(2, 2)
    
    # Main title with model name
    model_title = f'Multi-Season Predictions: Predicting 2025 Performance'
    if best_model_name:
        model_title += f'\nModel: {best_model_name}'
    fig.suptitle(model_title, fontsize=16, fontweight='bold', y=0.995)
    
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

def analyze_rq2_gender_specific_feature_importance(X, y, features_df, output_dir='output/rq2'):
    """
    Analyze feature importance separately for men and women in RQ2 (like RQ1 does).
    
    This function trains separate models for each gender to determine if
    different factors are important for predicting improvement in men vs women.
    
    Parameters:
    -----------
    X : DataFrame or array
        Feature matrix
    y : Series or array
        Target variable
    features_df : DataFrame
        DataFrame with metadata (gender, year, etc.)
    output_dir : str
        Output directory for results
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    
    print("\n" + "="*60)
    print("RQ2 GENDER-SPECIFIC FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use temporal split: train on 2023, test on 2024
    features_df_aligned = features_df.reset_index(drop=True)
    X_aligned = X.reset_index(drop=True) if hasattr(X, 'reset_index') else X
    y_aligned = y.reset_index(drop=True) if hasattr(y, 'reset_index') else y
    
    train_mask = features_df_aligned['year'] == 2023
    test_mask = features_df_aligned['year'] == 2024
    
    # Use positional indexing to ensure alignment
    if hasattr(X_aligned, 'iloc'):
        X_train_all = X_aligned.iloc[train_mask.values]
        X_test_all = X_aligned.iloc[test_mask.values]
    else:
        X_train_all = X_aligned[train_mask.values]
        X_test_all = X_aligned[test_mask.values]
    
    if hasattr(y_aligned, 'iloc'):
        y_train_all = y_aligned.iloc[train_mask.values]
        y_test_all = y_aligned.iloc[test_mask.values]
    else:
        y_train_all = y_aligned[train_mask.values]
        y_test_all = y_aligned[test_mask.values]
    
    features_train = features_df_aligned[train_mask].reset_index(drop=True)
    features_test = features_df_aligned[test_mask].reset_index(drop=True)
    
    # Get feature names
    if hasattr(X, 'columns'):
        feature_names_list = X.columns.tolist()
    else:
        feature_names_list = [f'feature_{i}' for i in range(X.shape[1])]
    
    gender_importance_comparison = []
    
    for gender in ['M', 'F']:
        gender_label = 'Men' if gender == 'M' else 'Women'
        print(f"\nAnalyzing {gender_label}...")
        
        # Filter training and test data by gender
        train_gender_mask = features_train['gender'] == gender
        test_gender_mask = features_test['gender'] == gender
        
        if train_gender_mask.sum() < 50:
            print(f"  Insufficient data for {gender_label} (n={train_gender_mask.sum()})")
            continue
        
        X_train_gender = X_train_all.iloc[train_gender_mask.values] if hasattr(X_train_all, 'iloc') else X_train_all[train_gender_mask.values]
        y_train_gender = y_train_all.iloc[train_gender_mask.values] if hasattr(y_train_all, 'iloc') else y_train_all[train_gender_mask.values]
        X_test_gender = X_test_all.iloc[test_gender_mask.values] if hasattr(X_test_all, 'iloc') else X_test_all[test_gender_mask.values]
        y_test_gender = y_test_all.iloc[test_gender_mask.values] if hasattr(y_test_all, 'iloc') else y_test_all[test_gender_mask.values]
        
        print(f"  Training samples: {len(X_train_gender)}, Test samples: {len(X_test_gender)}")
        
        # Use Random Forest for both men and women (consistent model type for comparable feature importance)
        best_model_name = 'Random Forest'
        best_model = RandomForestRegressor(n_estimators=100, random_state=42)
        best_model.fit(X_train_gender, y_train_gender)
        y_pred_test = best_model.predict(X_test_gender)
        best_r2 = r2_score(y_test_gender, y_pred_test)
        
        print(f"  Model for {gender_label}: {best_model_name} (R² = {best_r2:.4f})")
        
        # Get feature importance
        importances = np.zeros(len(feature_names_list))
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'named_steps'):
            model_step = best_model.named_steps.get('model', None)
            if model_step is not None and hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
        elif hasattr(best_model, 'coef_'):
            coef = best_model.coef_ if hasattr(best_model, 'coef_') else best_model.named_steps['model'].coef_
            importances = np.abs(coef)
            if importances.sum() > 0:
                importances = importances / importances.sum()
        
        # Store importance for comparison
        for i, (feature, importance) in enumerate(zip(feature_names_list, importances)):
            gender_importance_comparison.append({
                'Gender': gender_label,
                'Feature': feature,
                'Importance': importance,
                'Best_Model': best_model_name,
                'R²': best_r2,
                'Rank': None
            })
    
    # Create comparison DataFrame
    if len(gender_importance_comparison) == 0:
        print("No gender-specific feature importance data collected.")
        return None, None
    
    comparison_df = pd.DataFrame(gender_importance_comparison)
    
    # Calculate ranks within each gender
    for gender in comparison_df['Gender'].unique():
        mask = comparison_df['Gender'] == gender
        comparison_df.loc[mask, 'Rank'] = comparison_df.loc[mask, 'Importance'].rank(ascending=False, method='min')
    
    # Ensure Rank column is numeric
    comparison_df['Rank'] = pd.to_numeric(comparison_df['Rank'], errors='coerce')
    
    # Filter to top 15 features for each gender and sort by rank
    top_features_list = []
    for gender in comparison_df['Gender'].unique():
        gender_df = comparison_df[comparison_df['Gender'] == gender].copy()
        gender_df['Rank'] = pd.to_numeric(gender_df['Rank'], errors='coerce')
        top_15 = gender_df.nsmallest(15, 'Rank')
        top_15 = top_15.sort_values('Rank', ascending=True)
        top_features_list.append(top_15)
    
    # Combine and sort
    comparison_df_sorted = pd.concat(top_features_list, ignore_index=True)
    comparison_df_sorted = comparison_df_sorted.sort_values(['Gender', 'Rank'], ascending=[True, True])
    
    # Pivot for easier comparison
    pivot_df = comparison_df.pivot(index='Feature', columns='Gender', values='Importance').fillna(0)
    
    # Calculate difference in importance
    if 'Men' in pivot_df.columns and 'Women' in pivot_df.columns:
        pivot_df['Difference'] = pivot_df['Women'] - pivot_df['Men']
        pivot_df['Abs_Difference'] = pivot_df['Difference'].abs()
    
    # Test statistical significance of feature importance differences using bootstrap
    print("\nTesting statistical significance of feature importance differences...")
    significance_results = {}
    
    # Bootstrap feature importance for men and women
    n_bootstrap = 100
    men_mask = features_train['gender'] == 'M'
    women_mask = features_train['gender'] == 'F'
    
    X_train_men = X_train_all.iloc[men_mask.values] if hasattr(X_train_all, 'iloc') else X_train_all[men_mask.values]
    y_train_men = y_train_all.iloc[men_mask.values] if hasattr(y_train_all, 'iloc') else y_train_all[men_mask.values]
    X_train_women = X_train_all.iloc[women_mask.values] if hasattr(X_train_all, 'iloc') else X_train_all[women_mask.values]
    y_train_women = y_train_all.iloc[women_mask.values] if hasattr(y_train_all, 'iloc') else y_train_all[women_mask.values]
    
    if len(X_train_men) >= 50 and len(X_train_women) >= 50:
        print(f"  Running bootstrap test (n={n_bootstrap})...")
        
        # Bootstrap for men
        men_importances_boot = {f: [] for f in feature_names_list}
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X_train_men), len(X_train_men), replace=True)
            if hasattr(X_train_men, 'iloc'):
                X_boot = X_train_men.iloc[indices]
                y_boot = y_train_men.iloc[indices] if hasattr(y_train_men, 'iloc') else y_train_men[indices]
            else:
                X_boot = X_train_men[indices]
                y_boot = y_train_men[indices]
            
            model = RandomForestRegressor(n_estimators=100, random_state=None)
            model.fit(X_boot, y_boot)
            for i, feature in enumerate(feature_names_list):
                men_importances_boot[feature].append(model.feature_importances_[i])
        
        # Bootstrap for women
        women_importances_boot = {f: [] for f in feature_names_list}
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X_train_women), len(X_train_women), replace=True)
            if hasattr(X_train_women, 'iloc'):
                X_boot = X_train_women.iloc[indices]
                y_boot = y_train_women.iloc[indices] if hasattr(y_train_women, 'iloc') else y_train_women[indices]
            else:
                X_boot = X_train_women[indices]
                y_boot = y_train_women[indices]
            
            model = RandomForestRegressor(n_estimators=100, random_state=None)
            model.fit(X_boot, y_boot)
            for i, feature in enumerate(feature_names_list):
                women_importances_boot[feature].append(model.feature_importances_[i])
        
        # Test significance for each feature
        for feature in feature_names_list:
            men_vals = np.array(men_importances_boot[feature])
            women_vals = np.array(women_importances_boot[feature])
            t_stat, p_value = stats.ttest_ind(men_vals, women_vals)
            significance_results[feature] = {'p_value': p_value}
        
        # Apply Bonferroni correction
        n_tests = len(significance_results)
        alpha = 0.05
        bonferroni_alpha = alpha / n_tests
        
        # Add to pivot_df
        pivot_df['P_Value'] = pivot_df.index.map(lambda f: significance_results.get(f, {}).get('p_value', 1.0))
        pivot_df['P_Value_Bonferroni'] = pivot_df['P_Value'] * n_tests
        pivot_df['Significant_Bonferroni'] = pivot_df['P_Value'] < bonferroni_alpha
        pivot_df['Significance_Bonferroni'] = pivot_df['P_Value'].apply(
            lambda p: '***' if p < bonferroni_alpha else ''
        )
    else:
        # If no significance testing, add default values
        pivot_df['P_Value'] = np.nan
        pivot_df['Significant_Bonferroni'] = False
    
    # Get top 15 features by average importance
    pivot_df['Avg_Importance'] = (pivot_df['Men'] + pivot_df['Women']) / 2
    top_features_df = pivot_df.nlargest(15, 'Avg_Importance').sort_values('Avg_Importance', ascending=False)
    top_features = top_features_df.index.tolist()
    # Reverse so most important is at top (for horizontal bar chart, index 0 is at bottom)
    top_features = top_features[::-1]
    
    # Create visualization (similar to RQ1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Side-by-side comparison
    x = np.arange(len(top_features))
    width = 0.35
    
    men_vals = [pivot_df.loc[f, 'Men'] for f in top_features]
    women_vals = [pivot_df.loc[f, 'Women'] for f in top_features]
    
    # Color bars based on Bonferroni significance
    men_colors = []
    women_colors = []
    for f in top_features:
        is_sig = pivot_df.loc[f, 'Significant_Bonferroni'] if 'Significant_Bonferroni' in pivot_df.columns else False
        if is_sig:
            men_colors.append('#2E86AB')  # Blue for significant
            women_colors.append('#A23B72')  # Purple for significant
        else:
            men_colors.append('#6C757D')  # Gray for non-significant
            women_colors.append('#6C757D')
    
    bars1 = ax1.barh(x - width/2, men_vals, width, label='Men', alpha=0.8, color=men_colors)
    bars2 = ax1.barh(x + width/2, women_vals, width, label='Women', alpha=0.8, color=women_colors)
    
    max_importance = max(max(men_vals), max(women_vals))
    ax1.set_xlim(0, max_importance * 1.15)
    
    # Add significance markers (use Bonferroni significance markers)
    if 'Significance_Bonferroni' in pivot_df.columns:
        for i, f in enumerate(top_features):
            sig_level = pivot_df.loc[f, 'Significance_Bonferroni']
            if sig_level:
                max_val = max(men_vals[i], women_vals[i])
                x_pos = max_val + (max_importance * 0.02)
                ax1.text(x_pos, i, sig_level, 
                        fontsize=12, fontweight='bold', va='center', ha='left',
                        color='red' if pivot_df.loc[f, 'Difference'] > 0 else 'blue')
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(top_features)
    ax1.set_xlabel('Feature Importance', fontsize=12)
    # Count significant features (across ALL features, not just top 15 displayed)
    n_sig_bonf_all = sum(pivot_df['Significant_Bonferroni']) if 'Significant_Bonferroni' in pivot_df.columns else 0
    # Also count within displayed top 15
    n_sig_bonf_top15 = sum(pivot_df.loc[top_features, 'Significant_Bonferroni']) if 'Significant_Bonferroni' in pivot_df.columns else 0
    ax1.set_title(f'Random Forest Feature Importance Comparison: Men vs Women\n'
                 f'(Colored = Bonferroni Significant Between Genders, ***p<Bonferroni threshold)\n'
                 f'Bonferroni Significant: {n_sig_bonf_all} total ({n_sig_bonf_top15} shown) (RQ2)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Difference plot with significance
    differences = [pivot_df.loc[f, 'Difference'] for f in top_features]
    sig_levels = [pivot_df.loc[f, 'Significance_Bonferroni'] if 'Significance_Bonferroni' in pivot_df.columns else '' 
                 for f in top_features]
    
    # Color based on Bonferroni significance (significant between genders)
    diff_colors = []
    for i, f in enumerate(top_features):
        is_sig = pivot_df.loc[f, 'Significant_Bonferroni'] if 'Significant_Bonferroni' in pivot_df.columns else False
        if is_sig:
            diff_colors.append('#E63946' if differences[i] > 0 else '#06A77D')
        else:
            diff_colors.append('#6C757D')
    
    bars3 = ax2.barh(x, differences, alpha=0.7, color=diff_colors)
    
    max_diff = max(abs(min(differences)), abs(max(differences)))
    ax2.set_xlim(min(differences) * 1.15, max(differences) * 1.15)
    
    # Add significance markers
    for i, (diff, sig) in enumerate(zip(differences, sig_levels)):
        if sig:
            if diff > 0:
                x_pos = diff + (max_diff * 0.02)
                ha = 'left'
            else:
                x_pos = diff - (max_diff * 0.02)
                ha = 'right'
            ax2.text(x_pos, i, sig, fontsize=12, fontweight='bold', 
                    va='center', ha=ha,
                    color='red' if diff > 0 else 'blue')
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(top_features)
    ax2.set_xlabel('Difference (Women - Men)', fontsize=12)
    ax2.set_title(f'Random Forest Feature Importance Differences\n'
                 f'(Red/Green = Bonferroni Significant Between Genders, Gray = Not Significant)\n'
                 f'Bonferroni Significant: {n_sig_bonf_all} total ({n_sig_bonf_top15} shown) (RQ2)', 
                 fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = f'{output_dir}/multi_season_gender_feature_importance_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGender-specific feature importance saved to:")
    print(f"  - {output_dir}/multi_season_gender_feature_importance_comparison.pdf")
    
    # Create comparison CSV with requested format: Feature, Rank, Avg_Importance, Men_Importance, Women_Importance, P_Value, Bonferroni_Significant
    if 'Men' in pivot_df.columns and 'Women' in pivot_df.columns:
        # Calculate average importance if not already done
        if 'Avg_Importance' not in pivot_df.columns:
            pivot_df['Avg_Importance'] = (pivot_df['Men'] + pivot_df['Women']) / 2
        
        # Create comparison DataFrame with requested columns
        comparison_output = pd.DataFrame({
            'Feature': pivot_df.index,
            'Men_Importance': pivot_df['Men'],
            'Women_Importance': pivot_df['Women'],
            'Avg_Importance': pivot_df['Avg_Importance'],
            'P_Value': pivot_df.get('P_Value', np.nan),
            'Bonferroni_Significant': pivot_df.get('Significant_Bonferroni', False)
        })
        
        # Calculate rank based on average importance (rank 1 = most important)
        comparison_output['Rank'] = comparison_output['Avg_Importance'].rank(ascending=False, method='min')
        comparison_output['Rank'] = pd.to_numeric(comparison_output['Rank'], errors='coerce')
        
        # Sort by rank (most important first)
        comparison_output = comparison_output.sort_values('Rank', ascending=True)
        
        # Reorder columns as requested
        comparison_output = comparison_output[['Feature', 'Rank', 'Avg_Importance', 'Men_Importance', 'Women_Importance', 'P_Value', 'Bonferroni_Significant']]
        
        # Save the new format
        comparison_output.to_csv(f'{output_dir}/multi_season_gender_feature_importance_comparison.csv', index=False)
    
    # Also save pivot for reference
    pivot_df.to_csv(f'{output_dir}/multi_season_gender_feature_importance_pivot.csv')
    
    return comparison_df_sorted, pivot_df

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
    create_rq2_prediction_plots(results, test_metadata, features_df_filtered, X, y, output_dir, 
                               best_model_name=best_model_name, best_model=best_model)
    
    # Create gender-specific feature importance analysis (separate models for men and women, like RQ1)
    print("\nCreating RQ2 gender-specific feature importance analysis...")
    analyze_rq2_gender_specific_feature_importance(X, y, features_df_filtered, output_dir)
    
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

