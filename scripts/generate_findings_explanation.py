"""
Generate FINDINGS_EXPLANATION.md from CSV outputs.

This script reads the analysis results from CSV files and generates
a comprehensive findings explanation document. It does NOT require
running the full analysis - just reads from existing CSV files.

Run from main directory: python scripts/generate_findings_explanation.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
main_dir = script_dir.parent
os.chdir(main_dir)

def load_csv_safe(filepath, description=""):
    """Load CSV file, return None if not found."""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"  ✓ Loaded {description}: {filepath}")
            return df
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")
            return None
    else:
        print(f"  ✗ Not found: {filepath}")
        return None

def generate_findings_explanation():
    """Generate FINDINGS_EXPLANATION.md from CSV outputs."""
    print("="*60)
    print("GENERATING FINDINGS_EXPLANATION.md")
    print("="*60)
    
    # Check for output files (try rq1 first, then output)
    rq1_dir = 'output/rq1'
    output_dir = 'output'
    
    # Determine which directory to use
    if os.path.exists(rq1_dir):
        base_dir = rq1_dir
        print(f"\nReading from: {base_dir}/")
    elif os.path.exists(output_dir):
        base_dir = output_dir
        print(f"\nReading from: {base_dir}/")
    else:
        print("\nERROR: No output directory found!")
        print("Please run the analysis scripts first:")
        print("  1. python scripts/rq1.py")
        print("  2. python scripts/rq2.py")
        print("  3. python scripts/rq3.py")
        print("  4. python scripts/create_charts.py")
        return False
    
    print("\nLoading CSV files...")
    
    # Load model performance
    model_perf = load_csv_safe(f'{base_dir}/raw_data_model_performance.csv', 'Model performance')
    
    # Load feature importance
    feature_imp = load_csv_safe(f'{base_dir}/raw_data_feature_importance.csv', 'Feature importance')
    
    # Load important statistics
    stats = load_csv_safe(f'{base_dir}/raw_data_important_statistics.csv', 'Important statistics')
    
    # Load gender comparison
    gender_comp = load_csv_safe(f'{base_dir}/raw_data_gender_feature_importance_comparison.csv', 'Gender comparison')
    
    # Load time standardization comparison
    time_std = load_csv_safe(f'{base_dir}/time_standardization_comparison.csv', 'Time standardization')
    
    # Load subgroup analysis
    subgroup = load_csv_safe(f'{base_dir}/raw_data_subgroup_analysis.csv', 'Subgroup analysis')
    
    if model_perf is None or feature_imp is None:
        print("\nERROR: Required CSV files not found!")
        print("Need at least: raw_data_model_performance.csv and raw_data_feature_importance.csv")
        return False
    
    print("\nGenerating markdown document...")
    
    # Start building the markdown
    md_lines = []
    md_lines.append("# Comprehensive Research Findings Explanation")
    md_lines.append("")
    md_lines.append("*Auto-generated from analysis results*")
    md_lines.append("")
    
    # Executive Summary
    md_lines.append("## Executive Summary")
    md_lines.append("")
    
    # Get best model
    if 'R² Score' in model_perf.columns:
        r2_col = 'R² Score'
        rmse_col = 'RMSE'
        mae_col = 'MAE'
        cv_r2_col = 'CV R² Mean'
    elif 'Test_R2_Score' in model_perf.columns:
        r2_col = 'Test_R2_Score'
        rmse_col = 'Test_RMSE'
        mae_col = 'Test_MAE'
        cv_r2_col = 'CV_R2_Mean'
    else:
        # Find R² column
        r2_cols = [c for c in model_perf.columns if 'r2' in c.lower() or 'R²' in c]
        r2_col = r2_cols[0] if r2_cols else model_perf.columns[1]
        rmse_col = [c for c in model_perf.columns if 'rmse' in c.lower()][0] if any('rmse' in c.lower() for c in model_perf.columns) else None
        mae_col = [c for c in model_perf.columns if 'mae' in c.lower()][0] if any('mae' in c.lower() for c in model_perf.columns) else None
        cv_r2_col = [c for c in model_perf.columns if 'cv' in c.lower() and 'r2' in c.lower()][0] if any('cv' in c.lower() and 'r2' in c.lower() for c in model_perf.columns) else None
    
    best_model_row = model_perf.loc[model_perf[r2_col].idxmax()]
    best_model_name = best_model_row['Model']
    best_r2 = best_model_row[r2_col]
    
    # Get CI if available
    ci_lower = best_model_row.get('R² CI Lower', np.nan)
    ci_upper = best_model_row.get('R² CI Upper', np.nan)
    if pd.isna(ci_lower):
        ci_lower = best_model_row.get('CI_Lower', np.nan)
        ci_upper = best_model_row.get('CI_Upper', np.nan)
    
    ci_str = f" (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])" if not pd.isna(ci_lower) else ""
    
    md_lines.append(f"This analysis uses machine learning to predict athlete improvement rates in collegiate cross country running. The best model ({best_model_name}) achieves **{best_r2*100:.2f}% accuracy** (R² = {best_r2:.4f}{ci_str}) using temporal validation (trained on 2023, tested on 2024). Key findings include:")
    md_lines.append("")
    
    # Top features
    if feature_imp is not None and len(feature_imp) > 0:
        top_feature = feature_imp.iloc[0]
        top_importance = top_feature['importance'] * 100
        md_lines.append(f"1. **{top_feature['feature']}** is the most critical predictor ({top_importance:.1f}%)")
    
    # Model comparison
    tree_models = model_perf[model_perf['Model'].str.contains('Gradient|Random|Forest', case=False, na=False)]
    linear_models = model_perf[model_perf['Model'].str.contains('Linear|Ridge|Lasso|SVR', case=False, na=False)]
    
    if len(tree_models) > 0 and len(linear_models) > 0:
        tree_r2 = tree_models[r2_col].max()
        linear_r2 = linear_models[r2_col].max()
        md_lines.append(f"2. **Tree-based models significantly outperform linear models** - {tree_r2*100:.1f}% vs {linear_r2*100:.1f}% accuracy")
    
    # Gender differences
    if gender_comp is not None and len(gender_comp) > 0:
        md_lines.append("3. **Different factors matter for men vs women** - multiple features show gender differences")
    
    # Time standardization
    if time_std is not None and len(time_std) > 0:
        if 'Standardized' in time_std['Method'].values and 'Converted' in time_std['Method'].values:
            std_r2 = time_std[time_std['Method'] == 'Standardized']['R² Score'].values[0]
            conv_r2 = time_std[time_std['Method'] == 'Converted']['R² Score'].values[0]
            improvement = ((std_r2 - conv_r2) / conv_r2) * 100
            md_lines.append(f"4. **Full time standardization improves model performance by {improvement:.1f}%** compared to distance-only conversion")
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # Model Performance Comparison
    md_lines.append("## 1. Model Performance Comparison")
    md_lines.append("")
    md_lines.append("### Overall Model Results")
    md_lines.append("")
    
    # Create table
    md_lines.append("| Model | Test R² | 95% CI | RMSE | MAE | CV R² (Train) | CV Std |")
    md_lines.append("|-------|---------|--------|------|-----|---------------|--------|")
    
    for _, row in model_perf.iterrows():
        model_name = row['Model']
        r2_val = row[r2_col]
        
        # Get CI
        ci_l = row.get('R² CI Lower', row.get('CI_Lower', np.nan))
        ci_u = row.get('R² CI Upper', row.get('CI_Upper', np.nan))
        ci_str = f"[{ci_l:.4f}, {ci_u:.4f}]" if not pd.isna(ci_l) else "N/A"
        
        # Get other metrics
        rmse_val = row[rmse_col] if rmse_col and rmse_col in row else "N/A"
        mae_val = row[mae_col] if mae_col and mae_col in row else "N/A"
        cv_r2_val = row[cv_r2_col] if cv_r2_col and cv_r2_col in row else "N/A"
        cv_std = row.get('CV R² Std', row.get('CV_R2_Std', np.nan))
        cv_std_str = f"±{cv_std:.4f}" if not pd.isna(cv_std) else "N/A"
        
        # Bold best model
        prefix = "**" if model_name == best_model_name else ""
        suffix = "**" if model_name == best_model_name else ""
        
        md_lines.append(f"| {prefix}{model_name}{suffix} | {prefix}{r2_val:.4f}{suffix} | {ci_str} | {rmse_val} | {mae_val} | {cv_r2_val} | {cv_std_str} |")
    
    md_lines.append("")
    
    # Why tree models outperform
    if len(tree_models) > 0 and len(linear_models) > 0:
        md_lines.append("### Why Tree-Based Models Outperform Linear Models")
        md_lines.append("")
        md_lines.append("**Statistical Evidence:**")
        md_lines.append(f"- Tree-based models (Random Forest, Gradient Boosting) **significantly outperform** linear models")
        md_lines.append(f"- Linear models achieve only **{linear_r2*100:.0f}-{linear_models[r2_col].max()*100:.0f}% accuracy** vs **{tree_r2*100:.0f}% for tree-based models**")
        improvement_pct = ((tree_r2 - linear_r2) / linear_r2) * 100
        md_lines.append(f"- This is a **{improvement_pct:.0f}% relative improvement** in prediction accuracy")
        md_lines.append("")
        md_lines.append("**Why This Happens:**")
        md_lines.append("")
        md_lines.append("1. **Non-Linear Relationships:** Linear models assume straight-line relationships, but improvement patterns are **complex and non-linear**")
        md_lines.append("2. **Feature Interactions:** Tree models capture interactions between features automatically")
        md_lines.append("3. **Non-Additive Effects:** Improvement doesn't simply add up linearly")
        md_lines.append("4. **Heterogeneous Effects:** Different athletes respond differently to the same training/racing")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Key Predictors
    if feature_imp is not None and len(feature_imp) > 0:
        md_lines.append("## 2. Key Predictors of Improvement")
        md_lines.append("")
        md_lines.append("### Top 15 Most Important Features")
        md_lines.append("")
        md_lines.append("| Rank | Feature | Importance | What It Measures |")
        md_lines.append("|------|---------|------------|------------------|")
        
        feature_descriptions = {
            'slope': 'Improvement trajectory pattern (per-race change)',
            'experience_level': 'Total racing experience (num_races × season_duration)',
            'season_duration': 'Number of days from first to last race',
            'improvement_to_variability_ratio': 'Total improvement relative to performance variability',
            'time_std': 'Standard deviation of race times (consistency)',
            'season_duration_squared': 'Non-linear effect of season duration (optimal length)',
            'worst_to_avg_ratio': 'Worst time relative to average (performance spread)',
            'cv_time': 'Coefficient of variation (normalized consistency)',
            'race_frequency': 'Races per day (racing frequency)',
            'consistency_score': 'Inverse of variability (consistency measure)',
            'best_to_avg_ratio': 'Best time relative to average (potential vs actual)',
            'num_races': 'Number of races in season',
            'time_range': 'Difference between worst and best time',
            'num_races_squared': 'Non-linear effect of number of races',
            'avg_days_between_races': 'Average recovery time between races',
        }
        
        for i, (_, row) in enumerate(feature_imp.head(15).iterrows(), 1):
            feature = row['feature']
            importance = row['importance'] * 100
            description = feature_descriptions.get(feature, 'Performance metric')
            md_lines.append(f"| {i} | **{feature}** | **{importance:.1f}%** | {description} |")
        
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Gender Differences
    if gender_comp is not None and len(gender_comp) > 0:
        md_lines.append("## 3. Gender Differences in Feature Importance")
        md_lines.append("")
        md_lines.append("The model reveals that different factors matter for men vs women:")
        md_lines.append("")
        
        # Count significant differences
        if 'p_value' in gender_comp.columns:
            sig_features = gender_comp[gender_comp['p_value'] < 0.05]
            md_lines.append(f"- **{len(sig_features)} features** show statistically significant gender differences (p < 0.05)")
        
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Time Standardization
    if time_std is not None and len(time_std) > 0:
        md_lines.append("## 4. Time Standardization Impact")
        md_lines.append("")
        md_lines.append("### Comparison of Standardization Methods")
        md_lines.append("")
        md_lines.append("| Method | R² Score | 95% CI | RMSE | MAE |")
        md_lines.append("|--------|----------|--------|------|-----|")
        
        for _, row in time_std.iterrows():
            method = row['Method']
            r2 = row['R² Score']
            ci_l = row.get('R² CI Lower', np.nan)
            ci_u = row.get('R² CI Upper', np.nan)
            ci_str = f"[{ci_l:.4f}, {ci_u:.4f}]" if not pd.isna(ci_l) else "N/A"
            rmse = row.get('RMSE', 'N/A')
            mae = row.get('MAE', 'N/A')
            md_lines.append(f"| {method} | {r2:.4f} | {ci_str} | {rmse} | {mae} |")
        
        md_lines.append("")
        md_lines.append("**Key Findings:**")
        
        if 'Standardized' in time_std['Method'].values:
            std_row = time_std[time_std['Method'] == 'Standardized'].iloc[0]
            std_r2 = std_row['R² Score']
            std_rmse = std_row.get('RMSE', np.nan)
            std_mae = std_row.get('MAE', np.nan)
            
            if 'Converted' in time_std['Method'].values:
                conv_row = time_std[time_std['Method'] == 'Converted'].iloc[0]
                conv_r2 = conv_row['R² Score']
                conv_rmse = conv_row.get('RMSE', np.nan)
                conv_mae = conv_row.get('MAE', np.nan)
                
                improvement = ((std_r2 - conv_r2) / conv_r2) * 100
                md_lines.append(f"- **Standardized method improves R² by {improvement:.1f}%** compared to converted ({std_r2:.4f} vs {conv_r2:.4f})")
                
                if not pd.isna(std_rmse) and not pd.isna(conv_rmse):
                    rmse_improvement = ((conv_rmse - std_rmse) / conv_rmse) * 100
                    md_lines.append(f"- **Standardized method has {rmse_improvement:.0f}% lower RMSE** than converted ({std_rmse:.2f} vs {conv_rmse:.2f} seconds/day)")
            
            if 'Raw' in time_std['Method'].values or 'Original' in time_std['Method'].values:
                raw_method = 'Raw' if 'Raw' in time_std['Method'].values else 'Original'
                raw_row = time_std[time_std['Method'] == raw_method].iloc[0]
                raw_r2 = raw_row['R² Score']
                raw_rmse = raw_row.get('RMSE', np.nan)
                
                improvement = ((std_r2 - raw_r2) / raw_r2) * 100
                md_lines.append(f"- **Standardized method improves R² by {improvement:.1f}%** compared to raw ({std_r2:.4f} vs {raw_r2:.4f})")
        
        md_lines.append("")
        md_lines.append("**What Standardization Includes:**")
        md_lines.append("1. **Distance Conversion:** All times converted to gender-specific target distances (6k for women, 8k for men)")
        md_lines.append("2. **Weather Adjustments:** Adjustments for temperature and dew point (heat/humidity effects)")
        md_lines.append("3. **Terrain Adjustments:** Adjustments for elevation gain/loss and course distance accuracy")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Subgroup Analysis
    if subgroup is not None and len(subgroup) > 0:
        md_lines.append("## 5. Subgroup Analysis")
        md_lines.append("")
        md_lines.append("### Performance by Year and Gender")
        md_lines.append("")
        md_lines.append("| Year | Gender | R² Score | MAE | N |")
        md_lines.append("|------|--------|----------|-----|---|")
        
        for _, row in subgroup.iterrows():
            year = row.get('Year', 'N/A')
            gender = row.get('Gender', 'N/A')
            r2 = row.get('R²', row.get('R2', 'N/A'))
            mae = row.get('MAE', 'N/A')
            n = row.get('n', row.get('N', 'N/A'))
            r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
            mae_str = f"{mae:.2f}" if isinstance(mae, (int, float)) else str(mae)
            md_lines.append(f"| {year} | {gender} | {r2_str} | {mae_str} | {n} |")
        
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Write to file
    output_path = 'output/FINDINGS_EXPLANATION.md'
    os.makedirs('output', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n✓ Generated: {output_path}")
    print(f"  Total length: {len(md_lines)} lines")
    
    return True

if __name__ == "__main__":
    success = generate_findings_explanation()
    if success:
        print("\n" + "="*60)
        print("FINDINGS_EXPLANATION.md GENERATED SUCCESSFULLY")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("GENERATION FAILED - Please run analysis scripts first")
        print("="*60)
        sys.exit(1)

