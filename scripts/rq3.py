"""
RQ3: Gender differences in participation and performance

This script runs all analyses related to RQ3:
- Gender race participation analysis
- Gender-specific feature importance (from ML model)

All outputs are saved to output/rq3/

Run from main directory: python scripts/rq3.py
"""

import os
import sys
import shutil

# Setup paths for imports (works from main directory or scripts directory)
from _setup_paths import setup_paths
script_dir = setup_paths()

# Set output directory for RQ3 (relative to main directory)
rq3_output = 'output/rq3'
os.makedirs(rq3_output, exist_ok=True)

def main():
    """Run all RQ3 analyses."""
    print("="*60)
    print("RQ3: GENDER DIFFERENCES IN PARTICIPATION AND PERFORMANCE")
    print("="*60)
    
    print("\n1. Gender Race Participation Analysis...")
    from gender_race_participation_test import main as gender_main
    import gender_race_participation_test
    # Modify output directory
    original_output = gender_race_participation_test.output_dir
    gender_race_participation_test.output_dir = os.path.join(rq3_output, 'gender_analysis')
    os.makedirs(gender_race_participation_test.output_dir, exist_ok=True)
    gender_main()
    gender_race_participation_test.output_dir = original_output
    
    print("\n2. Gender-Specific Feature Importance & Model Fairness...")
    print("(Copying from RQ1 ML model outputs)")
    # Copy gender-specific files from rq1 if they exist
    rq1_output = 'output/rq1'
    gender_files = [
        'raw_data_gender_feature_importance_comparison.csv',
        'raw_data_gender_feature_importance_pivot.csv',
        'raw_data_subgroup_analysis.csv',  # Contains R² by gender (fairness analysis)
        'raw_data_important_statistics.csv'  # Contains detailed gender statistics
    ]
    
    gender_pdfs = [f for f in os.listdir(rq1_output) if f.startswith('raw_data_gender_') and f.endswith('.pdf')] if os.path.exists(rq1_output) else []
    
    for filename in gender_files + gender_pdfs:
        src = os.path.join(rq1_output, filename)
        if os.path.exists(src):
            dst = os.path.join(rq3_output, filename)
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")
        else:
            print(f"  Note: {filename} not found in {rq1_output}/ (run rq1.py first)")
    
    # Create a gender fairness summary if subgroup analysis exists
    subgroup_file = os.path.join(rq1_output, 'raw_data_subgroup_analysis.csv')
    if os.path.exists(subgroup_file):
        print("\n3. Creating Gender Fairness Summary...")
        import pandas as pd
        subgroup_df = pd.read_csv(subgroup_file)
        
        # Extract gender R² scores
        men_r2 = subgroup_df[subgroup_df['Gender'] == 'Men']['R²'].values[0] if len(subgroup_df[subgroup_df['Gender'] == 'Men']) > 0 else None
        women_r2 = subgroup_df[subgroup_df['Gender'] == 'Women']['R²'].values[0] if len(subgroup_df[subgroup_df['Gender'] == 'Women']) > 0 else None
        
        if men_r2 is not None and women_r2 is not None:
            fairness_summary = {
                'Metric': ['R² Score', 'R² Score', 'R² Difference', 'Relative Performance'],
                'Gender': ['Men', 'Women', 'Difference', 'Women/Men Ratio'],
                'Value': [men_r2, women_r2, men_r2 - women_r2, women_r2 / men_r2],
                'Interpretation': [
                    f'Model explains {men_r2*100:.1f}% of variance for men',
                    f'Model explains {women_r2*100:.1f}% of variance for women',
                    f'Men R² is {men_r2 - women_r2:.4f} higher',
                    f'Women R² is {women_r2/men_r2*100:.1f}% of men R²'
                ]
            }
            fairness_df = pd.DataFrame(fairness_summary)
            fairness_path = os.path.join(rq3_output, 'gender_fairness_summary.csv')
            fairness_df.to_csv(fairness_path, index=False)
            print(f"  Created: gender_fairness_summary.csv")
            
            # Print summary
            print(f"\n  Gender Fairness Analysis:")
            print(f"    Men R²:   {men_r2:.4f}")
            print(f"    Women R²: {women_r2:.4f}")
            diff = men_r2 - women_r2
            abs_diff = abs(diff)
            if diff > 0:
                # Men have higher R²
                pct_diff = (diff / women_r2) * 100
                print(f"    Difference: {diff:.4f} ({pct_diff:.1f}% higher for men)")
            else:
                # Women have higher R²
                pct_diff = (abs_diff / men_r2) * 100
                print(f"    Difference: {diff:.4f} ({pct_diff:.1f}% higher for women)")
            if abs_diff > 0.15:
                print(f"    ⚠️  Large difference detected - model fairness concerns")
            elif abs_diff > 0.05:
                print(f"    ⚠️  Moderate difference - model shows some unfairness")
            else:
                print(f"    ✓ Relatively fair - small difference between genders")
    
    print("\n" + "="*60)
    print("RQ3 ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to {rq3_output}/")
    print("\nKey outputs:")
    print(f"  - {rq3_output}/gender_analysis/ (participation analysis)")
    print(f"  - {rq3_output}/raw_data_gender_*.* (feature importance)")
    print(f"  - {rq3_output}/raw_data_subgroup_analysis.csv (R² by gender)")
    print(f"  - {rq3_output}/gender_fairness_summary.csv (fairness metrics)")

if __name__ == "__main__":
    main()

