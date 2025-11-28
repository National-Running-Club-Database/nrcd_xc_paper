"""
RQ2: Distribution of time changes between races

This script runs all analyses related to RQ2:
- Percentile time analysis
- Time standardization comparison (from ML model)

All outputs are saved to output/rq2/

Run from main directory: python scripts/rq2.py
"""

import os
import sys
import shutil

# Setup paths for imports (works from main directory or scripts directory)
from _setup_paths import setup_paths
script_dir = setup_paths()

# Set output directory for RQ2 (relative to main directory)
rq2_output = 'output/rq2'
os.makedirs(rq2_output, exist_ok=True)

def main():
    """Run all RQ2 analyses."""
    print("="*60)
    print("RQ2: DISTRIBUTION OF TIME CHANGES BETWEEN RACES")
    print("="*60)
    
    print("\n1. Percentile Time Analysis...")
    from percentile_time_analysis_final import main as percentile_main
    import percentile_time_analysis_final
    # Modify output directory
    original_output = percentile_time_analysis_final.output_dir
    percentile_time_analysis_final.output_dir = os.path.join(rq2_output, 'percentile_time_analysis')
    os.makedirs(percentile_time_analysis_final.output_dir, exist_ok=True)
    percentile_main()
    percentile_time_analysis_final.output_dir = original_output
    
    print("\n2. Time Standardization Comparison...")
    print("(Copying from RQ1 ML model outputs)")
    # Copy time standardization files from rq1 if they exist
    rq1_output = 'output/rq1'
    time_std_files = [
        'time_standardization_comparison.csv',
        'time_standardization_comparison.pdf'
    ]
    
    for filename in time_std_files:
        src = os.path.join(rq1_output, filename)
        if os.path.exists(src):
            dst = os.path.join(rq2_output, filename)
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")
        else:
            print(f"  Note: {filename} not found in {rq1_output}/ (run rq1.py first)")
    
    print("\n" + "="*60)
    print("RQ2 ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to {rq2_output}/")
    print("\nKey outputs:")
    print(f"  - {rq2_output}/percentile_time_analysis/")
    print(f"  - {rq2_output}/time_standardization_comparison.*")

if __name__ == "__main__":
    main()

