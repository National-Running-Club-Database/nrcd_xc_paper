"""
RQ1: Performance improvement patterns across race positions

This script runs all analyses related to RQ1:
- First to last race improvement
- Number of races impact on performance
- Team race participation analysis
- Nationals overlap analysis (racing more → better at nationals)
- Main ML model for improvement prediction

All outputs are saved to output/rq1/

Run from main directory: python scripts/rq1.py
"""

import os
import sys
import shutil

# Setup paths for imports (works from main directory or scripts directory)
from _setup_paths import setup_paths
script_dir = setup_paths()

# Set output directory for RQ1 (relative to main directory)
rq1_output = 'output/rq1'
os.makedirs(rq1_output, exist_ok=True)

def main():
    """Run all RQ1 analyses."""
    print("="*60)
    print("RQ1: PERFORMANCE IMPROVEMENT PATTERNS")
    print("="*60)
    
    # Import and run analyses with modified output directories
    print("\n1. First to Last Race Improvement Analysis...")
    from first_to_last_improvement import main as first_last_main
    import first_to_last_improvement
    original_output = first_to_last_improvement.output_dir
    first_to_last_improvement.output_dir = os.path.join(rq1_output, 'first_to_last_improvement')
    os.makedirs(first_to_last_improvement.output_dir, exist_ok=True)
    first_last_main()
    first_to_last_improvement.output_dir = original_output
    
    print("\n2. Number of Races Question Analysis...")
    from numberOfRacesQuestion import main as num_races_main
    import numberOfRacesQuestion
    original_output = numberOfRacesQuestion.output_dir
    numberOfRacesQuestion.output_dir = os.path.join(rq1_output, 'number_of_races_question')
    os.makedirs(numberOfRacesQuestion.output_dir, exist_ok=True)
    num_races_main()
    numberOfRacesQuestion.output_dir = original_output
    
    # Clean up any files that might have been created in the old location
    old_output = 'output/NumberOfRacesQuestion'
    if os.path.exists(old_output):
        # Only remove if it's empty or contains old files
        try:
            if os.path.isdir(old_output):
                # Check if directory is empty or only has old files
                files = os.listdir(old_output)
                if len(files) == 0:
                    os.rmdir(old_output)
                else:
                    # Remove old files but keep directory structure
                    for f in files:
                        old_file = os.path.join(old_output, f)
                        if os.path.isfile(old_file):
                            os.remove(old_file)
        except Exception as e:
            print(f"  Note: Could not clean up {old_output}: {e}")
    
    print("\n3. Number of Races Broken Down Analysis...")
    from numberOfRacesBrokenDown import main as num_races_broken_main
    import numberOfRacesBrokenDown
    original_output = numberOfRacesBrokenDown.output_dir
    numberOfRacesBrokenDown.output_dir = os.path.join(rq1_output, 'number_of_races_broken_down')
    os.makedirs(numberOfRacesBrokenDown.output_dir, exist_ok=True)
    num_races_broken_main()
    numberOfRacesBrokenDown.output_dir = original_output
    
    print("\n4. Team Race Participation Analysis...")
    from team_race_participation import main as team_participation_main
    import team_race_participation
    original_output = team_race_participation.output_dir
    team_race_participation.output_dir = os.path.join(rq1_output, 'team_race_participation')
    os.makedirs(team_race_participation.output_dir, exist_ok=True)
    team_participation_main()
    team_race_participation.output_dir = original_output
    
    print("\n5. Nationals Overlap Analysis (Racing More → Better at Nationals)...")
    from nationals_overlap_analysis import main as nationals_main
    import nationals_overlap_analysis
    original_output = nationals_overlap_analysis.output_dir
    nationals_overlap_analysis.output_dir = os.path.join(rq1_output, 'nationals_overlap')
    os.makedirs(nationals_overlap_analysis.output_dir, exist_ok=True)
    nationals_main()
    nationals_overlap_analysis.output_dir = original_output
    
    print("\n6. State Race Results Map...")
    from state_race_results_map import main as state_map_main
    state_map_main()
    # Move the output PDF to rq1 directory
    output_pdf = 'output/race_results_by_state_2023_2024_2025.pdf'
    if os.path.exists(output_pdf):
        dst_pdf = os.path.join(rq1_output, 'race_results_by_state_2023_2024_2025.pdf')
        if os.path.exists(dst_pdf):
            os.remove(dst_pdf)
        shutil.move(output_pdf, dst_pdf)
        print(f"  Moved: race_results_by_state_2023_2024_2025.pdf")
    
    print("\n7. Main ML Model - Improvement Prediction (3-year validation)...")
    from ml_improvement_prediction import main as ml_main
    ml_main()
    
    # Move ML model outputs to rq1 directory
    print("\n8. Organizing ML model outputs...")
    ml_outputs = [
        'raw_data_athlete_features.csv',
        'raw_data_feature_importance.csv',
        'raw_data_model_performance.csv',
        'raw_data_3year_validation_results.csv',
        'raw_data_important_statistics.csv',
        'raw_data_gender_feature_importance_comparison.csv',
        'raw_data_gender_feature_importance_pivot.csv',
        'raw_data_subgroup_analysis.csv',
        'time_standardization_comparison.csv',
        'time_standardization_comparison.pdf'
    ]
    
    # Get all PDFs from ML model (excluding FINDINGS_EXPLANATION.md)
    ml_pdfs = []
    if os.path.exists('output'):
        for f in os.listdir('output'):
            if f.startswith('raw_data_') and f.endswith('.pdf'):
                ml_pdfs.append(f)
            elif f.startswith('raw_data_') and f.endswith('.csv'):
                # Also catch any CSV files we might have missed
                if f not in ml_outputs:
                    ml_outputs.append(f)
    
    # Move all ML outputs (preserve FINDINGS_EXPLANATION.md)
    for filename in ml_outputs + ml_pdfs:
        src = os.path.join('output', filename)
        if os.path.exists(src):
            dst = os.path.join(rq1_output, filename)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
            print(f"  Moved: {filename}")
    
    # Note about FINDINGS_EXPLANATION.md
    if os.path.exists('output/FINDINGS_EXPLANATION.md'):
        print(f"  Preserved: FINDINGS_EXPLANATION.md (not moved)")
    
    print("\n" + "="*60)
    print("RQ1 ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to {rq1_output}/")
    print("\nKey outputs:")
    print(f"  - {rq1_output}/first_to_last_improvement/")
    print(f"  - {rq1_output}/number_of_races_question/")
    print(f"  - {rq1_output}/number_of_races_broken_down/")
    print(f"  - {rq1_output}/team_race_participation/")
    print(f"  - {rq1_output}/nationals_overlap/")
    print(f"  - {rq1_output}/race_results_by_state_2023_2024_2025.pdf")
    print(f"  - {rq1_output}/raw_data_*.csv (ML model results)")
    print(f"  - {rq1_output}/raw_data_*.pdf (ML model visualizations)")

if __name__ == "__main__":
    main()

