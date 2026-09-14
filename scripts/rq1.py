"""
RQ1: Performance improvement patterns across race positions

Runs the core RQ1 stack into output/rq1/:
- First-to-last / race-count descriptives, team & nationals analyses
- Weekly participation, state map, overlay figures
- Leakage-controlled ML + feature exclusion / compact primary suite
- Robustness (ablation, sensitivity, diagnostics), mixed effects
- Enrichment, mathematical contributions (weather-path, SER, ERO)
- Null-result diagnostics, CIKM R² audit, underexplored mechanisms

Team-association robustness lives in rq3.py (output/rq3/).

Run from repository root: python scripts/rq1.py
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
    
    print("\n6. Weekly Participation Analysis...")
    from weekly_participation_analysis import main as weekly_main
    import weekly_participation_analysis
    original_output = weekly_participation_analysis.output_dir
    weekly_participation_analysis.output_dir = os.path.join(rq1_output, 'weekly_participation')
    os.makedirs(weekly_participation_analysis.output_dir, exist_ok=True)
    weekly_main()
    weekly_participation_analysis.output_dir = original_output
    
    print("\n7. Top 25 Teams at Nationals Analysis...")
    from top25_team_analysis import main as top25_main
    import top25_team_analysis
    original_output = top25_team_analysis.output_dir
    top25_team_analysis.output_dir = os.path.join(rq1_output, 'top25_teams')
    os.makedirs(top25_team_analysis.output_dir, exist_ok=True)
    top25_main()
    top25_team_analysis.output_dir = original_output
    
    print("\n8. State Race Results Map...")
    try:
        import state_race_results_map
        state_race_results_map.output_dir = rq1_output
        from state_race_results_map import main as state_map_main
        state_map_main()
    except Exception as e:
        print(f"  ERROR running state race results map (continuing): {e}")
    
    print("\n9. Main ML Model - Improvement Prediction (3-year validation)...")
    from ml_improvement_prediction import main as ml_main
    # Pass output directory directly to ML model
    ml_main(output_dir=rq1_output)
    
    print("\n10-11. Combined Overlay Plots (2023, 2024, 2025) - Men's & Women's...")
    # Set output directory for overlay plots
    overlay_output_dir = os.path.join(rq1_output, 'overlay_plots')
    os.makedirs(overlay_output_dir, exist_ok=True)
    try:
        # Single gender-parametrized overlay module (replaces the old mens/womens pair)
        from create_combined_overlay_2023_2024_2025 import main as combined_overlay_main
        for gender in ('M', 'F'):
            combined_overlay_main(gender=gender, output_dir=overlay_output_dir)
    except Exception as e:
        print(f"  ERROR running combined overlay script: {e}")
        import traceback
        traceback.print_exc()

    print("\n11b. Combined years all-plots grid...")
    try:
        from create_combined_2023_2024_2025_all_plots_grid import main as grid_main
        grid_main(output_dir=overlay_output_dir)
    except Exception as e:
        print(f"  ERROR running all-plots grid: {e}")
        import traceback
        traceback.print_exc()

    print("\n12. Robustness: Feature Ablation (feature removal checks)...")
    try:
        from feature_ablation_robustness import main as ablation_main
        ablation_main(output_dir=rq1_output)
    except Exception as e:
        print(f"  ERROR running feature ablation robustness: {e}")
        import traceback
        traceback.print_exc()

    print("\n13. Robustness: Sensitivity Sweep (key modeling choices)...")
    try:
        from sensitivity_analysis_sweep import main as sensitivity_main
        sensitivity_main(output_dir=rq1_output)
    except Exception as e:
        print(f"  ERROR running sensitivity sweep: {e}")
        import traceback
        traceback.print_exc()

    print("\n14. Prediction diagnostics: learning curves and outlier sensitivity...")
    try:
        from prediction_diagnostics import main as diagnostics_main
        diagnostics_main(output_dir=rq1_output)
    except Exception as e:
        print(f"  ERROR running prediction diagnostics: {e}")
        import traceback
        traceback.print_exc()

    print("\n15. Feature exclusion audit (leakage vs over-exclusion; compact vs legacy)...")
    try:
        from feature_exclusion_audit import main as exclusion_main
        exclusion_main(output_dir=os.path.join(rq1_output, 'feature_exclusion_audit'))
    except Exception as e:
        print(f"  ERROR running feature exclusion audit: {e}")
        import traceback
        traceback.print_exc()

    print("\n15b. Compact primary six-model suite (table + learning curves + perm)...")
    try:
        from compact_model_suite import main as compact_main
        compact_main()
    except Exception as e:
        print(f"  ERROR running compact model suite: {e}")
        import traceback
        traceback.print_exc()

    print("\n16. Explanatory Model: Mixed-Effects (athlete random effects)...")
    try:
        from mixed_effects_explanatory_model import main as mixed_main
        mixed_main(output_dir=rq1_output)
    except Exception as e:
        print(f"  ERROR running mixed-effects model: {e}")
        import traceback
        traceback.print_exc()

    print("\n16b. Gender x race-count interaction mixed model...")
    try:
        from mixed_effects_interaction import main as interaction_main
        interaction_main(output_dir=rq1_output)
    except Exception as e:
        print(f"  ERROR running interaction mixed model: {e}")
        import traceback
        traceback.print_exc()

    print("\n17. Paper enrichment: dose-response, weather inflation, retention, early-window...")
    try:
        from paper_enrichment_analyses import main as enrichment_main
        enrichment_main(output_dir=os.path.join(rq1_output, 'enrichment'))
    except Exception as e:
        print(f"  ERROR running enrichment analyses: {e}")
        import traceback
        traceback.print_exc()

    print("\n18. Mathematical contributions: weather-path identity, SER, ERO...")
    try:
        from mathematical_contributions import main as math_main
        math_main(output_dir=os.path.join(rq1_output, 'mathematical_contributions'))
    except Exception as e:
        print(f"  ERROR running mathematical contributions: {e}")
        import traceback
        traceback.print_exc()

    print("\n19. Null-result diagnostics: noise ceiling, permutation-R^2, classification AUC...")
    try:
        from rq1_null_result_diagnostics import main as null_diag_main
        null_diag_main(output_dir=rq1_output)
    except Exception as e:
        print(f"  ERROR running null-result diagnostics: {e}")
        import traceback
        traceback.print_exc()

    print("\n20. CIKM vs analysis R^2 discrepancy audit (leakage mechanism)...")
    try:
        from cikm_r2_discrepancy_audit import main as cikm_audit_main
        cikm_audit_main(output_dir=os.path.join(rq1_output, 'cikm_r2_discrepancy_audit'))
    except Exception as e:
        print(f"  ERROR running CIKM R^2 audit: {e}")
        import traceback
        traceback.print_exc()

    print("\n21. Underexplored mechanisms: roster depth, peer density, quantile CATE...")
    try:
        from underexplored_mechanisms import main as mechanisms_main
        mechanisms_main(output_dir=os.path.join(rq1_output, 'underexplored_mechanisms'))
    except Exception as e:
        print(f"  ERROR running underexplored mechanisms: {e}")
        import traceback
        traceback.print_exc()

    print("\n22. Robustness / sensitivity checks (matched volume, team confounders, reliability, weather holdout)...")
    try:
        from robustness_checks import main as gap_main
        gap_main(output_dir=os.path.join(rq1_output, 'robustness_checks'))
    except Exception as e:
        print(f"  ERROR running robustness checks: {e}")
        import traceback
        traceback.print_exc()
    
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
    print(f"  - {rq1_output}/top25_teams/")
    print(f"  - {rq1_output}/weekly_participation/")
    print(f"  - {rq1_output}/race_results_by_state_2023_2024_2025.pdf")
    print(f"  - {rq1_output}/raw_data_*.csv (ML model results)")
    print(f"  - {rq1_output}/raw_data_*.pdf (ML model visualizations)")
    print(f"  - {rq1_output}/overlay_plots/")
    print(f"  - {rq1_output}/feature_exclusion_audit/ (compact primary tables)")
    print(f"  - {rq1_output}/mathematical_contributions/")
    print(f"  - {rq1_output}/robustness_feature_ablation/")
    print(f"  - {rq1_output}/sensitivity_sweep/")
    print(f"  - {rq1_output}/null_result_diagnostics/")
    print(f"  - {rq1_output}/cikm_r2_discrepancy_audit/")
    print(f"  - {rq1_output}/underexplored_mechanisms/")
    print(f"  - {rq1_output}/robustness_checks/")
    print(f"  - {rq1_output}/prediction_diagnostics/")
    print(f"  - {rq1_output}/mixed_effects/")

if __name__ == "__main__":
    main()

