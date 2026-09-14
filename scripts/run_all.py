"""
Run the full analysis pipeline (sample summary → RQ1 → RQ2 → RQ3).

Outputs:
  - output/sample_summary*
  - output/rq1/
  - output/rq2/
  - output/rq3/

Run from the repository root:
  python scripts/run_all.py
"""

from __future__ import annotations

import time

from _setup_paths import setup_paths

setup_paths()


def main() -> None:
    print("=" * 80)
    print("RUNNING FULL ANALYSIS PIPELINE")
    print("=" * 80)
    print("\nThis will run:")
    print("  0. Sample / coverage summary")
    print("  1. RQ1: within-season improvement, teams, leakage-controlled ML")
    print("  2. RQ2: multi-season progression (race-count consistency filter)")
    print("  3. RQ3: gender participation + team-association robustness")
    print("\nOutputs go under output/ (sample_summary*, rq1/, rq2/, rq3/).")
    print("=" * 80)

    start = time.time()

    print("\n" + "=" * 80)
    print("SAMPLE SUMMARY")
    print("=" * 80)
    from sample_summary import main as sample_main

    sample_main()

    print("\n" + "=" * 80)
    print("STARTING RQ1")
    print("=" * 80)
    from rq1 import main as rq1_main

    rq1_main()

    print("\n" + "=" * 80)
    print("STARTING RQ2")
    print("=" * 80)
    from rq2 import main as rq2_main

    rq2_main()

    print("\n" + "=" * 80)
    print("STARTING RQ3")
    print("=" * 80)
    from rq3 import main as rq3_main

    rq3_main()

    elapsed = time.time() - start
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nTotal execution time: {elapsed / 60:.1f} minutes ({elapsed:.0f} seconds)")
    print("\nOutput directories:")
    print("  - output/sample_summary*")
    print("  - output/rq1/")
    print("  - output/rq2/")
    print("  - output/rq3/")
    print("=" * 80)


if __name__ == "__main__":
    main()
