"""Run A0 across the explicitly selected full question dataset."""

import sys

from structure_vs_search.baselines.a0_bm25 import main


if __name__ == "__main__":
    if "--all-questions" not in sys.argv:
        raise SystemExit(
            "Full experiments require --all-questions. "
            "Use scripts/run_pilot.py for the agreed pilot subset."
        )
    main(default_mode="experiment")
