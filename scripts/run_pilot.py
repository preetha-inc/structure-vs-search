"""Run A0 on an explicitly defined pilot question subset."""

from structure_vs_search.baselines.a0_bm25 import main


if __name__ == "__main__":
    main(default_mode="pilot")
