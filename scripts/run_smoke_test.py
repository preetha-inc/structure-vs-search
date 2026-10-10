from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for filename in ("datasets.yaml", "models.yaml", "budgets.yaml", "experiments.yaml"):
        path = ROOT / "configs" / filename
        with path.open(encoding="utf-8") as file:
            yaml.safe_load(file)
        print(f"OK: {path.relative_to(ROOT)}")

    print("Project configuration smoke test passed.")


if __name__ == "__main__":
    main()
