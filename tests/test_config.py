from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_budget_config_is_provisional_and_positive():
    path = ROOT / "configs" / "budgets.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["status"] == "provisional"
    assert config["shared_evidence_token_cap"] > 0
    assert config["A0"]["bm25_top_k"] > 0
