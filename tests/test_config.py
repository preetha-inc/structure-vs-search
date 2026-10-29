from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_budget_config_is_provisional_and_positive():
    path = ROOT / "configs" / "budgets.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["status"] == "provisional"

    caps = config["shared_evidence_token_caps"]
    assert caps
    assert all(isinstance(cap, int) and cap > 0 for cap in caps)
    assert caps == sorted(caps)

    assert config["primary_evidence_token_cap"] in caps