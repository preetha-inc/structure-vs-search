def test_load_corpus_preserves_idx_id(tmp_path):
    import json

    from structure_vs_search.data.loader import load_corpus

    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps([{"idx": 42, "title": "Example", "text": "Example passage."}]),
        encoding="utf-8",
    )

    passages = load_corpus(path)

    assert passages[0]["id"] == "42"
    assert passages[0]["title"] == "Example"
    assert passages[0]["text"] == "Example passage."