"""A0: single-step BM25 retrieval with optional local HF answer generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from structure_vs_search.data.loader import load_corpus, load_questions
from structure_vs_search.evaluation.budget import (
    render_evidence,
    passage_text,
    select_passages_within_budget,
)
from structure_vs_search.evaluation.cost_logger import CostLogger
from structure_vs_search.evaluation.metrics import answer_metrics
from structure_vs_search.retrieval.bm25 import BM25Retriever


ROOT = Path(__file__).resolve().parents[3]


def read_config() -> tuple[dict[str, Any], dict[str, Any]]:
    with (ROOT / "configs" / "budgets.yaml").open(encoding="utf-8") as file:
        budgets = yaml.safe_load(file)
    with (ROOT / "configs" / "models.yaml").open(encoding="utf-8") as file:
        models = yaml.safe_load(file)
    return budgets, models


def load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Answer generation and exact token budgeting require transformers. "
            "Install it in your experiment environment, for example in Colab."
        ) from exc

    return AutoTokenizer.from_pretrained(model_id)


def load_generator(model_id: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Answer generation requires torch and transformers. "
            "Install them in your experiment environment, for example in Colab."
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model


def generate_answer(
    question: str,
    passages: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    """Generate an answer and return answer, input tokens, output tokens, latency."""
    import torch

    evidence = render_evidence(passages)
    prompt = (
        "Answer the question using only the evidence below. "
        "If the evidence is insufficient, say so briefly. "
        "Give a concise answer without explaining your reasoning.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_token_count = int(inputs["input_ids"].shape[1])
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    generated_ids = output[0, input_token_count:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return prediction, input_token_count, int(generated_ids.shape[0]), latency_ms


def read_question_ids(path: str | None) -> set[str] | None:
    if path is None:
        return None

    ids = {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not ids:
        raise ValueError(f"No question IDs found in {path}")
    return ids


def run(args: argparse.Namespace) -> None:
    budgets_config, models_config = read_config()
    questions = load_questions(args.questions)
    corpus = load_corpus(args.corpus)

    question_ids = read_question_ids(args.question_ids_file)
    if question_ids is not None:
        available_ids = {q["id"] for q in questions}
        missing = question_ids - available_ids
        if missing:
            raise ValueError(
                f"Question IDs not found in the question file: {sorted(missing)}"
            )
        questions = [q for q in questions if q["id"] in question_ids]

        # Preserve the ID-file order, not the dataset file's order.
        order = {
            line.strip(): i
            for i, line in enumerate(
                Path(args.question_ids_file).read_text(encoding="utf-8").splitlines()
            )
            if line.strip()
        }
        questions.sort(key=lambda q: order[q["id"]])
    elif not args.all_questions:
        raise ValueError(
            "Supply --question-ids-file for a defined subset, or "
            "--all-questions for a full run. Arbitrary first-N selection is disabled."
        )

    if not questions:
        raise ValueError("No questions selected.")

    model_id = models_config[args.model_tier]["model_id"]

    if args.retrieval_only:
        tokenizer = None
        model = None

        def count_tokens(text: str) -> int:
            # Diagnostic estimate only; not comparable to model token counts.
            return len(text.split())

        token_count_method = "whitespace_approximation_not_comparable"
    else:
        tokenizer = load_tokenizer(model_id)

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        model = load_generator(model_id)
        token_count_method = f"tokenizer:{model_id}"
    retriever = BM25Retriever(corpus)
    logger = CostLogger(args.output)

    token_caps = args.token_caps or budgets_config["shared_evidence_token_caps"]
    top_k = int(budgets_config["A0"]["bm25_top_k"])
    max_new_tokens = int(models_config[args.model_tier]["max_new_tokens"])

    for question_record in questions:
        # Deliberately construct a restricted view. Gold answers and supporting
        # facts are not passed to retrieval or answer generation.
        question_id = question_record["id"]
        question_text = question_record["question"]
        ranked = retriever.search(question_text, top_k=top_k)

        for token_cap in token_caps:
            selected = select_passages_within_budget(
                ranked,
                token_cap=int(token_cap),
                count_tokens=count_tokens,
            )
            evidence_passages = selected["passages"]

            prediction = None
            input_tokens = 0
            output_tokens = 0
            latency_ms = 0.0
            metrics = None

            if model is not None:
                prediction, input_tokens, output_tokens, latency_ms = generate_answer(
                    question_text,
                    evidence_passages,
                    tokenizer,
                    model,
                    max_new_tokens,
                )

                # Gold is accessed only after generation for evaluation.
                gold = question_record.get("answer")
                if isinstance(gold, str):
                    metrics = answer_metrics(prediction, gold)

            logger.log(
                {
                    "question_id": question_id,
                    "method": "A0_BM25",
                    "model_id": model_id,
                    "token_count_method": token_count_method,
                    "model_tier": args.model_tier,
                    "retrieval_only": args.retrieval_only,
                    "token_cap": int(token_cap),
                    "prediction": prediction,
                    "em": None if metrics is None else metrics["em"],
                    "f1": None if metrics is None else metrics["f1"],
                    "retrieved_passages": [
                        {
                            "id": p["id"],
                            "rank": p["rank"],
                            "score": p["score"],
                        }
                        for p in ranked
                    ],
                    "selected_passage_ids": [p["id"] for p in evidence_passages],
                    "evidence_token_count": selected["evidence_token_count"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model_calls": 0 if model is None else 1,
                    "latency_ms": latency_ms,
                }
            )

    print(f"Questions processed: {len(questions)}")
    print(f"Evidence budgets: {token_caps}")
    print(f"Output appended to: {args.output}")
    if args.retrieval_only:
        print("Retrieval-only run: no answer-quality metrics were produced.")


def main(default_mode: str = "pilot") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Question JSON/JSONL file")
    parser.add_argument("--corpus", required=True, help="Passage corpus JSON/JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL log path")
    parser.add_argument("--question-ids-file", help="One agreed question ID per line")
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument(
        "--model-tier",
        choices=("development", "evaluation"),
        default="development",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation; uses approximate whitespace counts for diagnostics",
    )
    parser.add_argument(
        "--token-caps",
        type=int,
        nargs="+",
        help="Override configured token caps, for debugging only",
    )
    args = parser.parse_args()

    if default_mode == "pilot" and not args.question_ids_file:
        parser.error("Pilot runs require --question-ids-file with the agreed question IDs.")
    if args.question_ids_file and args.all_questions:
        parser.error("Use either --question-ids-file or --all-questions, not both.")
    if args.token_caps and any(cap < 0 for cap in args.token_caps):
        parser.error("Token caps must be non-negative.")

    run(args)


if __name__ == "__main__":
    main()
