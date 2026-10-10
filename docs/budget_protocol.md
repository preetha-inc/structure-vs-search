# Budget Protocol

Status: provisional. Confirm the settings after the pilot and before full runs.

## Shared evidence budget

Use the reader model's tokenizer to measure passage evidence. The initial
shared evidence-token cap is 4,096 tokens per question, with 2,048 and 8,192
as planned sensitivity settings.

- A0: Retrieve the top 10 BM25 passages, then select ranked evidence within
  the shared cap.
- A1 and D: Retrieve up to 3 new passages per step, for at most 8 steps,
  stopping earlier when the evidence cap is reached.
- A2: Use the same agentic defaults if the corpus audit justifies this control.
- S: Select HippoRAG 2's ranked passages until the same evidence cap is reached.

## Cost accounting

The evidence cap does not replace full inference-cost accounting. Log prompt
tokens, completion tokens, model calls, and latency. Count repeated passage
exposure each time passage text is sent to an LLM call, and separately report
unique evidence passages and tokens.

Record graph construction and indexing cost separately from online retrieval.
Report both the offline cost and an amortized per-query estimate at stated
query volumes.

Do not describe these provisional values as a finalized protocol until they
have been checked against the pilot implementation.
