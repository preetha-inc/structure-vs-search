# Experiment Plan

Status: working plan; revise after corpus audit and pilot.

## Main question

When is graph structure worth its cost for multi-hop document QA, and can
lexical connectivity predict when agent-controlled sparse retrieval is sufficient?

## Planned systems

- A0: One-shot BM25.
- A1: Agentic BM25.
- A2: Corpus-appropriate document-aware sparse controls, conditional on the audit.
- D: Agentic dense retrieval, to distinguish sparse-retrieval limitations from
  limitations of agentic retrieval in general.
- S: HippoRAG 2.
- R: A test-time router that begins with sparse retrieval and selectively
  escalates to graph retrieval.

## Initial workflow

1. Audit corpus structure, passage lengths, titles, and document metadata.
2. Run a small end-to-end HippoRAG 2 pilot.
3. Implement and validate A0.
4. Implement A1 and D with shared controller and reader settings.
5. Decide whether A2 is meaningful based on the corpus audit.
6. Implement and tune R on development data.
7. Lock the budget protocol before full evaluation.

## Analysis

Measure answer quality, evidence coverage, token use, model calls, latency,
and graph construction cost. Analyze whether lexical connectivity and other
test-time evidence signals predict differences between retrieval systems.
Treat this as an empirical question, not an assumed result.
