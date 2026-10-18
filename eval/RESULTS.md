# Evaluation results

How good are the links Constellation writes, and what does each pipeline stage
buy? Every row below comes from **real recorded runs** (Groq `openai/gpt-oss-120b`,
Gemini `gemini-embedding-2`, recorded 2026-10-18) replayed offline from
`eval/cassettes/`. Regenerate the table with:

```bash
python3 eval/run_eval.py --replay --write
```

## Setup

- **Vault** (`eval/data/vault/`): 51 synthetic notes across five topics (machine
  learning, distributed systems, cooking, ancient/modern history, gardening).
  It includes near-miss distractors that share vocabulary across topics but are
  unrelated: *Neural Network Pruning* vs *Pruning Fruit Trees*, *Byzantine Fault
  Tolerance* vs *Byzantine Empire*, *CAP Theorem and Network Partitions* vs
  *Partition of India*, *Batch Normalization* vs *Batch Cooking*, and others.
- **Gold** (`eval/data/gold_links.json`): undirected pairs labelled by reading the
  notes, before any pipeline run. There are 35 **required** pairs and 35 **optional**
  pairs (defensible but debatable). An optional pair the pipeline predicts is neither
  rewarded nor penalised, and optional pairs are left out of recall. The file also
  lists 12 **distractor** pairs. Any other predicted pair is a false positive.
- **Prediction**: the links the run would render into `## Related Links`
  (dry run, after the per-note cap), counted as undirected pairs.
- **Cost**: Groq list price for gpt-oss-120b ($0.15 / M input, $0.60 / M output
  tokens); embeddings are not included (free tier). *Est. live LLM time* adds up
  the recorded latency of every LLM request (rate-limit retries included) and
  divides by the concurrency of 2. Replay itself takes under a second.

## Results

<!-- results:start -->
Vault: 51 notes; gold: 35 required pairs, 35 optional, 12 distractors. Model `openai/gpt-oss-120b`, embeddings `models/gemini-embedding-2`, concurrency 2. Precision/recall/F1 in %.

| Config | Pairs | TP | FP | Distractor hits | Precision | Recall | F1 | Tags | LLM calls | Tokens in/out | Est. cost $ | Est. live LLM time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `default` | 62 | 33 | 10 | 0 | 76.7 | 94.3 | 84.6 | 184 | 179 | 260,352 / 148,946 | 0.1284 | 980.4 |
| `no-concept-verify` | 59 | 32 | 13 | 0 | 71.1 | 91.4 | 80.0 | 184 | 129 | 191,308 / 104,360 | 0.0913 | 705.1 |
| `no-relation-verify` | 75 | 34 | 17 | 0 | 66.7 | 97.1 | 79.1 | 184 | 153 | 204,548 / 124,998 | 0.1057 | 815.8 |
| `k5` | 55 | 31 | 7 | 0 | 81.6 | 88.6 | 84.9 | 184 | 175 | 235,384 / 139,434 | 0.119 | 981.9 |
| `k20` | 61 | 32 | 7 | 0 | 82.0 | 91.4 | 86.5 | 184 | 181 | 293,048 / 151,811 | 0.135 | 1121.0 |
| `max-links-3` | 60 | 33 | 8 | 0 | 80.5 | 94.3 | 86.8 | 184 | 179 | 260,352 / 148,946 | 0.1284 | 980.4 |
| `max-links-10` | 62 | 33 | 10 | 0 | 76.7 | 94.3 | 84.6 | 184 | 179 | 260,352 / 148,946 | 0.1284 | 980.4 |
| `retrieval-only` | 144 | 35 | 78 | 2 | 31.0 | 100.0 | 47.3 | 184 | 102 | 120,350 / 88,077 | 0.0709 | 525.0 |
| `retrieval-only-2` | 72 | 31 | 18 | 1 | 63.3 | 88.6 | 73.8 | 184 | 102 | 120,350 / 88,077 | 0.0709 | 525.0 |

| Config | What changes |
|---|---|
| `default` | Full pipeline (k=10, 5 links/note) |
| `no-concept-verify` | Skip concept verification |
| `no-relation-verify` | Skip relationship verification |
| `k5` | Retrieval k=5 |
| `k20` | Retrieval k=20 |
| `max-links-3` | MAX_LINKS_PER_NOTE=3 |
| `max-links-10` | MAX_LINKS_PER_NOTE=10 |
| `retrieval-only` | Embedding top-k only, no relationship LLM (5 links/note) |
| `retrieval-only-2` | Embedding top-k only, 2 links/note |
<!-- results:end -->

## What the numbers say

**The relationship LLM is the stage that matters.** Linking by embedding
similarity alone finds every required pair, but at 31% precision: it attaches
five neighbours to every note, isolated ones included. Even with a tight cap of 2
links per note it reaches only 73.8 F1, against 84.6 for the full pipeline. It is
also the only setup that links distractors (*Neural Network Pruning* to *Pruning
Fruit Trees*, *Sourdough Starter* to *Composting*). No LLM configuration made any
of the 12 distractor links.

**Relationship verification earns its cost.** Dropping it raises recall
slightly (97.1 vs 94.3) but adds 7 false positives: 66.7 vs 76.7 precision,
-5.5 F1. The pass costs 26 calls (about 18% of the run's dollar cost).

**Concept verification helps, but less per dollar.** Skipping it costs 4.6 F1
(80.0 vs 84.6) and saves 51 calls, about 29% of the cost. It is worth keeping
for link quality, but it is the first stage to cut if cost matters more than a
few points. Tag counts were identical across configurations (184), so the
verifier's effect on tags is not visible in this metric.

**Retrieval k and the link cap barely matter here.** k=20 scored best (86.5
F1) and k=5 had the highest precision of the k settings (81.6) but lost recall.
After verification no note has more than 4 links (43 of 51 have 3 or
fewer), so caps of 5 and 10 never bind. A cap of 3 trims the four 4-link
notes and removes two false positives (86.8 F1). On a 35-pair gold set one pair is worth about 1.5 F1
points, so differences of 1-2 points between these rows are within the noise of a
single run, not a reason to change the defaults.

**Where the default still goes wrong.** Of its 10 false positives, four are
arguable calls the labels counted as wrong, e.g. *Gradient Descent* to
*Overfitting and Regularization* and *Gossip Protocols* to *Leader Election*. The
labels were fixed before the first run and not edited afterwards. The clearest
failure mode is the deliberately isolated *Stovetop Popcorn* note: the model
links it to *Maillard Reaction*, *Hash Browns* and *Cast Iron Seasoning*
rather than leaving it unlinked. The two required pairs it misses are *Batch
Normalization* to *Learning Rate Schedules* and *Database Sharding* to
*Replication and Quorums*.

## Caveats

- These are single runs. The model runs at temperature 0, but Groq's output still
  varies slightly between runs, so re-recording will move numbers by a point or two.
- The vault is synthetic and small (51 notes, 5 well-separated topics). Real vaults
  are messier, with more overlap between topics, so absolute scores will be lower.
  The relative ordering of stages is the more transferable result.
- Gold labels are one person's judgement; the optional tier absorbs
  the most debatable pairs, but not all of them.

## Re-recording

Needs `GROQ_API_KEY_*` and `GOOGLE_API_KEY_*` in the environment or `.env`:

```bash
python3 eval/run_eval.py --record            # all configurations
python3 eval/run_eval.py --replay --write --update-baseline
```

Requests already present in any cassette are reused, so recording every
configuration made 409 live LLM calls in total (about 690K input and 330K output
tokens, roughly $0.30 at list price). The Gemini free tier allows 100 embedding requests per
minute per key, so the recorder paces embedding calls and waits out the quota.
Model settings are pinned in `run_eval.py`, so a local `.env` cannot change what
is measured. CI (`.github/workflows/eval.yml`) replays the default
configuration and fails if its F1 drops more than 5 points below
`eval/baseline.json`.
