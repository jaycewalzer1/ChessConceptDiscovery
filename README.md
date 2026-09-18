# ChessConceptDiscovery

**Research question.** At a fixed search budget, does explicitly exploring candidate
moves with *different chess features* help a small AlphaZero-style agent recover
correct moves that its own initial policy ranks low?

"Creativity" is used here in a deliberately narrow, operational sense: **discovering a
useful solution that was initially unlikely under the agent's own policy**, where
"useful" means *verified correct by an exact oracle*. No claim is made about novelty to
humanity, artistic ability, subjective experience, or general creative intelligence.

No chess engine, engine API, or language model is used anywhere in the pipeline.
`python-chess` supplies the rules, FEN/PGN handling and board rendering only. All
move-quality judgements come from the learned network, the search, and a
self-implemented exact forced-mate verifier that is kept isolated from the agents at
test time.

## Results

One run, `outputs/laptop-main/`, on an Apple M4 Max (CPU only, 7 of 14 logical cores,
TensorFlow 2.21, Python 3.12.5). 74.2 minutes end to end: 6.6 data, 4.3 training, 9.5
development calibration, 53.7 evaluation. All ten stages completed; nothing was truncated.

The benchmark is 200 held-out bounded forced-mate positions drawn from **175 independent
source families**, graded by the exact verifier. One frozen 5,576,661-parameter
checkpoint is used by all four conditions.

| Condition | 256 sims | 1024 sims (primary) | 3072 sims |
| --- | --- | --- | --- |
| Standard PUCT | 23.5% | 29.0% | 35.0% |
| Quality-only | 27.5% | 29.0% | 37.0% |
| Random | 19.5% | 29.0% | 35.2% |
| **Feature-diverse** | **26.5%** | **32.5%** | **41.5%** |

**Primary endpoint — inconclusive.** At the primary budget, feature-diverse allocation
beat standard PUCT by **+3.5 percentage points (95% CI −1.5 to +8.8)**, paired over 200
positions with the interval bootstrapped over source families. The interval contains
zero, so by the pre-registered verdict rule this is *not* a demonstrated effect. On the
paired breakdown, feature-diverse solved 17 positions standard missed, standard solved 10
that feature-diverse missed, and both solved 48.

The direction was positive at all three budgets and the gap widened with search effort
(+3.0 pp at 256, +3.5 pp at 1024, +6.5 pp at 3072). That is the pattern a real effect
would produce. It is also a pattern that noise at this sample size can produce. This run
cannot distinguish the two.

**Secondary endpoint — one signal, on the hard subset.** Eligibility was fixed in advance
from the *common* raw network policy, so the denominator is identical for every method:
positions where **every** verified solution falls outside the policy's top three. 132 of
the 200 positions qualify, which is itself a finding — this agent's priors are weak, the
median correct move ranks **6th**, and only 13% rank first.

| Condition | Overlooked-solution recovery (n = 132) |
| --- | --- |
| Standard PUCT | 13.6% |
| Quality-only | 11.4% |
| Random | 15.9% |
| **Feature-diverse** | **18.9%** |

Against quality-only on this subset, feature-diverse gained **+7.6 pp (95% CI +2.2 to
+13.5)** — the one interval in the study that excludes zero. Against standard it gained
+5.3 pp (CI −1.5 to +12.3), which does not.

This is consistent with a simple mechanism: when the answer is a move the policy already
likes, deliberately looking elsewhere cannot help, and spending the budget on the
favourites (quality-only) is fine. When the answer is a move the policy would never
consider, looking elsewhere is the only thing that can find it. But **six comparisons were
run and one was significant**, which is roughly the number chance alone would produce.
No multiplicity correction was applied. Treat this as a lead, not a finding.

**Budget accounting held.** Every condition consumed exactly 1024 simulations at the
primary budget. Network evaluations differed honestly — 499 for standard against 677–701
for the allocation conditions — because each of those is charged the full logical cost of
the screening search it requires, even though one physical screen was shared.

**Verification.** `python -m scripts.verify_artifacts outputs/laptop-main` runs 457
checks with zero failures: every rendered board's piece census matches the FEN it claims
to show, every oracle line replays legally and ends in checkmate, every number in every
case study resolves to its raw `results.jsonl` row, and both HTML reports are
self-contained. The reports were also opened and inspected in a browser.

### What this does not show

No claim is made that the agent was creative in any ordinary sense, that the descriptor
distances measure creativity, or that this generalises beyond bounded forced mates. The
agent is weak in absolute terms — a 29% baseline on mate-in-1-and-2 puzzles is not strong
play — and a weak prior is precisely the regime where a diversity intervention has the
most room to help. The effect may shrink or vanish in a stronger agent.

## Proposed next experiment

**Power, not cleverness.** The observed effect is small (3–6 pp) and 175 families give a
confidence interval roughly ±5 pp wide, so the current design cannot resolve an effect of
the size actually observed. The single most useful next run changes no method at all:

> Re-run the **identical frozen protocol** on approximately **600 test positions from ~500
> source families, at the 3072-simulation budget only**, where the observed gap is
> largest.

Dropping the two smaller budgets pays for the larger test set at roughly the same
wall-clock cost, and tripling the number of families roughly halves the interval width.
That is enough to separate "+6 pp real" from "zero" for the primary endpoint. The
experiment is pre-registered by construction: `protocol.json` already exists and its
SHA-256 is recorded, so the only changes are `test_cap`, `data_target`, and the budget
list.

Two secondary questions worth attaching to that run, both cheap:

1. **Is the mechanism the one claimed?** Split the test set by the raw-policy rank of the
   best solution. The prediction is that the advantage is near zero when the solution is
   already rank 1–3 and concentrates entirely in the tail. This run hints at that; a
   properly powered run can test it as a stated prediction rather than read it off
   afterwards.
2. **Does it survive a better prior?** Train the bootstrap for longer on more data until
   the baseline reaches roughly 50%, then re-run. If the advantage shrinks as the policy
   improves, the honest conclusion is that feature diversity compensates for a weak
   policy rather than adding a capability — which is a useful negative result and the
   most likely outcome.

## Quick start

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m pytest tests/test_core.py -q          # 43 correctness tests

# short end-to-end smoke run (a few minutes, NOT a research result)
.venv/bin/python -m scripts.run_experiment --profile smoke --max-minutes 12 --seed 7

# full comparative experiment
.venv/bin/python -m scripts.run_experiment --profile laptop --max-minutes 120 --seed 42
```

Everything lands in `outputs/<run_id>/`. Open `report.html` directly in a browser —
it is self-contained (inline SVG, no CDN, no server needed).

### Options

| Flag | Meaning |
| --- | --- |
| `--profile {smoke,laptop}` | resource profile (budgets, dataset size, epochs) |
| `--max-minutes` | total wall-time budget for data + training + experiments |
| `--seed` | master seed |
| `--output-dir` / `--run-id` | where artifacts go |
| `--model-path` | reuse an existing frozen checkpoint |
| `--data-path` | reuse an existing dataset directory |
| `--device {cpu,auto}` | force CPU or allow an accelerator |
| `--threads` / `--memory-gib` | logical-core cap and soft memory target |
| `--resume` | reuse completed stages and already-written result rows |
| `--skip-tests` | skip the in-run correctness suite |

Resuming an interrupted run:

```bash
.venv/bin/python -m scripts.run_experiment --profile laptop --resume \
    --output-dir outputs/<run_id> --run-id <run_id>
```

## The four conditions

All four run on the **same frozen checkpoint**, the same environment, the same action
encoding, and the same total simulation budget.

| # | Condition | Candidate selection |
| --- | --- | --- |
| 1 | Standard PUCT | no intervention; move by root visit count |
| 2 | Quality-only | top-K by screening visit share |
| 3 | Random | top-1 anchor + K−1 uniform draws from the pool (seeds recorded) |
| 4 | Feature-diverse | greedy `normalised quality + λ · min standardised feature distance` |

Conditions 2–4 share one screening implementation and one eligible pool (every legal
root move). A screening search may be physically shared between them, but its **full
logical cost is charged to each condition that needs it**, and the feature-computation
wall time is charged to condition 4. No condition inherits another's search tree or any
oracle result.

## Artifacts

```
outputs/<run_id>/
  protocol.json             frozen pre-registration + SHA-256
  hardware.json             detected OS/CPU/memory/accelerator, thread caps
  data_manifest.json        split sizes, family assignment, split hashes, leakage controls
  training_summary.json     bootstrap + self-play histories
  checkpoint_metadata.json  architecture, parameter count, weights SHA-256
  results.jsonl             one row per position x condition x budget x seed
  summary.json              aggregated statistics and the verdict
  report.md / report.html   the report (HTML is self-contained and printable)
  case_studies.html         printable case-study gallery
  case_studies/             index.json, pgn/, figures/ (standalone board SVGs)
  examples.pgn              all case-study PGNs
  figures/                  fig1..fig8 as SVG + PNG + a .json of the plotted numbers
  logs/                     run log and pytest output
```

## What was repaired

See [`METHODOLOGY.md`](METHODOLOGY.md) for the full audit. In short, the original
repository could not be imported or trained: `agents/alphazero/agent.py` imported a
`mcts` module that did not exist, `NeuralNetwork.train()` was `pass`, the policy head
was `Dense(input_size, activation='softmax')`, `save_weights`/`load_weights` called
methods that do not exist on `tf.keras.Model`, `encode_board` returned zeros,
`decode_action` and `get_valid_moves` were `pass`, `ChessEnv` was defined twice (the
first on an OpenSpiel base class behind an empty gitlink), and two scripts had trailing
spaces in their filenames.

## Limitations

- The observation is **not** fully Markov with respect to threefold repetition; only a
  twofold-repetition flag is encoded. The *search* state keeps the full board with its
  move stack, and the transposition key never merges states with different repetition or
  halfmove status.
- Training is **supervised tactical bootstrapping followed by an AlphaZero-style
  procedure**, not tabula-rasa AlphaZero.
- The benchmark is bounded forced mates, not general chess strength.
- Descriptor distances are a working construct for this experiment. They are **not** an
  established measure of creativity.
