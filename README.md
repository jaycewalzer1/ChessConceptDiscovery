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
