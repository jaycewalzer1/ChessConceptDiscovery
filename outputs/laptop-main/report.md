# Feature-guided exploration in a small AlphaZero-style chess agent

Run `laptop-main` | generated 2026-09-17T18:47:55-04:00 | total wall time 74.2 min

## Verdict: INCONCLUSIVE

Observed difference +3.5 pp with 95% CI -1.5 to +8.8 pp; the interval contains zero.

## Question

At a fixed search budget, does explicitly exploring candidate moves with different chess
features help a small AlphaZero-style agent recover correct moves its initial policy
overlooked? Creativity is operationalised narrowly as *discovering a useful solution that
was initially unlikely under the agent's own policy*. No claim is made about novelty to
humanity, artistic ability, or general creative intelligence.

## Setup

- Hardware: Apple M4 Max, 14 logical cores (capped at 7), 36 GiB RAM, accelerator: CPU only (no GPU/Metal device exposed to TensorFlow)
- Test set: 200 positions across 175 independent families
- Primary budget: 1024 MCTS simulations per position
- Checkpoint: 5,576,661 parameters, SHA-256 `f62131806f66ee90...`
- Training: supervised tactical bootstrapping on exactly-verified forced-mate solutions, followed by an AlphaZero-style self-play/MCTS fine-tuning stage; this is NOT tabula-rasa AlphaZero training

## Results at the primary budget

| Condition | Solved / n | Success rate [95% CI] | Mean sims | Mean NN evals | Mean time (s) |
|---|---:|---:|---:|---:|---:|
| Standard PUCT | 58/200 | 29.0% [22.1, 35.9] | 1024 | 499 | 0.783 |
| Quality-only allocation | 58/200 | 29.0% [22.4, 36.0] | 1024 | 677 | 0.979 |
| Random allocation | 58/200 | 29.0% [23.0, 35.2] | 1024 | 701 | 1.002 |
| Feature-diverse allocation | 65/200 | 32.5% [25.8, 39.3] | 1024 | 690 | 1.000 |

## Paired comparisons

| Comparison | Difference | 95% CI (pp) | n | Feature only | Baseline only | Both solved | Both failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Feature-diverse - Standard PUCT | +3.5 pp | [-1.5, +8.8] | 200 | 17 | 10 | 48 | 125 |
| Feature-diverse - Quality-only | +3.5 pp | [-1.1, +8.4] | 200 | 15 | 8 | 50 | 127 |
| Feature-diverse - Random | +3.5 pp | [-1.9, +9.3] | 200 | 24 | 26 | 41 | 109 |

## Overlooked-solution subset (n = 132)

Eligibility (fixed before test scoring): all verified solutions fall outside the raw
policy's top three legal moves. The denominator is identical for every method.

| Condition | Solved / n | Rate [95% CI] |
|---|---:|---:|
| Standard PUCT | 18/132 | 13.6% [8.2, 19.9] |
| Quality-only allocation | 15/132 | 11.4% [6.2, 17.2] |
| Random allocation | 21/132 | 15.9% [10.7, 21.5] |
| Feature-diverse allocation | 25/132 | 18.9% [12.3, 26.2] |

## Artifacts

- `outputs/laptop-main/report.html` - full illustrated report
- `outputs/laptop-main/case_studies.html` - printable case-study gallery
- `outputs/laptop-main/figures/` - standalone SVG + PNG + plotted-data JSON
- `outputs/laptop-main/case_studies/figures/` - standalone board SVGs
- `outputs/laptop-main/examples.pgn` - case-study PGNs
- `outputs/laptop-main/results.jsonl` - raw per-position rows

## Reproduce

```
python -m scripts.run_experiment --profile laptop --max-minutes 120 --seed 42 --run-id laptop-main
```