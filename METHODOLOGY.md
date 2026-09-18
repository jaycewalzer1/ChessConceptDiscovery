# Methodology

## 1. Repository audit

The starting point was commit `bad1482` ("Piece development metric"). Every defect below
was confirmed by reading the original files before any modification.

| File | Defect | Disposition |
| --- | --- | --- |
| `agents/alphazero/agent.py` | `from mcts import MCTS` — no such module anywhere in the repo; flat imports of `neural_network` / `chess_env` that the package layout never supported; `select_action` computed a policy, discarded it, then called `self.mcts.search(state, policy)`, which did not exist | **repaired**, class name and public surface kept |
| `agents/alphazero/neural_network.py` | policy head was `Dense(input_size, activation='softmax')`, tying a 768-wide flat input to a 768-way "policy"; `train()` was `pass`; `save_weights`/`load_weights` delegated to `self.save` / `self.load`, and `tf.keras.Model` has no `load` | **repaired**, `tf.keras.Model` base and two-head design kept |
| `agents/alphazero/training.py` | collected `(state, action, reward, next_state)` DQN transitions, not AlphaZero `(observation, policy target, outcome)` records | **repaired**, `TrainingLoop` name kept |
| `envs/chess_env.py` | `ChessEnv` defined **twice** in one file; the first subclassed `open_spiel.python.games.ChessGame` behind a gitlink with no `.gitmodules`; read `self.state.reward(self.state.current_player())` *after* terminal, where the player index is invalid; `calculate_development` used `board.has_castling_rights(chess.WHITE)` as a proxy for "has castled" | **replaced** with a single python-chess environment; `calculate_development` repaired to detect actual castling |
| `envs/chess_utils.py` | `encode_board` returned `np.zeros((8,8,12))`; `decode_action` and `get_valid_moves` were `pass` | **replaced** |
| `scripts/eval_agent.py`, `scripts/run_training.py` | imported `envs.chess.chess_env` (nonexistent path); mismatched call signatures | **repaired** |
| `scripts/run_training.py `, `scripts/setup.py  ` | trailing spaces in the filenames (confirmed with `ls -b`) | renamed via `git mv` |
| `open_spiel` | gitlink `160000 commit 2228e1c…` with no `.gitmodules`, so the tree could never be cloned complete | removed from the index |

**Reused:** the class names `AlphaZeroAgent`, `NeuralNetwork`, `TrainingLoop`, `ChessEnv`,
the two-head policy/value design, the `tf.keras.Model` base, the `envs/` + `agents/` +
`scripts/` organization, and the development heuristic's intent.

**Newly written:** `agents/alphazero/mcts.py`, `agents/alphazero/features.py`,
`agents/alphazero/allocators.py`, `envs/mate_solver.py`, `data/generate.py`,
`analysis/{stats,plots,case_studies,report}.py`, `scripts/run_experiment.py`,
`tests/test_core.py`.

**Environment substitution.** OpenSpiel was replaced by `python-chess` for the rules
layer. This is an environment substitution, not a replacement of the AlphaZero learner or
search, both of which are implemented here. `chess.engine` is never imported.

## 2. Encodings

**Actions.** A flat from/to/promotion encoding: `64 × 64 × 5 = 20480` actions
(`ENCODING_VERSION = "fromto-promo-v1"`). Promotion index 0 means "no promotion", and
knight/bishop/rook/queen promotions are four *distinct* actions, so underpromotions are
never collapsed. Squares are canonicalised with `chess.square_mirror` so the side to move
always plays "up the board". Round trips are unit-tested over six positions including
underpromotion fixtures.

**Observations.** `(8, 8, 20)` float planes (`OBS_VERSION = "obs-20plane-v1"`):
planes 0–5 own pieces, 6–11 opponent pieces, 12 a constant-one plane, 13–16 the four
castling rights, 17 the en-passant file, 18 the halfmove clock / 100, 19 a
twofold-repetition flag.

**The observation is not fully Markov.** It carries a twofold-repetition flag but no move
history, so a threefold-repetition claim is not recoverable from the tensor alone. This is
stated explicitly rather than glossed as "Markov". The *search* does not rely on the
tensor: `ChessEnv.clone()` copies the board **with its move stack**, and the search key is
`(transposition_key, halfmove_clock, is_repetition(2))`, so positions with different
repetition or draw status are never merged.

**Masking.** `legal_action_mask` produces a boolean vector over all 20480 actions.
Illegal actions are masked to `-1e9` *before* any softmax or log-softmax, so illegal moves
receive exactly zero probability and the cross-entropy stays numerically stable.

## 3. Network

A small residual convolutional trunk over the `(8,8,20)` observation
(`channels`/`blocks` configurable; the laptop profile uses 64 channels and 4 blocks,
5,576,661 parameters). Two heads: **policy logits** of width `NUM_ACTIONS` (not a softmax,
so masking is possible) and a `tanh`-bounded scalar **value for the player to move**. The
trunk output is exposed via `return_hidden=True` / `hidden_representation()`.

Training uses masked cross-entropy against the target distribution plus a **masked** MSE
value loss: each sample carries `v_weight ∈ {0,1}`, and unknown game-theoretic outcomes
get weight 0. A truncated game is therefore never silently labelled a draw.

Checkpoints store architecture metadata, weights, optimiser state, the NumPy RNG state,
and the encoding/observation versions. `load_weights` verifies the tensor count and every
individual shape and raises on mismatch.

> **Bug found by the test suite.** In Keras 3, `self.build((None,)+obs_shape)` does *not*
> build sublayers, so `self.weights` came back partial and `zip(self.weights, saved)`
> silently truncated — restored predictions diverged from saved ones. Fixed by forcing a
> real dummy forward pass in `__init__` and by adding the explicit length/shape checks.

## 4. Search

PUCT MCTS with per-node visit counts, priors, and value sums.

- **Value convention:** every node's value is from the perspective of *its own* player to
  move; the backup negates at every ply, and a child's contribution to the root player is
  `-child.value`.
- **Terminals:** a rules-terminal node gets its exact game value and is never re-evaluated
  by the network. A node that hits the ply limit is marked `is_truncated` and takes the
  network value — truncation is recorded separately from a real drawn terminal.
- **Instrumentation:** every search reports *actual* `simulations`, `nn_evals`,
  `nn_batches`, `terminal_expansions`, `cache_hits`, `env_transitions`, `peak_tree_size`,
  `truncations` and `elapsed_s`. Loop iterations are never reported as equivalent compute.
- Dirichlet root noise and temperature are used in self-play only and are off in all
  deterministic evaluation.

## 5. Oracle

`envs/mate_solver.py` implements an exact bounded forced-mate verifier by alpha-beta-free
recursive search with an explicit node budget:

- the attacker's choice is **existential** (some move suffices);
- the defender's replies are **universally** quantified (all legal replies must fail),
  and a defender stalemate — or any rules draw — refutes the mate claim;
- `solve_root_moves` returns the **complete set** of root moves achieving mate within the
  horizon;
- if the node budget is exhausted the status is `unknown`, and such tasks are excluded
  from verified scoring rather than guessed.

The oracle is never queried by any search agent at test time. It is used to build the
dataset, to grade results afterwards, and to produce the verified continuation and
refutation lines shown in the case studies.

## 6. Dataset

Two generators:

1. **Playout** — random-ish legal games (with a capture bias) are played out, and every
   visited position is classified by the verifier. Positions produced this way carry their
   full legal move history and are genuinely reachable from the start position.
2. **Endgame templates** — ten documented `(white_pieces, black_pieces)` material
   templates are placed at random, then a short legal prefix is played before
   classification. These are *constructed*, and the manifest records that. Note that
   `board.is_valid()` alone is **not** proof of historical reachability, so the
   `reachable_from_start` flag is recorded per task rather than assumed.

Deduplication is by exact FEN and by a piece-placement + side-to-move key. Splits are
**by source family** (one playout game or one endgame construction seed), never by
adjacent rows, and each split's SHA-256 hash is recorded.

**Solution labels are stored in separate files** (`*_labels.jsonl`) from the agent-facing
positions (`*_positions.jsonl`), and a test asserts that the agent-facing file contains no
solution field.

## 7. Conditions and budget accounting

All four conditions run on the same frozen checkpoint with the same total simulation
budget `N`.

1. **Standard PUCT** — plain search, move by root visit count.
2. **Quality-only** — screen with `S = 0.25 N` simulations, take the top-K by screening
   visit share, split the remaining `N − S` evenly, choose the candidate with the highest
   post-allocation root-player value.
3. **Random** — identical, but the candidate set is the top-1 anchor plus `K−1` uniform
   draws from the pool. The anchor is **pre-registered** in the module docstring so that
   conditions 2–4 share their first candidate and the contrast is purely *how alternatives
   are chosen*. Seeds are recorded in every row.
4. **Feature-diverse** — identical, but candidates are chosen greedily to maximise
   `normalised screening quality + λ · (minimum standardised feature distance to the
   already-selected candidates) / √F`.

The eligible pool is **every legal root move** (capped at 48, which never bound on this
benchmark; the median position has 40 legal moves), so plausible low-prior alternatives
are always reachable.

**Accounting.** One physical screening search may be reused across conditions 2–4, but it
is deep-copied and its **full logical cost** — simulations, network evaluations, batches,
environment transitions and wall time — is charged to each condition separately. Feature
computation time is added to condition 4. No condition inherits another condition's search
tree, and no condition ever sees an oracle result. Results are reported at matched *actual*
search work: simulations, network evaluations, and wall time side by side.

## 8. Descriptors

24 cheap deterministic descriptors computed from the **root player's** perspective over a
short continuation: material change by piece type, capture/check/promotion counts,
immediate capture/check/promotion flags, moved-piece value, pawn advancement, centre /
kingside / queenside occupancy change, own and opponent mobility change, king-ring attack
change on both sides, open-file change, minor-piece activity, a material-deficit
"sacrifice proxy", and the continuation length.

Continuations come from the *most-visited line in the screening tree*. When a candidate's
subtree was never visited, only the root move is available; in that case the descriptor
mode is recorded as **`successor`** and the resulting construct is named
**state-feature diversity**, not strategic-plan diversity. The mode is recorded in every
result row.

Descriptors exclude board hashes, random numbers and solution labels. Standardisation is
fitted on **train + dev descriptors only**, and the fitted mean/std are written into the
frozen protocol.

These distances are a working construct for this experiment. They are **not** an
established measure of creativity.

## 9. Protocol freeze

`protocol.json` is written after development calibration and before any test score is
computed. It records the research question, the operational definition, the primary and
secondary endpoints, the verdict rule, dataset manifests and split hashes, the checkpoint
hash, feature definitions and normalisation statistics, selection rules, budgets, seeds,
inclusion/exclusion criteria, the statistical plan and the example-selection rule. The
whole object is hashed with SHA-256 and timestamped. No hyper-parameter was adjusted on
test data and no seed was selected for a favourable outcome.

## 10. Statistics

- **Primary endpoint:** fraction of held-out test tasks where the selected root move is a
  member of the exactly-verified complete solution set.
- **Overlooked-solution recovery:** restricted to positions where *every* verified
  solution falls outside the **common raw-network-policy top three**. Because the raw
  policy is computed once per position from the frozen model with no search, this
  denominator is identical for every method by construction.
- Comparisons are **paired** on positions; repeated seeds are averaged within a position
  before comparison.
- Uncertainty is an absolute percentage-point difference with a 95% percentile bootstrap
  interval that **resamples source families**, not rows.

Internal value estimates, tree diversity, policy entropy, longer principal variations and
surprising mistakes are reported where relevant as *mechanism diagnostics only*, and are
never equated with successful creativity.

## 11. Honest labelling of the training procedure

The model is **not** trained tabula rasa. Stage 1 is supervised tactical bootstrapping on
exactly-verified forced-mate solutions from the training split (label-smoothed target over
legal moves, `v = 1.0`, `v_weight = 1.0` — a verified forced mate is a decisive win for the
side to move). Stage 2 is short genuine MCTS self-play fine-tuning with visit-count policy
targets and outcome value targets; truncated games keep `v_weight = 0`. Failure to find a
mate is never used as a draw, loss, or zero-value label.
