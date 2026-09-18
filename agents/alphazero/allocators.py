"""The four experimental conditions (search-budget allocation strategies).

NEW module.

All four conditions use the SAME frozen model, environment, action space,
observation encoding and nominal simulation budget ``N``:

1. ``standard``  -- ordinary PUCT: ``N`` simulations from the root, move chosen
   by root visit counts.  No diversity intervention.
2. ``quality``   -- screening search of ``S = round(screen_frac * N)``
   simulations, then the remaining ``R = N - S`` simulations split evenly over
   the top-``K`` candidates by screening quality.
3. ``random``    -- identical to (2) except the ``K`` candidates are the
   top-1 quality move plus ``K-1`` moves sampled uniformly (with a recorded
   seed) from the remaining eligible pool.
4. ``feature``   -- identical to (2) except the ``K`` candidates are chosen
   greedily to maximise ``normalised_quality + lambda * min-distance`` in the
   standardised chess-feature space.

Everything other than *which candidates receive the extra simulations* is
identical between conditions 2-4: same screening implementation, same eligible
pool, same candidate count, same per-candidate simulation split, same final
move-selection rule.

BUDGET ACCOUNTING
-----------------
The screening search is executed once per position and its search tree is
*deep-copied* for each of conditions 2-4; the full screening cost (simulations,
network evaluations and wall time) is charged to every condition that needs it.
No condition inherits another condition's statistics, cache or subtree state.
Feature computation for condition 4 is timed and added to that condition's
elapsed time even though it only reads states the screening search already
produced.

RANDOM-ALLOCATION NOTE (pre-registered)
---------------------------------------
Condition 3 anchors its first candidate on the top screening move, exactly like
conditions 2 and 4, and samples only the *remaining* ``K-1`` candidates.  This
makes the contrast between conditions 2, 3 and 4 purely about **how the
alternatives are chosen**, rather than about whether the best-looking move is
considered at all.  This choice was fixed before any test data was opened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from agents.alphazero.features import FEATURE_NAMES, continuation_descriptor
from agents.alphazero.mcts import MCTS, Node, SearchStats

CONDITIONS = ["standard", "quality", "random", "feature"]


@dataclass
class AllocationConfig:
    simulations: int = 400
    screen_frac: float = 0.25
    num_candidates: int = 4
    pool_cap: int = 48
    quality_floor: float = 0.0
    diversity_weight: float = 0.5
    continuation_len: int = 4
    c_puct: float = 1.5
    max_plies: int = 40

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class AllocationResult:
    condition: str
    action: int
    move_uci: str
    stats: SearchStats
    candidates: list = field(default_factory=list)  # list of dicts
    screen_quality: dict = field(default_factory=dict)  # action -> normalised quality
    root_visits: dict = field(default_factory=dict)
    final_scores: dict = field(default_factory=dict)
    feature_mode: str = "n/a"
    notes: str = ""
    pv_san: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# tree utilities
# --------------------------------------------------------------------------- #
def clone_tree(node: Node) -> Node:
    new = Node(prior=node.prior)
    new.visit_count = node.visit_count
    new.value_sum = node.value_sum
    new.expanded = node.expanded
    new.terminal_value = node.terminal_value
    new.is_truncated = node.is_truncated
    new.children = {a: clone_tree(c) for a, c in node.children.items()}
    return new


def most_visited_continuation(env, root: Node, action: int, max_len: int):
    """Extract the most-visited line under ``action`` from the screening tree.

    Returns ``(moves, mode)`` where ``mode`` is ``"continuation"`` when at
    least one *searched* reply was available and ``"successor"`` when the
    candidate subtree was unvisited so only the root move could be used.
    """
    work = env.clone()
    move = work.decode(action)
    if move not in work.board.legal_moves:
        return [], "successor"
    moves = [move]
    work.push_move(move)
    node = root.children.get(action)
    mode = "successor"
    depth = 1
    while node is not None and node.children and depth < max_len:
        best_a, best_c = None, None
        for a, c in node.children.items():
            if best_c is None or c.visit_count > best_c.visit_count:
                best_a, best_c = a, c
        if best_c is None or best_c.visit_count == 0:
            break
        nxt = work.decode(best_a)
        if nxt not in work.board.legal_moves:
            break
        moves.append(nxt)
        work.push_move(nxt)
        mode = "continuation"
        node = best_c
        depth += 1
        if work.board.is_game_over(claim_draw=False):
            break
    return moves, mode


# --------------------------------------------------------------------------- #
# screening
# --------------------------------------------------------------------------- #
def run_screening(model, env, cfg: AllocationConfig):
    """Run the shared screening search.  Returns ``(root, stats, wall_time)``."""
    mcts = MCTS(model, c_puct=cfg.c_puct, max_plies=cfg.max_plies)
    n_screen = max(1, int(round(cfg.screen_frac * cfg.simulations)))
    t0 = time.perf_counter()
    root, stats = mcts.run(env, n_screen)
    wall = time.perf_counter() - t0
    stats.elapsed_s = wall
    return root, stats, wall


def eligible_pool(env, root: Node, cfg: AllocationConfig):
    """Eligible candidate pool: every legal root move, capped at ``pool_cap``.

    Ranking for the cap is (screening visits, prior) descending, so the cap --
    when it binds at all -- drops only the least-explored, lowest-prior moves.
    In the benchmark used here the cap was never binding.
    """
    items = [(a, c.visit_count, c.prior) for a, c in root.children.items()]
    items.sort(key=lambda t: (-t[1], -t[2], t[0]))
    return [a for a, _, _ in items[: cfg.pool_cap]]


def screening_quality(root: Node, pool):
    """Normalised screening quality in [0, 1] for each pool member.

    Score = visit share, with the prior used as a strictly smaller tie-break so
    that completely unvisited moves are still ordered sensibly.
    """
    total = max(sum(root.children[a].visit_count for a in pool), 1)
    raw = {}
    for a in pool:
        c = root.children[a]
        raw[a] = c.visit_count / total + 1e-3 * c.prior
    lo, hi = min(raw.values()), max(raw.values())
    span = hi - lo
    if span <= 0:
        return {a: 1.0 for a in pool}
    return {a: (v - lo) / span for a, v in raw.items()}


# --------------------------------------------------------------------------- #
# candidate selection rules
# --------------------------------------------------------------------------- #
def select_quality(pool, quality, k):
    order = sorted(pool, key=lambda a: (-quality[a], a))
    return order[:k]


def select_random(pool, quality, k, rng):
    anchor = sorted(pool, key=lambda a: (-quality[a], a))[0]
    rest = [a for a in pool if a != anchor]
    if len(rest) <= k - 1:
        return [anchor] + rest
    chosen = rng.choice(np.array(rest), size=k - 1, replace=False)
    return [anchor] + [int(a) for a in chosen]


def select_feature_diverse(pool, quality, descriptors, k, lam):
    """Greedy quality + min-distance selection in standardised feature space."""
    d = np.sqrt(max(descriptors.shape[1], 1))
    selected = [sorted(pool, key=lambda a: (-quality[a], a))[0]]
    idx = {a: i for i, a in enumerate(pool)}
    while len(selected) < min(k, len(pool)):
        best, best_score = None, -np.inf
        sel_vecs = descriptors[[idx[a] for a in selected]]
        for a in pool:
            if a in selected:
                continue
            v = descriptors[idx[a]]
            dist = float(np.min(np.linalg.norm(sel_vecs - v, axis=1))) / d
            score = quality[a] + lam * dist
            if score > best_score or (score == best_score and (best is None or a < best)):
                best, best_score = a, score
        if best is None:
            break
        selected.append(best)
    return selected


# --------------------------------------------------------------------------- #
# the allocation runner
# --------------------------------------------------------------------------- #
def _allocate_and_finalise(model, env, cfg, screen_root, screen_stats, screen_wall, candidates, condition, extra_time=0.0, feature_mode="n/a", notes=""):
    """Spend the remaining simulations on ``candidates`` and pick the final move.

    Identical for conditions 2-4.  The remaining budget ``R = N - S`` is split
    as evenly as possible over the candidates (earlier candidates absorb the
    remainder deterministically).  The final move is the candidate with the
    highest post-allocation root-player value estimate, tie-broken by total
    visits then action id.  Every candidate receives at least one simulation,
    so there are no unvisited candidates at selection time.
    """
    mcts = MCTS(model, c_puct=cfg.c_puct, max_plies=cfg.max_plies)
    root = clone_tree(screen_root)
    remaining = max(cfg.simulations - screen_stats.simulations, len(candidates))
    per = remaining // len(candidates)
    extra = remaining - per * len(candidates)

    t0 = time.perf_counter()
    alloc_stats = SearchStats()
    final_scores, cand_records = {}, []
    for i, a in enumerate(candidates):
        n = per + (1 if i < extra else 0)
        child_env = env.clone()
        child_env.step(a)
        child = root.children[a]
        _, st = mcts.run(child_env, n, root=child, stats=SearchStats())
        alloc_stats = alloc_stats.merge(st)
        # child.value is the value for the CHILD's mover (the opponent),
        # so the root player's estimate is its negation.
        q_root = -child.value
        final_scores[a] = q_root
        cand_records.append(
            {
                "action": int(a),
                "move_uci": env.decode(a).uci(),
                "san": env.san(env.decode(a)),
                "screen_visits": int(screen_root.children[a].visit_count),
                "alloc_sims": int(n),
                "post_visits": int(child.visit_count),
                "q_root": float(q_root),
            }
        )
    alloc_wall = time.perf_counter() - t0

    best = max(candidates, key=lambda a: (final_scores[a], root.children[a].visit_count, -a))
    pv_san, _ = mcts.principal_variation(env, root, max_len=6)

    stats = SearchStats(
        simulations=screen_stats.simulations + alloc_stats.simulations,
        nn_evals=screen_stats.nn_evals + alloc_stats.nn_evals,
        nn_batches=screen_stats.nn_batches + alloc_stats.nn_batches,
        terminal_expansions=screen_stats.terminal_expansions + alloc_stats.terminal_expansions,
        cache_hits=screen_stats.cache_hits + alloc_stats.cache_hits,
        env_transitions=screen_stats.env_transitions + alloc_stats.env_transitions,
        peak_tree_size=max(screen_stats.peak_tree_size, alloc_stats.peak_tree_size),
        elapsed_s=screen_wall + alloc_wall + extra_time,
        truncations=screen_stats.truncations + alloc_stats.truncations,
    )
    return AllocationResult(
        condition=condition,
        action=int(best),
        move_uci=env.decode(best).uci(),
        stats=stats,
        candidates=cand_records,
        root_visits={int(a): int(c.visit_count) for a, c in root.children.items()},
        final_scores={int(a): float(v) for a, v in final_scores.items()},
        feature_mode=feature_mode,
        notes=notes,
        pv_san=pv_san,
    )


def run_condition(condition, model, env, cfg: AllocationConfig, scaler=None, seed: int = 0,
                  screen_cache=None):
    """Run one condition on one position and return an :class:`AllocationResult`.

    ``screen_cache`` may hold a previously computed ``(root, stats, wall)``
    triple for this position.  It is *deep-copied* and its full cost is charged
    to this condition (see module docstring).
    """
    if condition == "standard":
        mcts = MCTS(model, c_puct=cfg.c_puct, max_plies=cfg.max_plies)
        t0 = time.perf_counter()
        root, stats = mcts.run(env, cfg.simulations)
        stats.elapsed_s = time.perf_counter() - t0
        action = MCTS.select_move(root, temperature=0.0)
        pv_san, _ = mcts.principal_variation(env, root, max_len=6)
        return AllocationResult(
            condition=condition,
            action=int(action),
            move_uci=env.decode(action).uci(),
            stats=stats,
            root_visits={int(a): int(c.visit_count) for a, c in root.children.items()},
            final_scores={int(a): float(-c.value) for a, c in root.children.items()},
            notes="plain PUCT, move by root visit count",
            pv_san=pv_san,
        )

    if screen_cache is None:
        screen_root, screen_stats, screen_wall = run_screening(model, env, cfg)
    else:
        cached_root, cached_stats, screen_wall = screen_cache
        screen_root = clone_tree(cached_root)
        screen_stats = SearchStats(**cached_stats.as_dict())

    pool = eligible_pool(env, screen_root, cfg)
    quality = screening_quality(screen_root, pool)
    if cfg.quality_floor > 0:
        kept = [a for a in pool if quality[a] >= cfg.quality_floor]
        if len(kept) >= cfg.num_candidates:
            pool = kept
    k = min(cfg.num_candidates, len(pool))
    rng = np.random.default_rng(seed)

    extra_time, feature_mode, notes = 0.0, "n/a", ""
    if condition == "quality":
        candidates = select_quality(pool, quality, k)
        notes = "top-K by screening visit share"
    elif condition == "random":
        candidates = select_random(pool, quality, k, rng)
        notes = f"anchor + {k-1} uniform samples, seed={seed}"
    elif condition == "feature":
        t0 = time.perf_counter()
        descs, modes = [], []
        for a in pool:
            moves, mode = most_visited_continuation(env, screen_root, a, cfg.continuation_len)
            descs.append(continuation_descriptor(env.board, moves, mode=mode))
            modes.append(mode)
        D = np.asarray(descs, dtype=np.float64)
        if scaler is not None:
            D = scaler.transform(D)
        feature_mode = "continuation" if any(m == "continuation" for m in modes) else "successor"
        candidates = select_feature_diverse(pool, quality, D, k, cfg.diversity_weight)
        extra_time = time.perf_counter() - t0
        notes = f"greedy quality + {cfg.diversity_weight} * min-distance, mode={feature_mode}"
    else:
        raise ValueError(f"unknown condition {condition!r}")

    res = _allocate_and_finalise(
        model, env, cfg, screen_root, screen_stats, screen_wall, candidates,
        condition, extra_time=extra_time, feature_mode=feature_mode, notes=notes,
    )
    res.screen_quality = {int(a): float(quality[a]) for a in pool}
    return res
