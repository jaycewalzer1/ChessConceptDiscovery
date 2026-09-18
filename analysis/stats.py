"""Uncertainty-aware analysis of the raw per-position results.

NEW module.  Every number in the report is derived from ``results.jsonl`` by
this code; nothing is hand-entered.

Clustering
----------
Positions harvested from the same source game / construction seed share a
``family`` and are **not** independent.  Confidence intervals therefore come
from a cluster bootstrap that resamples *families* with replacement.  Repeated
stochastic seeds for the same (position, condition, budget) are averaged
*within* the position before any resampling, so seeds never inflate the
effective sample size.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

CONDITION_COLORS = {
    "standard": "#4C72B0",
    "quality": "#DD8452",
    "random": "#937860",
    "feature": "#55A868",
}
CONDITION_LABELS = {
    "standard": "Standard PUCT",
    "quality": "Quality-only allocation",
    "random": "Random allocation",
    "feature": "Feature-diverse allocation",
}


def load_results(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_seeds(rows, budget=None):
    """Collapse repeated seeds into one value per (task_id, condition, budget).

    Returns ``{(condition, budget): {task_id: {...}}}`` with ``correct`` as the
    mean success over seeds (0/1 for deterministic conditions).
    """
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if budget is not None and r["budget"] != budget:
            continue
        buckets[(r["condition"], r["budget"])][r["task_id"]].append(r)
    out = {}
    for key, per_task in buckets.items():
        agg = {}
        for tid, rs in per_task.items():
            agg[tid] = {
                "task_id": tid,
                "family": rs[0]["family"],
                "correct": float(np.mean([r["correct"] for r in rs])),
                "n_seeds": len(rs),
                "elapsed_s": float(np.mean([r["stats"]["elapsed_s"] for r in rs])),
                "nn_evals": float(np.mean([r["stats"]["nn_evals"] for r in rs])),
                "simulations": float(np.mean([r["stats"]["simulations"] for r in rs])),
                "overlooked_eligible": bool(rs[0]["overlooked_eligible"]),
            }
        out[key] = agg
    return out


def _cluster_bootstrap(values_by_family, n_boot=5000, seed=0, stat=np.mean):
    """Bootstrap a statistic by resampling families with replacement."""
    fams = list(values_by_family)
    if not fams:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    flat = np.concatenate([np.asarray(values_by_family[f], dtype=float) for f in fams])
    point = float(stat(flat))
    if len(fams) < 2:
        return point, float("nan"), float("nan")
    boots = np.empty(n_boot)
    arrays = [np.asarray(values_by_family[f], dtype=float) for f in fams]
    idx_range = len(fams)
    for b in range(n_boot):
        pick = rng.integers(0, idx_range, size=idx_range)
        boots[b] = stat(np.concatenate([arrays[i] for i in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def success_rate(agg_task_map, n_boot=5000, seed=0, subset=None):
    """Success rate with a family-cluster bootstrap 95% CI."""
    by_fam = defaultdict(list)
    n = 0
    for tid, v in agg_task_map.items():
        if subset is not None and tid not in subset:
            continue
        by_fam[v["family"]].append(v["correct"])
        n += 1
    point, lo, hi = _cluster_bootstrap(by_fam, n_boot=n_boot, seed=seed)
    return {
        "rate": point, "ci_low": lo, "ci_high": hi,
        "n": n, "successes": float(sum(v["correct"] for tid, v in agg_task_map.items()
                                       if subset is None or tid in subset)),
        "n_families": len(by_fam),
    }


def paired_difference(agg_a, agg_b, n_boot=5000, seed=0, subset=None):
    """Paired difference (a - b) in percentage points on shared positions."""
    shared = set(agg_a) & set(agg_b)
    if subset is not None:
        shared &= set(subset)
    by_fam = defaultdict(list)
    wins = losses = both = neither = 0
    for tid in shared:
        d = agg_a[tid]["correct"] - agg_b[tid]["correct"]
        by_fam[agg_a[tid]["family"]].append(d)
        ca, cb = agg_a[tid]["correct"], agg_b[tid]["correct"]
        if ca > cb:
            wins += 1
        elif cb > ca:
            losses += 1
        elif ca >= 0.5:
            both += 1
        else:
            neither += 1
    point, lo, hi = _cluster_bootstrap(by_fam, n_boot=n_boot, seed=seed)
    return {
        "diff_pp": point * 100.0,
        "ci_low_pp": lo * 100.0,
        "ci_high_pp": hi * 100.0,
        "n_paired": len(shared),
        "n_families": len(by_fam),
        "a_only": wins,
        "b_only": losses,
        "both_solved": both,
        "both_failed": neither,
    }


def overlooked_subset(rows):
    """Task ids where ALL verified solutions fall outside the raw-policy top 3.

    The eligibility rule is defined in the frozen protocol from the COMMON raw
    network policy, so the denominator is identical for every method.
    """
    seen = {}
    for r in rows:
        seen.setdefault(r["task_id"], r["overlooked_eligible"])
    return {tid for tid, ok in seen.items() if ok}


def build_summary(rows, protocol, n_boot=5000, seed=0):
    budgets = sorted({r["budget"] for r in rows})
    conditions = protocol["conditions"]
    primary_budget = protocol["primary_budget"]
    agg = aggregate_seeds(rows)
    overlooked = overlooked_subset(rows)

    summary = {
        "budgets": budgets,
        "conditions": conditions,
        "primary_budget": primary_budget,
        "n_test_positions": len({r["task_id"] for r in rows}),
        "n_families": len({r["family"] for r in rows}),
        "overlooked_denominator": len(overlooked),
        "by_budget": {},
        "paired": {},
        "overlooked": {},
        "cost": {},
        "seeds_per_condition": {},
    }

    for b in budgets:
        summary["by_budget"][str(b)] = {}
        for c in conditions:
            m = agg.get((c, b))
            if not m:
                continue
            summary["by_budget"][str(b)][c] = success_rate(m, n_boot=n_boot, seed=seed)
            summary["seeds_per_condition"][f"{c}@{b}"] = int(
                np.mean([v["n_seeds"] for v in m.values()])
            ) if m else 0

    base_map = agg.get(("feature", primary_budget), {})
    for other in ["standard", "quality", "random"]:
        om = agg.get((other, primary_budget), {})
        if base_map and om:
            summary["paired"][f"feature_vs_{other}"] = paired_difference(
                base_map, om, n_boot=n_boot, seed=seed
            )
            summary["paired"][f"feature_vs_{other}_overlooked"] = paired_difference(
                base_map, om, n_boot=n_boot, seed=seed, subset=overlooked
            )

    for c in conditions:
        m = agg.get((c, primary_budget))
        if not m:
            continue
        summary["overlooked"][c] = success_rate(m, n_boot=n_boot, seed=seed, subset=overlooked)
        vals = list(m.values())
        summary["cost"][c] = {
            "mean_elapsed_s": float(np.mean([v["elapsed_s"] for v in vals])),
            "mean_nn_evals": float(np.mean([v["nn_evals"] for v in vals])),
            "mean_simulations": float(np.mean([v["simulations"] for v in vals])),
            "total_elapsed_s": float(np.sum([v["elapsed_s"] for v in vals])),
        }

    # policy-rank diagnostics (identical across conditions by construction)
    ranks = [r["best_solution_policy_rank"] for r in rows if r["best_solution_policy_rank"] is not None]
    summary["policy_diagnostics"] = {
        "median_best_solution_rank": float(np.median(ranks)) if ranks else None,
        "frac_solution_in_top1": float(np.mean([r == 1 for r in ranks])) if ranks else None,
        "frac_solution_in_top3": float(np.mean([r <= 3 for r in ranks])) if ranks else None,
    }

    # exclusions
    summary["exclusions"] = {
        "invalid_moves": sum(1 for r in rows if not r.get("move_was_legal", True)),
        "solver_unknown": sum(1 for r in rows if r.get("solver_status") != "ok"),
    }
    return summary, agg, overlooked


def verdict(summary):
    """Deterministic verdict rule, fixed in the protocol before test scoring."""
    p = summary["paired"].get("feature_vs_standard")
    if not p or p["n_paired"] == 0:
        return "infrastructure-only", "No paired test comparisons were produced."
    lo, hi, d = p["ci_low_pp"], p["ci_high_pp"], p["diff_pp"]
    if np.isnan(lo):
        return "inconclusive", "Too few independent families for an interval estimate."
    if lo > 0:
        return "positive", (f"Feature-diverse allocation beat standard PUCT by "
                            f"{d:+.1f} pp (95% CI {lo:+.1f} to {hi:+.1f}), interval excludes zero.")
    if hi < 0:
        return "negative", (f"Feature-diverse allocation was WORSE than standard PUCT by "
                            f"{d:+.1f} pp (95% CI {lo:+.1f} to {hi:+.1f}), interval excludes zero.")
    return "inconclusive", (f"Observed difference {d:+.1f} pp with 95% CI "
                            f"{lo:+.1f} to {hi:+.1f} pp; the interval contains zero.")
