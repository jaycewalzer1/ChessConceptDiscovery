"""Required quantitative graphs, generated directly from the raw results.

NEW module.  Every figure is written as both SVG and PNG plus a ``.json``
sidecar containing the exact plotted numbers, so any claim in the report can be
traced back to data.  No mock values, no decorative charts, no 3-D, no
truncated rate axes.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.stats import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    aggregate_seeds,
    paired_difference,
    success_rate,
)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Distinct hatches so the figures stay readable in greyscale / print.
CONDITION_HATCH = {"standard": "", "quality": "//", "random": "..", "feature": "xx"}


def _save(fig, out_dir, name, data):
    os.makedirs(out_dir, exist_ok=True)
    svg = os.path.join(out_dir, name + ".svg")
    png = os.path.join(out_dir, name + ".png")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    with open(os.path.join(out_dir, name + ".json"), "w") as fh:
        json.dump(data, fh, indent=2)
    plt.close(fig)
    return {"svg": svg, "png": png, "data": os.path.join(out_dir, name + ".json")}


def _err(rate, lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return 0.0, 0.0
    return max(rate - lo, 0.0), max(hi - rate, 0.0)


# --------------------------------------------------------------------------- #
# 1. success vs search budget
# --------------------------------------------------------------------------- #
def plot_success_vs_budget(rows, summary, out_dir):
    budgets = summary["budgets"]
    conds = summary["conditions"]
    plotted = {"budgets": budgets, "series": {}}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    single = len(budgets) == 1
    if single:
        width = 0.8 / len(conds)
        for i, c in enumerate(conds):
            s = summary["by_budget"][str(budgets[0])].get(c)
            if not s:
                continue
            lo, hi = _err(s["rate"], s["ci_low"], s["ci_high"])
            ax.bar(i * width, s["rate"] * 100, width=width * 0.9,
                   color=CONDITION_COLORS[c], hatch=CONDITION_HATCH[c], edgecolor="black",
                   linewidth=0.6, label=f"{CONDITION_LABELS[c]} (n={s['n']})",
                   yerr=[[lo * 100], [hi * 100]], capsize=4, error_kw={"ecolor": "black"})
            plotted["series"][c] = [s]
        ax.set_xticks([])
        ax.set_xlabel(f"Single completed budget: {budgets[0]} simulations per position")
    else:
        for c in conds:
            xs, ys, elo, ehi, ns = [], [], [], [], []
            for b in budgets:
                s = summary["by_budget"][str(b)].get(c)
                if not s:
                    continue
                xs.append(b)
                ys.append(s["rate"] * 100)
                l, h = _err(s["rate"], s["ci_low"], s["ci_high"])
                elo.append(l * 100)
                ehi.append(h * 100)
                ns.append(s["n"])
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[elo, ehi], marker="o", capsize=4,
                        color=CONDITION_COLORS[c], label=f"{CONDITION_LABELS[c]} (n={ns[0]})",
                        linewidth=2)
            plotted["series"][c] = [
                {"budget": b, "rate": y / 100, "ci_low_pp": y - l, "ci_high_pp": y + h, "n": n}
                for b, y, l, h, n in zip(xs, ys, elo, ehi, ns)
            ]
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel("Nominal search budget (MCTS simulations per position)")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Task success rate (%)\nselected move in verified solution set")
    ax.set_title("Forced-mate task success vs search budget\n(held-out test set, 95% family-cluster bootstrap CI)")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_dir, "fig1_success_vs_budget", plotted)


# --------------------------------------------------------------------------- #
# 2. paired differences
# --------------------------------------------------------------------------- #
def plot_paired_differences(summary, out_dir):
    keys = [("feature_vs_standard", "vs Standard PUCT"),
            ("feature_vs_quality", "vs Quality-only"),
            ("feature_vs_random", "vs Random allocation")]
    labels, vals, los, his, ns = [], [], [], [], []
    for k, lab in keys:
        p = summary["paired"].get(k)
        if not p:
            continue
        labels.append(lab)
        vals.append(p["diff_pp"])
        los.append(0.0 if np.isnan(p["ci_low_pp"]) else p["diff_pp"] - p["ci_low_pp"])
        his.append(0.0 if np.isnan(p["ci_high_pp"]) else p["ci_high_pp"] - p["diff_pp"])
        ns.append(p["n_paired"])

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    if not labels:
        ax.text(0.5, 0.5, "No paired comparisons available", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_dir, "fig2_paired_differences", {"note": "empty"})

    y = np.arange(len(labels))
    ax.errorbar(vals, y, xerr=[los, his], fmt="o", color=CONDITION_COLORS["feature"],
                capsize=5, markersize=8, linewidth=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n(n={n} paired)" for l, n in zip(labels, ns)])
    ax.set_xlabel("Paired difference in success rate (percentage points)\npositive = feature-diverse allocation better")
    ax.set_title(f"Feature-diverse allocation vs controls at {summary['primary_budget']} simulations\n"
                 "95% family-cluster bootstrap CI")
    ax.invert_yaxis()
    return _save(fig, out_dir, "fig2_paired_differences", {
        "labels": labels, "diff_pp": vals, "ci_low_pp": [v - l for v, l in zip(vals, los)],
        "ci_high_pp": [v + h for v, h in zip(vals, his)], "n_paired": ns,
    })


# --------------------------------------------------------------------------- #
# 3. recovery of initially overlooked solutions
# --------------------------------------------------------------------------- #
def plot_overlooked_recovery(summary, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    denom = summary["overlooked_denominator"]
    data = {"denominator": denom, "conditions": {}}
    if denom == 0 or not summary["overlooked"]:
        ax.text(0.5, 0.5,
                "No test positions met the pre-registered eligibility rule\n"
                "(all verified solutions outside the raw policy's top 3).\n"
                "Denominator = 0; no recovery estimate is possible.",
                ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        ax.set_title("Recovery of initially overlooked solutions - EMPTY DATA")
        return _save(fig, out_dir, "fig3_overlooked_recovery", data)

    conds = [c for c in summary["conditions"] if c in summary["overlooked"]]
    xs = np.arange(len(conds))
    rates, elo, ehi = [], [], []
    for c in conds:
        s = summary["overlooked"][c]
        rates.append(s["rate"] * 100)
        l, h = _err(s["rate"], s["ci_low"], s["ci_high"])
        elo.append(l * 100)
        ehi.append(h * 100)
        data["conditions"][c] = s
    ax.bar(xs, rates, color=[CONDITION_COLORS[c] for c in conds],
           hatch=[CONDITION_HATCH[c] for c in conds], edgecolor="black", linewidth=0.7,
           yerr=[elo, ehi], capsize=5)
    for x, c, r in zip(xs, conds, rates):
        s = summary["overlooked"][c]
        ax.text(x, r + 2, f"{s['successes']:.0f}/{s['n']}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABELS[c].replace(" allocation", "\nallocation") for c in conds], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate on eligible subset (%)")
    ax.set_title("Recovery of initially overlooked solutions\n"
                 f"eligible subset: all verified solutions outside raw-policy top 3 (n = {denom})")
    return _save(fig, out_dir, "fig3_overlooked_recovery", data)


# --------------------------------------------------------------------------- #
# 4. success vs computation
# --------------------------------------------------------------------------- #
def plot_success_vs_compute(summary, out_dir, training_overhead=None):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    conds = [c for c in summary["conditions"] if c in summary["cost"]]
    b = summary["primary_budget"]
    data = {"primary_budget": b, "points": {}, "setup_overhead": training_overhead}
    for c in conds:
        s = summary["by_budget"][str(b)].get(c)
        cost = summary["cost"][c]
        if not s:
            continue
        for ax, xk, xl in ((axes[0], "mean_elapsed_s", "Mean search wall time per position (s)"),
                           (axes[1], "mean_nn_evals", "Mean network evaluations per position")):
            lo, hi = _err(s["rate"], s["ci_low"], s["ci_high"])
            ax.errorbar([cost[xk]], [s["rate"] * 100], yerr=[[lo * 100], [hi * 100]],
                        marker="o", markersize=10, capsize=4, color=CONDITION_COLORS[c],
                        label=CONDITION_LABELS[c], linestyle="none")
            ax.set_xlabel(xl)
        data["points"][c] = {"rate": s["rate"], "n": s["n"], **cost}
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.set_ylabel("Task success rate (%)")
    axes[0].legend(fontsize=8, loc="best")
    note = ""
    if training_overhead:
        note = (f"\nOne-off setup excluded from these axes: data generation "
                f"{training_overhead.get('data_s', 0):.0f} s, training "
                f"{training_overhead.get('train_s', 0):.0f} s")
    fig.suptitle(f"Success vs actual per-position computation at {b} nominal simulations{note}", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_success_vs_compute", data)


# --------------------------------------------------------------------------- #
# 5. win / both / neither
# --------------------------------------------------------------------------- #
def plot_win_tie_loss(summary, out_dir):
    keys = [("feature_vs_standard", "Standard PUCT"),
            ("feature_vs_quality", "Quality-only"),
            ("feature_vs_random", "Random")]
    rows = [(lab, summary["paired"][k]) for k, lab in keys if k in summary["paired"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    if not rows:
        ax.text(0.5, 0.5, "No paired comparisons available", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_dir, "fig5_win_tie_loss", {"note": "empty"})

    cats = ["feature only", "both solved", "both failed", "baseline only"]
    colors = ["#55A868", "#8FBC8F", "#C0C0C0", "#4C72B0"]
    hatches = ["xx", "", "..", "//"]
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    data = {}
    for i, (cat, col, hh) in enumerate(zip(cats, colors, hatches)):
        key = ["a_only", "both_solved", "both_failed", "b_only"][i]
        vals = np.array([r[1][key] for r in rows], dtype=float)
        ax.barh(y, vals, left=left, color=col, hatch=hh, edgecolor="black",
                linewidth=0.6, label=cat)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(l + v / 2, yi, f"{int(v)}", ha="center", va="center", fontsize=9)
        left += vals
    for lab, p in rows:
        data[lab] = {k: p[k] for k in ["a_only", "both_solved", "both_failed", "b_only", "n_paired"]}
    ax.set_yticks(y)
    ax.set_yticklabels([f"vs {lab}\n(n={p['n_paired']})" for lab, p in rows])
    ax.set_xlabel("Number of held-out test positions")
    ax.set_title(f"Per-position outcome breakdown at {summary['primary_budget']} simulations\n"
                 "(both-solved and both-failed shown separately, never merged)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    ax.invert_yaxis()
    return _save(fig, out_dir, "fig5_win_tie_loss", data)


# --------------------------------------------------------------------------- #
# 6. case-level candidate rank / probability
# --------------------------------------------------------------------------- #
def plot_candidate_ranks(rows, summary, out_dir, max_cases=40):
    b = summary["primary_budget"]
    feat = {r["task_id"]: r for r in rows if r["condition"] == "feature" and r["budget"] == b}
    std = {r["task_id"]: r for r in rows if r["condition"] == "standard" and r["budget"] == b}
    shared = sorted(set(feat) & set(std))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    if not shared:
        ax.text(0.5, 0.5, "No shared positions available", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_dir, "fig6_candidate_ranks", {"note": "empty"})

    # Deterministic selection: positions where the two methods disagree first,
    # then remaining positions in task-id order.
    disagree = [t for t in shared if feat[t]["chosen_move_uci"] != std[t]["chosen_move_uci"]]
    agree = [t for t in shared if t not in set(disagree)]
    picked = (disagree + agree)[:max_cases]

    xs = np.arange(len(picked))
    data = {"selection_rule": "disagreements first, then task-id order", "cases": []}
    for i, t in enumerate(picked):
        f, s = feat[t], std[t]
        ax.scatter(i - 0.12, s["policy_rank_of_chosen"], marker="s", s=52,
                   color=CONDITION_COLORS["standard"], edgecolor="black", linewidth=0.5,
                   zorder=3, label="Standard PUCT choice" if i == 0 else None)
        ax.scatter(i + 0.12, f["policy_rank_of_chosen"], marker="D", s=52,
                   color=CONDITION_COLORS["feature"], edgecolor="black", linewidth=0.5,
                   zorder=3, label="Feature-diverse choice" if i == 0 else None)
        if f["best_solution_policy_rank"] is not None:
            ax.scatter(i, f["best_solution_policy_rank"], marker="*", s=95, color="#C44E52",
                       zorder=4, label="Best verified solution" if i == 0 else None)
        for dx, r, ok in ((-0.12, s, s["correct"]), (0.12, f, f["correct"])):
            if ok:
                ax.scatter(i + dx, r["policy_rank_of_chosen"], marker="o", s=150,
                           facecolors="none", edgecolors="#2E7D32", linewidth=1.6, zorder=5,
                           label="verified solved" if (i == 0 and dx > 0) else None)
        data["cases"].append({
            "task_id": t,
            "standard_rank": s["policy_rank_of_chosen"], "standard_prob": s["policy_prob_of_chosen"],
            "standard_correct": s["correct"],
            "feature_rank": f["policy_rank_of_chosen"], "feature_prob": f["policy_prob_of_chosen"],
            "feature_correct": f["correct"],
            "best_solution_rank": f["best_solution_policy_rank"],
        })
    ax.set_xticks(xs)
    ax.set_xticklabels([t for t in picked], rotation=90, fontsize=6)
    ax.set_yscale("log")
    ax.set_ylabel("Rank under the RAW network policy\n(1 = most likely; model estimate, not verification)")
    ax.set_xlabel("Test position id (disagreements shown first)")
    ax.set_title("Case-level view: initial policy rank of the selected move vs the verified solution\n"
                 "green ring = move verified correct by the exact oracle")
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_candidate_ranks", data)


# --------------------------------------------------------------------------- #
# 7. training curve (development / training diagnostics, clearly separated)
# --------------------------------------------------------------------------- #
def plot_training(training_summary, out_dir):
    hist = (training_summary or {}).get("bootstrap", {}).get("history", [])
    sp = (training_summary or {}).get("selfplay", {}).get("iterations", [])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    data = {"bootstrap": hist, "selfplay": sp}
    if hist:
        ep = [h["epoch"] + 1 for h in hist]
        axes[0].plot(ep, [h["policy_loss"] for h in hist], marker="o", label="policy loss")
        axes[0].plot(ep, [h["value_loss"] for h in hist], marker="s", label="value loss")
        axes[0].set_xlabel("Bootstrap epoch (TRAINING split only)")
        axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "no bootstrap history", ha="center", va="center")
    axes[0].set_title("Stage 1: supervised tactical bootstrap")
    if sp and any(i.get("loss") is not None for i in sp):
        it = [i["iteration"] + 1 for i in sp if i.get("loss") is not None]
        axes[1].plot(it, [i["loss"] for i in sp if i.get("loss") is not None], marker="o", color="#937860")
        axes[1].set_xlabel("Self-play iteration")
        axes[1].set_ylabel("Total loss")
    else:
        axes[1].text(0.5, 0.5, "self-play stage did not run\nor produced no updates",
                     ha="center", va="center")
    axes[1].set_title("Stage 2: AlphaZero-style self-play fine-tuning")
    fig.suptitle("Training diagnostics (NOT held-out test performance)", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_training", data)


def plot_dev_calibration(dev_summary, out_dir):
    """Development-set grid over the diversity weight (selection happened here)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    grid = (dev_summary or {}).get("diversity_grid", [])
    if not grid:
        ax.text(0.5, 0.5, "no development grid recorded", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_dir, "fig8_dev_calibration", {"note": "empty"})
    lams = [g["diversity_weight"] for g in grid]
    rates = [g["success_rate"] * 100 for g in grid]
    ax.plot(lams, rates, marker="o", color=CONDITION_COLORS["feature"], label="feature-diverse (dev)")
    base = (dev_summary or {}).get("standard_success_rate")
    if base is not None:
        ax.axhline(base * 100, color=CONDITION_COLORS["standard"], linestyle="--",
                   label="standard PUCT (dev)")
    chosen = (dev_summary or {}).get("chosen_diversity_weight")
    if chosen is not None:
        ax.axvline(chosen, color="black", linestyle=":", label=f"frozen lambda = {chosen}")
    ax.set_xlabel("Diversity weight lambda")
    ax.set_ylabel("DEVELOPMENT success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Development calibration (n = {dev_summary.get('n_dev', 0)} dev positions)\n"
                 "hyper-parameters were frozen from this grid before any test scoring")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "fig8_dev_calibration", dev_summary)


def generate_all(rows, summary, out_dir, training_summary=None, dev_summary=None,
                 training_overhead=None):
    figs = {}
    figs["fig1_success_vs_budget"] = plot_success_vs_budget(rows, summary, out_dir)
    figs["fig2_paired_differences"] = plot_paired_differences(summary, out_dir)
    figs["fig3_overlooked_recovery"] = plot_overlooked_recovery(summary, out_dir)
    figs["fig4_success_vs_compute"] = plot_success_vs_compute(summary, out_dir, training_overhead)
    figs["fig5_win_tie_loss"] = plot_win_tie_loss(summary, out_dir)
    figs["fig6_candidate_ranks"] = plot_candidate_ranks(rows, summary, out_dir)
    figs["fig7_training"] = plot_training(training_summary, out_dir)
    figs["fig8_dev_calibration"] = plot_dev_calibration(dev_summary, out_dir)
    return figs
