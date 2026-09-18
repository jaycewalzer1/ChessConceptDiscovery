"""Illustrated chess case studies.

NEW module.  Board diagrams are rendered from the **real recorded positions**
with ``chess.svg`` (deterministic vector drawing).  No image generation model is
used anywhere.

Selection rule (deterministic, recorded in the output index)
------------------------------------------------------------
Categories are filled in this fixed order, each from the pool of shared test
positions at the primary budget, sorted by ``task_id``:

  1. ``feature_win``     feature-diverse solved, standard PUCT failed
  2. ``standard_win``    standard PUCT solved, feature-diverse failed
  3. ``both_solved``     both solved
  4. ``both_failed``     neither solved
  5. ``vs_random``       feature-diverse solved and random allocation failed

Up to ``per_category`` cases are taken from each, capped at ``max_cases`` in
total.  An empty category is reported explicitly rather than back-filled.
"""

from __future__ import annotations

import html
import json
import os

import chess
import chess.pgn
import chess.svg

from envs.mate_solver import refutation_line, winning_continuation

CATEGORY_TITLES = {
    "feature_win": "Feature-diverse allocation solved it, standard PUCT did not",
    "standard_win": "Standard PUCT solved it, feature-diverse allocation did not",
    "both_solved": "Both methods solved it",
    "both_failed": "Neither method solved it",
    "vs_random": "Feature-diverse allocation solved it, random allocation did not",
}


def _svg(board, size=330, arrows=(), fill=None, orientation=chess.WHITE, check=None):
    return chess.svg.board(
        board,
        size=size,
        coordinates=True,
        arrows=list(arrows),
        fill=fill or {},
        orientation=orientation,
        check=check,
    )


def _arrow(move, color):
    return chess.svg.Arrow(move.from_square, move.to_square, color=color)


def select_cases(rows, summary, per_category=2, max_cases=10):
    b = summary["primary_budget"]
    by_cond = {}
    for r in rows:
        if r["budget"] != b:
            continue
        by_cond.setdefault(r["condition"], {}).setdefault(r["task_id"], r)
    feat = by_cond.get("feature", {})
    std = by_cond.get("standard", {})
    rnd = by_cond.get("random", {})
    qual = by_cond.get("quality", {})
    shared = sorted(set(feat) & set(std))

    cats = {k: [] for k in CATEGORY_TITLES}
    for t in shared:
        f, s = feat[t], std[t]
        if f["correct"] and not s["correct"]:
            cats["feature_win"].append(t)
        elif s["correct"] and not f["correct"]:
            cats["standard_win"].append(t)
        elif f["correct"] and s["correct"]:
            cats["both_solved"].append(t)
        else:
            cats["both_failed"].append(t)
    for t in sorted(set(feat) & set(rnd)):
        if feat[t]["correct"] and not rnd[t]["correct"]:
            cats["vs_random"].append(t)

    chosen, used = [], set()
    for cat in ["feature_win", "standard_win", "both_solved", "both_failed", "vs_random"]:
        taken = 0
        for t in cats[cat]:
            if t in used or len(chosen) >= max_cases or taken >= per_category:
                continue
            used.add(t)
            taken += 1
            chosen.append((cat, t))
    empty = [c for c in CATEGORY_TITLES if not cats[c]]
    return chosen, cats, empty, {"feature": feat, "standard": std, "random": rnd, "quality": qual}


def build_case(case_id, category, task_id, per_cond, task_row, out_dir, run_id):
    """Build one case study: dict + PGN + standalone SVG board sequence."""
    feat = per_cond["feature"].get(task_id)
    std = per_cond["standard"].get(task_id)
    rnd = per_cond["random"].get(task_id)
    qual = per_cond["quality"].get(task_id)
    ref = feat or std
    fen = ref["fen"]
    board = chess.Board(fen)
    stm = board.turn
    orientation = stm  # always view from the side solving the task
    solutions = ref["solution_moves"]
    mate_in = ref["mate_in"]

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    assets = {}

    # 1. initial position
    check_sq = board.king(stm) if board.is_check() else None
    initial_svg = _svg(board, size=360, orientation=orientation, check=check_sq)
    assets["initial"] = _write(fig_dir, f"{case_id}_initial", initial_svg)

    # 2. side-by-side chosen moves
    def move_panel(row, color):
        if row is None:
            return None, None
        mv = chess.Move.from_uci(row["chosen_move_uci"])
        if mv not in board.legal_moves:
            return None, None
        return _svg(board, size=330, orientation=orientation,
                    arrows=[_arrow(mv, color)],
                    fill={mv.from_square: "#ffd54f88", mv.to_square: "#ef9a9a88"}), board.san(mv)

    std_svg, std_san = move_panel(std, "#4C72B0")
    feat_svg, feat_san = move_panel(feat, "#2E7D32")
    if std_svg:
        assets["standard_move"] = _write(fig_dir, f"{case_id}_standard", std_svg)
    if feat_svg:
        assets["feature_move"] = _write(fig_dir, f"{case_id}_feature", feat_svg)

    # 3. verified winning continuation (oracle) - distinct from the search PV
    best_solution = solutions[0] if solutions else None
    oracle_line = winning_continuation(board, best_solution, mate_in) if best_solution else []
    seq_boards, seq_labels = [], []
    work = chess.Board(fen)
    seq_boards.append(work.copy())
    seq_labels.append(f"Start - {'White' if stm else 'Black'} to move, mate in {mate_in}")
    num = work.fullmove_number
    for i, san in enumerate(oracle_line):
        try:
            mv = work.parse_san(san)
        except ValueError:
            break
        prefix = f"{work.fullmove_number}." if work.turn == chess.WHITE else f"{work.fullmove_number}..."
        work.push(mv)
        seq_boards.append(work.copy())
        seq_labels.append(f"{prefix} {san}")
    seq_assets = []
    for i, (bd, lab) in enumerate(zip(seq_boards, seq_labels)):
        chk = bd.king(bd.turn) if bd.is_check() else None
        last = [_arrow(bd.peek(), "#C44E52")] if bd.move_stack else []
        svg = _svg(bd, size=260, orientation=orientation, arrows=last, check=chk)
        seq_assets.append({"label": lab, "fen": bd.fen(),
                           **_write(fig_dir, f"{case_id}_seq{i}", svg)})
    assets["sequence"] = seq_assets

    # 4. refutation for a failed move
    refutations = {}
    for name, row in (("standard", std), ("feature", feat), ("random", rnd), ("quality", qual)):
        if row is None or row["correct"]:
            continue
        line = refutation_line(board, row["chosen_move_uci"], mate_in)
        refutations[name] = line

    # 5. PGN
    game = chess.pgn.Game()
    game.setup(chess.Board(fen))
    game.headers["Event"] = f"ChessConceptDiscovery case {case_id}"
    game.headers["Site"] = run_id
    game.headers["White"] = "White"
    game.headers["Black"] = "Black"
    game.headers["Result"] = "*"
    game.headers["FEN"] = fen
    game.headers["SetUp"] = "1"
    game.headers["Annotator"] = "exact forced-mate verifier"
    node = game
    replay = chess.Board(fen)
    for san in oracle_line:
        try:
            mv = replay.parse_san(san)
        except ValueError:
            break
        replay.push(mv)
        node = node.add_variation(mv)
    node.comment = f"verified forced mate in {mate_in}; complete solution set: {', '.join(solutions)}"

    # 6. explanation from recorded facts only (deterministic template)
    explanation = _explain(feat, std, rnd, qual, solutions, mate_in, board, refutations)

    case = {
        "case_id": case_id,
        "category": category,
        "category_title": CATEGORY_TITLES[category],
        "task_id": task_id,
        "family": ref["family"],
        "source": ref["source"],
        "fen": fen,
        "side_to_move": "White" if stm else "Black",
        "orientation": "White" if orientation else "Black",
        "mate_in": mate_in,
        "horizon_plies": ref["horizon_plies"],
        "num_legal_moves": ref["num_legal_moves"],
        "solution_moves": solutions,
        "solution_sans": [board.san(chess.Move.from_uci(u)) for u in solutions
                          if chess.Move.from_uci(u) in board.legal_moves],
        "solver_status": ref["solver_status"],
        "methods": {},
        "oracle_line_san": oracle_line,
        "refutations": refutations,
        "assets": assets,
        "explanation": explanation,
        "result_ids": {},
        "pgn": str(game),
    }
    for name, row in (("standard", std), ("quality", qual), ("random", rnd), ("feature", feat)):
        if row is None:
            continue
        mv = chess.Move.from_uci(row["chosen_move_uci"])
        case["methods"][name] = {
            "move_uci": row["chosen_move_uci"],
            "move_san": board.san(mv) if mv in board.legal_moves else "(illegal)",
            "correct": bool(row["correct"]),
            "policy_rank": row["policy_rank_of_chosen"],
            "policy_prob": row["policy_prob_of_chosen"],
            "simulations": row["stats"]["simulations"],
            "nn_evals": row["stats"]["nn_evals"],
            "elapsed_s": row["stats"]["elapsed_s"],
            "candidates": row.get("candidates", []),
            "search_pv_san": row.get("search_pv_san", []),
            "feature_mode": row.get("feature_mode", "n/a"),
        }
        case["result_ids"][name] = row["result_id"]
    case["best_solution_policy_rank"] = ref["best_solution_policy_rank"]
    case["best_solution_policy_prob"] = ref["best_solution_policy_prob"]
    return case


def _write(fig_dir, name, svg_text):
    path = os.path.join(fig_dir, name + ".svg")
    with open(path, "w") as fh:
        fh.write(svg_text)
    return {"svg_path": path, "svg_inline": svg_text}


def _explain(feat, std, rnd, qual, solutions, mate_in, board, refutations):
    """Deterministic explanation template built from recorded facts only."""
    parts = []
    n_sol = len(solutions)
    parts.append(
        f"The exact verifier proved {n_sol} root move"
        f"{'s' if n_sol != 1 else ''} force mate in {mate_in} "
        f"({', '.join(solutions)}), checking every legal defensive reply."
    )
    if feat is not None:
        cands = feat.get("candidates", [])
        if cands:
            names = ", ".join(f"{c['san']} (screen visits {c['screen_visits']}, "
                              f"{c['alloc_sims']} extra sims)" for c in cands)
            parts.append(f"Feature-diverse allocation explored {len(cands)} candidates: {names}.")
        parts.append(
            f"It selected {feat['chosen_move_uci']}, which the raw network policy ranked "
            f"#{feat['policy_rank_of_chosen']} with probability "
            f"{feat['policy_prob_of_chosen']:.3f}. "
            f"The verifier says this move {'DOES' if feat['correct'] else 'does NOT'} "
            f"force mate within {mate_in}."
        )
    if std is not None:
        parts.append(
            f"Standard PUCT selected {std['chosen_move_uci']} (raw policy rank "
            f"#{std['policy_rank_of_chosen']}, p={std['policy_prob_of_chosen']:.3f}); "
            f"verifier: {'solved' if std['correct'] else 'not a forced mate within horizon'}."
        )
    for name, line in refutations.items():
        if line:
            parts.append(
                f"Refutation of the {name} choice: after {line[0]} Black replies {line[1]}, "
                f"which survives the {mate_in}-move horizon, so the mate claim fails. "
                "This shows the task objective was missed; it is not a proof that the move loses the game."
            )
        else:
            parts.append(
                f"The {name} choice is not in the verified solution set. The bounded verifier "
                "did not establish an explicit escape line within its node budget, so we report "
                "only that it fails the mate-within-horizon objective."
            )
    return " ".join(parts)


def generate(rows, summary, out_dir, run_id, per_category=2, max_cases=10):
    chosen, cats, empty, per_cond = select_cases(rows, summary, per_category, max_cases)
    case_dir = os.path.join(out_dir, "case_studies")
    os.makedirs(case_dir, exist_ok=True)
    pgn_dir = os.path.join(case_dir, "pgn")
    os.makedirs(pgn_dir, exist_ok=True)

    cases = []
    by_task = {r["task_id"]: r for r in rows}
    for i, (cat, tid) in enumerate(chosen):
        cid = f"case{i+1:02d}"
        try:
            c = build_case(cid, cat, tid, per_cond, by_task.get(tid), case_dir, run_id)
        except Exception as exc:  # keep the run alive, record the failure
            cases.append({"case_id": cid, "task_id": tid, "category": cat, "error": repr(exc)})
            continue
        with open(os.path.join(pgn_dir, f"{cid}.pgn"), "w") as fh:
            fh.write(c["pgn"] + "\n")
        cases.append(c)

    index = {
        "run_id": run_id,
        "selection_rule": (
            "Fixed order feature_win -> standard_win -> both_solved -> both_failed -> vs_random; "
            f"up to {per_category} per category, {max_cases} total, task-id order within a category."
        ),
        "category_counts": {k: len(v) for k, v in cats.items()},
        "empty_categories": empty,
        "cases": [{k: v for k, v in c.items() if k not in ("assets", "pgn")} for c in cases],
    }
    with open(os.path.join(case_dir, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)

    # combined PGN
    with open(os.path.join(out_dir, "examples.pgn"), "w") as fh:
        for c in cases:
            if "pgn" in c:
                fh.write(c["pgn"] + "\n\n")
    return cases, index
