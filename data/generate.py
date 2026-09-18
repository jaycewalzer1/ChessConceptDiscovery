"""Automatic generation of independently verified forced-mate tasks.

NEW module.

Tasks are produced entirely locally -- no puzzle database download, no external
engine.  Two generators are used and recorded per task:

``playout``   A random legal game is played from the standard start position
              (a "source game").  Every position along that game is checked by
              the exact forced-mate verifier.  Positions harvested from the
              same source game share a ``family`` id.
``endgame``   A random small-material endgame position is *constructed* and
              then reached by a short random legal prefix so the recorded
              position has an explicit legal history.  All positions from one
              construction seed share a ``family`` id.

For every task we store: generator source, the legal history (UCI move list)
that produced it, the FEN, the task horizon in plies, the **complete** verified
solution-move set, solver status and node count, and a SHA-256 hash of the
position + solution set.

``board.is_valid()`` alone is NOT treated as proof of historical reachability.
Every stored task carries the actual move sequence that produced it, so all
tasks in this benchmark are reachable by legal play from the standard initial
position (playout) or from a constructed diagram plus a legal prefix (endgame,
flagged as ``reachable_from_start = False``).

SPLITTING
---------
Splits are by **family**, never by row, so no two positions from the same source
game or construction seed can land in different splits.  Exact FEN duplicates
are removed globally before splitting.  Near-duplicate detection uses a
piece-placement-only key (FEN field 1 + side to move); board transformations are
deliberately NOT applied because mirroring breaks castling and pawn-direction
semantics.  This limitation is recorded in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import chess

from envs.mate_solver import classify_mate_task


@dataclass
class Task:
    task_id: str
    family: str
    source: str  # "playout" | "endgame"
    fen: str
    history_uci: list
    reachable_from_start: bool
    mate_in: int
    horizon_plies: int
    solution_moves: list
    num_legal_moves: int
    solver_status: str
    solver_nodes: int
    position_hash: str

    def to_dict(self):
        return asdict(self)


def _hash(fen: str, solutions) -> str:
    h = hashlib.sha256()
    h.update(fen.encode())
    h.update("|".join(sorted(solutions)).encode())
    return h.hexdigest()[:16]


def _placement_key(fen: str) -> str:
    parts = fen.split()
    return parts[0] + " " + parts[1]


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def generate_from_playout(rng, family: str, max_plies: int = 120, want_mate_in=(1, 2),
                          node_budget: int = 200_000, max_per_game: int = 3,
                          max_legal: int = 45, greedy_capture_p: float = 0.35):
    """Play one random legal game and harvest verified forced-mate positions.

    Moves are sampled uniformly at random, with probability ``greedy_capture_p``
    biased toward captures/checks so that material drops and short mates become
    reachable inside the ply cap.
    """
    board = chess.Board()
    history = []
    found = []
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=False):
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        if len(legal) <= max_legal and len(board.piece_map()) <= 14:
            res = classify_mate_task(board, max_n=max(want_mate_in), node_budget=node_budget)
            if res.status == "ok" and res.mate_in in want_mate_in and res.solution_moves:
                fen = board.fen()
                found.append(
                    Task(
                        task_id=f"{family}-p{len(history)}",
                        family=family,
                        source="playout",
                        fen=fen,
                        history_uci=list(history),
                        reachable_from_start=True,
                        mate_in=res.mate_in,
                        horizon_plies=2 * res.mate_in - 1,
                        solution_moves=res.solution_moves,
                        num_legal_moves=len(legal),
                        solver_status=res.status,
                        solver_nodes=res.nodes,
                        position_hash=_hash(fen, res.solution_moves),
                    )
                )
                if len(found) >= max_per_game:
                    break
        interesting = [m for m in legal if board.is_capture(m) or board.gives_check(m)]
        if interesting and rng.random() < greedy_capture_p:
            mv = interesting[int(rng.integers(len(interesting)))]
        else:
            mv = legal[int(rng.integers(len(legal)))]
        history.append(mv.uci())
        board.push(mv)
    return found


ENDGAME_TEMPLATES = [
    # (white piece types, black piece types) -- kings are implicit
    ([chess.QUEEN], []),
    ([chess.ROOK, chess.ROOK], []),
    ([chess.QUEEN, chess.ROOK], []),
    ([chess.ROOK], [chess.PAWN]),
    ([chess.QUEEN], [chess.ROOK]),
    ([chess.ROOK, chess.BISHOP], [chess.PAWN]),
    ([chess.QUEEN, chess.KNIGHT], [chess.ROOK]),
    ([chess.ROOK, chess.KNIGHT], []),
    ([chess.QUEEN, chess.BISHOP], [chess.KNIGHT]),
    ([chess.ROOK, chess.PAWN], [chess.KNIGHT]),
]


def _random_endgame_board(rng):
    """Construct a random small-material diagram from a documented template."""
    tmpl = ENDGAME_TEMPLATES[int(rng.integers(len(ENDGAME_TEMPLATES)))]
    white_pieces, black_pieces = tmpl
    for _ in range(200):
        board = chess.Board(None)
        squares = list(rng.permutation(64))
        used = set()

        def place(piece, color):
            while squares:
                sq = int(squares.pop())
                if sq in used:
                    continue
                if piece == chess.PAWN and chess.square_rank(sq) in (0, 7):
                    continue
                used.add(sq)
                board.set_piece_at(sq, chess.Piece(piece, color))
                return True
            return False

        ok = place(chess.KING, chess.WHITE) and place(chess.KING, chess.BLACK)
        for p in white_pieces:
            ok = ok and place(p, chess.WHITE)
        for p in black_pieces:
            ok = ok and place(p, chess.BLACK)
        if not ok:
            continue
        board.turn = chess.WHITE
        board.castling_rights = 0
        board.halfmove_clock = 0
        board.fullmove_number = 1
        if board.is_valid():
            return board
    return None


def generate_from_endgame(rng, family: str, want_mate_in=(1, 2), node_budget: int = 200_000,
                          prefix_plies: int = 2, max_legal: int = 45, tries: int = 40):
    """Construct endgame diagrams, add a short legal prefix, and verify."""
    found = []
    for _ in range(tries):
        base = _random_endgame_board(rng)
        if base is None:
            continue
        board = base.copy(stack=True)
        history = []
        for _ in range(prefix_plies):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over(claim_draw=False):
                break
            mv = legal[int(rng.integers(len(legal)))]
            history.append(mv.uci())
            board.push(mv)
        if board.is_game_over(claim_draw=False):
            continue
        if len(list(board.legal_moves)) > max_legal:
            continue
        res = classify_mate_task(board, max_n=max(want_mate_in), node_budget=node_budget)
        if res.status == "ok" and res.mate_in in want_mate_in and res.solution_moves:
            fen = board.fen()
            found.append(
                Task(
                    task_id=f"{family}-e{len(found)}",
                    family=family,
                    source="endgame",
                    fen=fen,
                    history_uci=history,
                    reachable_from_start=False,
                    mate_in=res.mate_in,
                    horizon_plies=2 * res.mate_in - 1,
                    solution_moves=res.solution_moves,
                    num_legal_moves=len(list(board.legal_moves)),
                    solver_status=res.status,
                    solver_nodes=res.nodes,
                    position_hash=_hash(fen, res.solution_moves),
                )
            )
            return found, base.fen()
    return found, None


# --------------------------------------------------------------------------- #
# dataset assembly
# --------------------------------------------------------------------------- #
def generate_dataset(seed: int, target: int, time_budget_s: float, want_mate_in=(1, 2),
                     node_budget: int = 200_000, log=print, endgame_frac: float = 0.55):
    """Generate up to ``target`` verified tasks within ``time_budget_s``."""
    rng = np.random.default_rng(seed)
    tasks, seen_fen, seen_placement = [], set(), set()
    t0 = time.perf_counter()
    n_games = n_endgames = 0
    while len(tasks) < target and time.perf_counter() - t0 < time_budget_s:
        use_endgame = rng.random() < endgame_frac
        if use_endgame:
            fam = f"eg{n_endgames:05d}"
            found, _ = generate_from_endgame(rng, fam, want_mate_in, node_budget)
            n_endgames += 1
        else:
            fam = f"pg{n_games:05d}"
            found = generate_from_playout(rng, fam, want_mate_in=want_mate_in, node_budget=node_budget)
            n_games += 1
        for t in found:
            if t.fen in seen_fen:
                continue
            pk = _placement_key(t.fen)
            if pk in seen_placement:
                continue
            seen_fen.add(t.fen)
            seen_placement.add(pk)
            tasks.append(t)
        if len(tasks) % 25 == 0 and found:
            log(f"[data] {len(tasks)} tasks, {time.perf_counter()-t0:.0f}s elapsed")
    elapsed = time.perf_counter() - t0
    log(f"[data] generated {len(tasks)} tasks in {elapsed:.1f}s "
        f"({n_games} playout games, {n_endgames} endgame seeds)")
    return tasks, {
        "seed": seed,
        "target": target,
        "generated": len(tasks),
        "elapsed_s": elapsed,
        "playout_games": n_games,
        "endgame_seeds": n_endgames,
        "want_mate_in": list(want_mate_in),
        "solver_node_budget": node_budget,
    }


def split_by_family(tasks, seed: int, frac_train: float = 0.45, frac_dev: float = 0.2):
    """Split tasks into train/dev/test **by family**, never by row."""
    rng = np.random.default_rng(seed)
    families = sorted({t.family for t in tasks})
    rng.shuffle(families)
    n = len(families)
    n_tr = int(round(frac_train * n))
    n_dev = int(round(frac_dev * n))
    fam_split = {}
    for i, f in enumerate(families):
        fam_split[f] = "train" if i < n_tr else ("dev" if i < n_tr + n_dev else "test")
    out = {"train": [], "dev": [], "test": []}
    for t in tasks:
        out[fam_split[t.family]].append(t)
    return out, fam_split


def write_manifest(path, splits, gen_info, fam_split, extra=None):
    def summarise(ts):
        return {
            "n": len(ts),
            "families": len({t.family for t in ts}),
            "mate_in_counts": {
                str(k): sum(1 for t in ts if t.mate_in == k) for k in sorted({t.mate_in for t in ts})
            },
            "sources": {
                s: sum(1 for t in ts if t.source == s) for s in sorted({t.source for t in ts})
            },
            "median_legal_moves": float(np.median([t.num_legal_moves for t in ts])) if ts else None,
            "split_hash": hashlib.sha256("|".join(sorted(t.position_hash for t in ts)).encode()).hexdigest(),
        }

    manifest = {
        "generation": gen_info,
        "splits": {k: summarise(v) for k, v in splits.items()},
        "family_assignment": fam_split,
        "leakage_controls": {
            "split_unit": "source family (one playout game or one endgame construction seed)",
            "exact_fen_dedup": True,
            "near_dup_key": "piece placement + side to move",
            "transformations_applied": "none (mirroring breaks castling/pawn semantics)",
            "known_limits": "near-duplicate detection is placement-based only; "
                            "positions differing solely in en-passant/halfmove counters are kept.",
        },
        "solution_storage": "solution_moves are stored in the label file only; "
                            "agent-facing test positions carry FEN + horizon only.",
    }
    if extra:
        manifest.update(extra)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def write_split_files(out_dir, splits):
    """Write agent-facing positions and oracle labels to SEPARATE files."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, ts in splits.items():
        pos_path = os.path.join(out_dir, f"{name}_positions.jsonl")
        lab_path = os.path.join(out_dir, f"{name}_labels.jsonl")
        with open(pos_path, "w") as fh:
            for t in ts:
                fh.write(json.dumps({
                    "task_id": t.task_id, "family": t.family, "source": t.source,
                    "fen": t.fen, "history_uci": t.history_uci,
                    "horizon_plies": t.horizon_plies, "mate_in": t.mate_in,
                    "num_legal_moves": t.num_legal_moves,
                    "reachable_from_start": t.reachable_from_start,
                }) + "\n")
        with open(lab_path, "w") as fh:
            for t in ts:
                fh.write(json.dumps({
                    "task_id": t.task_id, "solution_moves": t.solution_moves,
                    "solver_status": t.solver_status, "solver_nodes": t.solver_nodes,
                    "position_hash": t.position_hash,
                }) + "\n")
        paths[name] = {"positions": pos_path, "labels": lab_path}
    return paths


def load_split(paths, name, with_labels: bool = True):
    positions = [json.loads(l) for l in open(paths[name]["positions"])]
    if not with_labels:
        return positions
    labels = {json.loads(l)["task_id"]: json.loads(l) for l in open(paths[name]["labels"])}
    for p in positions:
        p.update(labels[p["task_id"]])
    return positions
