"""Exact bounded forced-mate verifier (evaluation oracle).

NEW module (not present in the original repository).

This is a small, self-contained alpha-beta-free exhaustive searcher over legal
moves implemented with ``python-chess``.  It is used **only** to build and grade
the benchmark.  No external chess engine, UCI process or LLM is involved.

Semantics
---------
``is_forced_mate_in(board, n)`` is True iff the side to move has a move such
that, **for every** legal reply of the defender, the attacker can force
checkmate within the remaining budget.  Concretely, with ``n`` attacker moves
allowed (horizon ``2n - 1`` plies):

    attacker (existential)  ->  defender (universal)  ->  ...  -> checkmate

A defender that is stalemated, or that reaches any rules draw, **refutes** the
mate claim.  Checks, promotions (including underpromotions), castling and
en-passant are handled exactly because legal move generation is delegated to
``python-chess``.

``solve_root_moves`` returns the **complete set** of root moves that force mate
in exactly ``n`` (i.e. mate within ``n`` but not within ``n - 1``), together
with a status flag.  If the node budget is exhausted the status is
``"unknown"`` and the caller must exclude the position from verified scoring --
a partial enumeration is never reported as a proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


class NodeBudgetExceeded(Exception):
    pass


@dataclass
class SolverStats:
    nodes: int = 0
    budget: int = 0
    exceeded: bool = False


@dataclass
class MateResult:
    """Result of enumerating root moves for a forced-mate task."""

    status: str  # "ok" | "unknown"
    mate_in: int | None  # attacker moves required, None if no mate found
    solution_moves: list = field(default_factory=list)  # UCI strings, sorted
    nodes: int = 0


def _mates_in_at_most(board: chess.Board, n: int, stats: SolverStats) -> bool:
    """True iff the side to move forces mate in at most ``n`` of its own moves."""
    if n <= 0:
        return False
    for move in board.legal_moves:
        stats.nodes += 1
        if stats.nodes > stats.budget:
            raise NodeBudgetExceeded
        board.push(move)
        try:
            if board.is_checkmate():
                return True
            if n == 1:
                continue
            # Defender to move: every legal reply must still lose.
            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                # stalemate / insufficient material / 75-move / fivefold -> not mate
                continue
            ok = True
            for reply in board.legal_moves:
                stats.nodes += 1
                if stats.nodes > stats.budget:
                    raise NodeBudgetExceeded
                board.push(reply)
                try:
                    # After the defender's reply the attacker is to move again.
                    if board.outcome(claim_draw=False) is not None:
                        ok = False  # defender reached a draw/terminal -> escape
                    elif not _mates_in_at_most(board, n - 1, stats):
                        ok = False
                finally:
                    board.pop()
                if not ok:
                    break
            if ok:
                return True
        finally:
            board.pop()
    return False


def is_forced_mate_in(board: chess.Board, n: int, node_budget: int = 400_000) -> tuple:
    """Return ``(status, is_mate)`` for "side to move mates in at most n"."""
    stats = SolverStats(budget=node_budget)
    work = board.copy(stack=False)
    try:
        return "ok", _mates_in_at_most(work, n, stats)
    except NodeBudgetExceeded:
        return "unknown", False


def _root_move_mates_in_at_most(board: chess.Board, move: chess.Move, n: int, stats: SolverStats) -> bool:
    """True iff playing ``move`` forces mate within ``n`` attacker moves total."""
    board.push(move)
    try:
        if board.is_checkmate():
            return True
        if n <= 1:
            return False
        if board.outcome(claim_draw=False) is not None:
            return False
        for reply in board.legal_moves:
            stats.nodes += 1
            if stats.nodes > stats.budget:
                raise NodeBudgetExceeded
            board.push(reply)
            try:
                if board.outcome(claim_draw=False) is not None:
                    return False
                if not _mates_in_at_most(board, n - 1, stats):
                    return False
            finally:
                board.pop()
        return True
    finally:
        board.pop()


def solve_root_moves(board: chess.Board, n: int, node_budget: int = 400_000) -> MateResult:
    """Enumerate **all** root moves that force mate in at most ``n`` moves.

    Returns ``MateResult`` with ``status == "unknown"`` if the node budget was
    exhausted before enumeration completed.
    """
    stats = SolverStats(budget=node_budget)
    work = board.copy(stack=False)
    solutions = []
    try:
        for move in work.legal_moves:
            stats.nodes += 1
            if stats.nodes > stats.budget:
                raise NodeBudgetExceeded
            if _root_move_mates_in_at_most(work, move, n, stats):
                solutions.append(move.uci())
    except NodeBudgetExceeded:
        return MateResult(status="unknown", mate_in=None, solution_moves=[], nodes=stats.nodes)
    return MateResult(
        status="ok",
        mate_in=n if solutions else None,
        solution_moves=sorted(solutions),
        nodes=stats.nodes,
    )


def classify_mate_task(board: chess.Board, max_n: int = 3, node_budget: int = 400_000) -> MateResult:
    """Find the smallest ``n <= max_n`` for which the side to move forces mate.

    Returns the complete solution set for that ``n``.  ``mate_in is None`` when
    no forced mate exists inside ``max_n`` (status ``"ok"``) or the search was
    cut short (status ``"unknown"``).
    """
    total_nodes = 0
    for n in range(1, max_n + 1):
        res = solve_root_moves(board, n, node_budget=node_budget)
        total_nodes += res.nodes
        if res.status == "unknown":
            return MateResult(status="unknown", mate_in=None, solution_moves=[], nodes=total_nodes)
        if res.solution_moves:
            return MateResult(
                status="ok", mate_in=n, solution_moves=res.solution_moves, nodes=total_nodes
            )
    return MateResult(status="ok", mate_in=None, solution_moves=[], nodes=total_nodes)


def winning_continuation(board: chess.Board, move_uci: str, n: int, node_budget: int = 400_000) -> list:
    """Produce one verified winning line after ``move_uci``.

    Each defender reply chosen is the one that survives longest under the
    oracle (a "most stubborn defence"), so the line is a legitimate
    illustration of the forced mate.  It illustrates the oracle's proof; it is
    **not** itself the proof, which quantifies over all replies.
    """
    stats = SolverStats(budget=node_budget)
    work = board.copy(stack=False)
    line = []
    move = chess.Move.from_uci(move_uci)
    if move not in work.legal_moves:
        return []
    line.append(work.san(move))
    work.push(move)
    remaining = n - 1
    try:
        while not work.is_game_over(claim_draw=False) and remaining >= 0:
            if work.is_checkmate():
                break
            # defender to move: pick the reply requiring the deepest mate
            best_reply, best_depth = None, -1
            for reply in work.legal_moves:
                work.push(reply)
                try:
                    if work.is_game_over(claim_draw=False):
                        d = 0
                    else:
                        d = remaining
                        for k in range(1, remaining + 1):
                            if _mates_in_at_most(work, k, stats):
                                d = k
                                break
                finally:
                    work.pop()
                if d > best_depth:
                    best_reply, best_depth = reply, d
            if best_reply is None:
                break
            line.append(work.san(best_reply))
            work.push(best_reply)
            if work.is_game_over(claim_draw=False):
                break
            # attacker to move: pick a move that still mates in best_depth
            chosen = None
            for cand in work.legal_moves:
                if _root_move_mates_in_at_most(work, cand, max(best_depth, 1), stats):
                    chosen = cand
                    break
            if chosen is None:
                break
            line.append(work.san(chosen))
            work.push(chosen)
            remaining = max(best_depth - 1, 0)
            if work.is_checkmate():
                break
    except NodeBudgetExceeded:
        pass
    return line


def refutation_line(board: chess.Board, move_uci: str, n: int, node_budget: int = 200_000) -> list:
    """Find a defensive escape showing ``move_uci`` does NOT mate within ``n``.

    Returns ``[attacker_san, defender_san]`` where the defender's reply survives
    the horizon, or ``[]`` if no escape can be established within the budget
    (e.g. the move is illegal or the budget is exhausted).
    """
    stats = SolverStats(budget=node_budget)
    work = board.copy(stack=False)
    move = chess.Move.from_uci(move_uci)
    if move not in work.legal_moves:
        return []
    san = work.san(move)
    work.push(move)
    try:
        if work.is_checkmate():
            return []
        outcome = work.outcome(claim_draw=False)
        if outcome is not None:
            return [san, f"(game ends: {outcome.result()})"]
        for reply in work.legal_moves:
            reply_san = work.san(reply)
            work.push(reply)
            try:
                if work.outcome(claim_draw=False) is not None:
                    return [san, reply_san]
                if not _mates_in_at_most(work, n - 1, stats):
                    return [san, reply_san]
            finally:
                work.pop()
    except NodeBudgetExceeded:
        return []
    return []
