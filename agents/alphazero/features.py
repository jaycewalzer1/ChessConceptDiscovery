"""Deterministic, inexpensive candidate descriptors.

NEW module.

A *candidate* is a root move.  Its descriptor summarises what changes along a
short continuation that the screening search already explored (or, as a
documented fallback, along the immediate successor state only).  All quantities
are computed **from the root player's perspective**.

IMPORTANT SCOPE NOTE
--------------------
These are cheap board statistics.  Distance in this feature space is the
*hypothesis under test*; it is **not** an established measure of creativity or
of strategic intent.  A feature named "sacrifice_proxy" records an observed
material deficit along a short line -- it does **not** assert that a sound
sacrifice was found.  Board hashes, random numbers and solution labels are
deliberately excluded from the descriptor.

Two extraction modes are supported and recorded per run:

``continuation``  descriptors computed over a continuation of up to
                  ``max_len`` plies taken from the screening tree.
``successor``     fallback used when no reliable comparable continuation
                  exists; only the immediate successor state is used.  When
                  this mode is active the condition must be described as
                  *state-feature diversity*, not *strategic-plan diversity*.
"""

from __future__ import annotations

import numpy as np
import chess

PIECE_VALUES = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}

FEATURE_NAMES = [
    "d_material_pawn",
    "d_material_knight",
    "d_material_bishop",
    "d_material_rook",
    "d_material_queen",
    "n_captures",
    "n_checks",
    "n_promotions",
    "gives_check_immediate",
    "is_capture_immediate",
    "is_promotion_immediate",
    "moved_piece_value",
    "d_pawn_advancement",
    "d_center_occupancy",
    "d_kingside_occupancy",
    "d_queenside_occupancy",
    "d_own_mobility",
    "d_opp_mobility",
    "d_opp_king_ring_attacks",
    "d_own_king_ring_attacks",
    "d_open_files",
    "d_minor_activity",
    "sacrifice_proxy",
    "continuation_length",
]
NUM_FEATURES = len(FEATURE_NAMES)

CENTER = [chess.D4, chess.E4, chess.D5, chess.E5, chess.C4, chess.C5, chess.F4, chess.F5]
KINGSIDE_FILES = [5, 6, 7]
QUEENSIDE_FILES = [0, 1, 2]


def _material(board: chess.Board, color: bool) -> dict:
    return {pt: len(board.pieces(pt, color)) for pt in PIECE_VALUES}


def _material_total(board: chess.Board, color: bool) -> float:
    return sum(PIECE_VALUES[pt] * len(board.pieces(pt, color)) for pt in PIECE_VALUES)


def _pawn_advancement(board: chess.Board, color: bool) -> float:
    total = 0.0
    for sq in board.pieces(chess.PAWN, color):
        r = chess.square_rank(sq)
        total += r if color == chess.WHITE else (7 - r)
    return total


def _region_occupancy(board: chess.Board, color: bool, squares) -> float:
    return float(sum(1 for sq in squares if (p := board.piece_at(sq)) is not None and p.color == color))


def _file_squares(files):
    return [chess.square(f, r) for f in files for r in range(8)]


def _mobility(board: chess.Board, color: bool) -> float:
    """Pseudo-legal mobility for ``color`` regardless of whose turn it is."""
    work = board.copy(stack=False)
    work.turn = color
    work.ep_square = None
    return float(sum(1 for _ in work.generate_pseudo_legal_moves()))


def _king_ring_attacks(board: chess.Board, attacker: bool) -> float:
    """Number of squares around the defending king attacked by ``attacker``."""
    defender = not attacker
    ksq = board.king(defender)
    if ksq is None:
        return 0.0
    ring = [s for s in chess.SQUARES if chess.square_distance(s, ksq) == 1]
    return float(sum(1 for s in ring if board.is_attacked_by(attacker, s)))


def _open_files(board: chess.Board) -> float:
    n = 0
    for f in range(8):
        if not any(board.piece_type_at(chess.square(f, r)) == chess.PAWN for r in range(8)):
            n += 1
    return float(n)


def _minor_activity(board: chess.Board, color: bool) -> float:
    home = 0 if color == chess.WHITE else 7
    n = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if chess.square_rank(sq) != home:
                n += 1
    return float(n)


def continuation_descriptor(root_board: chess.Board, moves, mode: str = "continuation") -> np.ndarray:
    """Compute the descriptor for a candidate given its continuation.

    ``moves`` is a list of ``chess.Move`` starting with the candidate root move.
    All deltas are ``end_state - start_state`` from the ROOT PLAYER's view.
    """
    me = root_board.turn
    opp = not me
    start = root_board.copy(stack=False)

    if not moves:
        raise ValueError("continuation_descriptor requires at least the root move")

    work = root_board.copy(stack=True)
    n_captures = n_checks = n_promotions = 0
    gives_check_immediate = is_capture_immediate = is_promotion_immediate = 0.0
    moved_piece_value = 0.0
    min_material_diff = None
    applied = 0

    base_diff = _material_total(start, me) - _material_total(start, opp)

    for i, mv in enumerate(moves):
        if mv not in work.legal_moves:
            break
        if i == 0:
            piece = work.piece_at(mv.from_square)
            moved_piece_value = PIECE_VALUES.get(piece.piece_type, 0.0) if piece else 0.0
            is_capture_immediate = 1.0 if work.is_capture(mv) else 0.0
            is_promotion_immediate = 1.0 if mv.promotion else 0.0
            gives_check_immediate = 1.0 if work.gives_check(mv) else 0.0
        if work.is_capture(mv):
            n_captures += 1
        if mv.promotion:
            n_promotions += 1
        if work.gives_check(mv):
            n_checks += 1
        work.push(mv)
        applied += 1
        diff = _material_total(work, me) - _material_total(work, opp) - base_diff
        min_material_diff = diff if min_material_diff is None else min(min_material_diff, diff)
        if work.is_game_over(claim_draw=False):
            break
        if mode == "successor":
            break

    end = work
    mat_s, mat_e = _material(start, me), _material(end, me)
    omat_s, omat_e = _material(start, opp), _material(end, opp)

    feats = [
        (mat_e[chess.PAWN] - mat_s[chess.PAWN]) - (omat_e[chess.PAWN] - omat_s[chess.PAWN]),
        (mat_e[chess.KNIGHT] - mat_s[chess.KNIGHT]) - (omat_e[chess.KNIGHT] - omat_s[chess.KNIGHT]),
        (mat_e[chess.BISHOP] - mat_s[chess.BISHOP]) - (omat_e[chess.BISHOP] - omat_s[chess.BISHOP]),
        (mat_e[chess.ROOK] - mat_s[chess.ROOK]) - (omat_e[chess.ROOK] - omat_s[chess.ROOK]),
        (mat_e[chess.QUEEN] - mat_s[chess.QUEEN]) - (omat_e[chess.QUEEN] - omat_s[chess.QUEEN]),
        float(n_captures),
        float(n_checks),
        float(n_promotions),
        gives_check_immediate,
        is_capture_immediate,
        is_promotion_immediate,
        moved_piece_value,
        _pawn_advancement(end, me) - _pawn_advancement(start, me),
        _region_occupancy(end, me, CENTER) - _region_occupancy(start, me, CENTER),
        _region_occupancy(end, me, _file_squares(KINGSIDE_FILES)) - _region_occupancy(start, me, _file_squares(KINGSIDE_FILES)),
        _region_occupancy(end, me, _file_squares(QUEENSIDE_FILES)) - _region_occupancy(start, me, _file_squares(QUEENSIDE_FILES)),
        _mobility(end, me) - _mobility(start, me),
        _mobility(end, opp) - _mobility(start, opp),
        _king_ring_attacks(end, me) - _king_ring_attacks(start, me),
        _king_ring_attacks(end, opp) - _king_ring_attacks(start, opp),
        _open_files(end) - _open_files(start),
        _minor_activity(end, me) - _minor_activity(start, me),
        float(min(min_material_diff, 0.0)) if min_material_diff is not None else 0.0,
        float(applied),
    ]
    return np.asarray(feats, dtype=np.float32)


class FeatureScaler:
    """Standardisation fitted on TRAINING/DEVELOPMENT data only."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
        self.std = std
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.mean is None:
            return X
        return (X - self.mean) / self.std

    def to_dict(self):
        return {
            "feature_names": FEATURE_NAMES,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(mean=d.get("mean"), std=d.get("std"))
