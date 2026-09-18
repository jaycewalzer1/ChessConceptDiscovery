"""Board / action encoding utilities for the AlphaZero-style chess agent.

Reused from the original repository: the module name and the public function
names ``encode_board``, ``decode_action`` and ``get_valid_moves``.  The original
bodies were stubs (``encode_board`` returned an all-zero tensor, the other two
returned ``None``); they are implemented here on top of ``python-chess``.

ACTION ENCODING (version "fromto-promo-v1")
-------------------------------------------
A flat ``from x to x promotion`` encoding::

    action_id = from_square * 64 * 5 + to_square * 5 + promotion_index

with ``promotion_index`` in::

    0 -> no promotion
    1 -> knight   2 -> bishop   3 -> rook   4 -> queen

so underpromotions each get their own distinct action id.  Total action space
is ``64 * 64 * 5 = 20480``.  The encoding is larger than the compressed
AlphaZero 8x8x73 plane space but is unambiguous and cheap to verify by
round-trip tests.

CANONICAL ORIENTATION
---------------------
Observations and action ids are always expressed from the point of view of the
*side to move*.  When Black is to move the board is mirrored (vertical flip)
so that the side to move always plays "up the board".  Squares are mapped with
``chess.square_mirror``; promotion piece types are unchanged.  The value head
therefore always predicts the value **for the player to move**.

OBSERVATION PLANES (version "obs-20plane-v1"), shape (8, 8, 20)
---------------------------------------------------------------
  0-5    own pieces      (P, N, B, R, Q, K) in canonical orientation
  6-11   opponent pieces (P, N, B, R, Q, K)
  12     constant plane, always 1.0 (side-to-move is implicit via canonical
         orientation; kept so the network can learn a board-wide bias)
  13     own kingside castling right
  14     own queenside castling right
  15     opponent kingside castling right
  16     opponent queenside castling right
  17     en-passant target square (single 1.0 on the canonical square)
  18     halfmove clock / 100, broadcast
  19     1.0 if the current position has occurred before in the game history

NOTE ON MARKOV-NESS: plane 19 is a single repetition bit and planes 0-18 carry
no move history.  The observation is therefore **not** fully Markov with
respect to the threefold-repetition rule.  The *search* state keeps the full
``chess.Board`` (including its move stack), so game rules are applied exactly;
only the network input is history-compressed.  This is stated explicitly in the
report rather than glossed over.
"""

from __future__ import annotations

import numpy as np
import chess

ENCODING_VERSION = "fromto-promo-v1"
OBS_VERSION = "obs-20plane-v1"

NUM_SQUARES = 64
NUM_PROMO = 5
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES * NUM_PROMO  # 20480
OBS_SHAPE = (8, 8, 20)

# promotion_index -> chess piece type (None for "no promotion")
PROMO_INDEX_TO_PIECE = {0: None, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}
PIECE_TO_PROMO_INDEX = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


# --------------------------------------------------------------------------- #
# canonical orientation helpers
# --------------------------------------------------------------------------- #
def canonical_square(square: int, side_to_move: bool) -> int:
    """Map a real square to the canonical (side-to-move-up) frame."""
    return square if side_to_move == chess.WHITE else chess.square_mirror(square)


def real_square(square: int, side_to_move: bool) -> int:
    """Inverse of :func:`canonical_square` (the mapping is an involution)."""
    return square if side_to_move == chess.WHITE else chess.square_mirror(square)


# --------------------------------------------------------------------------- #
# action encoding
# --------------------------------------------------------------------------- #
def encode_action(move: chess.Move, side_to_move: bool) -> int:
    """Encode a ``chess.Move`` (in real coordinates) to a canonical action id."""
    f = canonical_square(move.from_square, side_to_move)
    t = canonical_square(move.to_square, side_to_move)
    p = PIECE_TO_PROMO_INDEX[move.promotion]
    return f * (NUM_SQUARES * NUM_PROMO) + t * NUM_PROMO + p


def decode_action(action: int, board: chess.Board) -> chess.Move:
    """Decode a canonical action id back into a ``chess.Move`` for ``board``.

    ``board`` supplies the side to move used for the orientation mapping.  The
    returned move is in real coordinates and is *not* validated for legality;
    use :func:`legal_action_mask` for that.
    """
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"action {action} out of range [0, {NUM_ACTIONS})")
    p = action % NUM_PROMO
    t = (action // NUM_PROMO) % NUM_SQUARES
    f = action // (NUM_SQUARES * NUM_PROMO)
    stm = board.turn
    return chess.Move(real_square(f, stm), real_square(t, stm), promotion=PROMO_INDEX_TO_PIECE[p])


def get_valid_moves(board: chess.Board) -> list:
    """Return the list of legal moves for ``board`` (real coordinates)."""
    return list(board.legal_moves)


def legal_action_ids(board: chess.Board) -> list:
    stm = board.turn
    return [encode_action(m, stm) for m in board.legal_moves]


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape (NUM_ACTIONS,) marking legal canonical actions."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    stm = board.turn
    for m in board.legal_moves:
        mask[encode_action(m, stm)] = True
    return mask


def masked_softmax(logits, mask) -> np.ndarray:
    """Softmax over ``logits`` restricted to ``mask``; returns a proper pmf.

    Illegal entries are exactly zero.  If the mask is empty an all-zero vector
    is returned (callers must handle terminal states before calling this).
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    out = np.zeros_like(logits)
    if not mask.any():
        return out
    sel = logits[mask]
    sel = sel - sel.max()
    e = np.exp(sel)
    s = e.sum()
    if not np.isfinite(s) or s <= 0.0:
        out[mask] = 1.0 / mask.sum()
        return out
    out[mask] = e / s
    return out


# --------------------------------------------------------------------------- #
# board encoding
# --------------------------------------------------------------------------- #
def encode_board(board: chess.Board) -> np.ndarray:
    """Encode ``board`` into the canonical (8, 8, 20) observation tensor."""
    stm = board.turn
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)

    for sq, piece in board.piece_map().items():
        c = canonical_square(sq, stm)
        rank, file = chess.square_rank(c), chess.square_file(c)
        plane = _PIECE_ORDER.index(piece.piece_type)
        if piece.color != stm:
            plane += 6
        obs[rank, file, plane] = 1.0

    obs[:, :, 12] = 1.0

    opp = not stm
    obs[:, :, 13] = 1.0 if board.has_kingside_castling_rights(stm) else 0.0
    obs[:, :, 14] = 1.0 if board.has_queenside_castling_rights(stm) else 0.0
    obs[:, :, 15] = 1.0 if board.has_kingside_castling_rights(opp) else 0.0
    obs[:, :, 16] = 1.0 if board.has_queenside_castling_rights(opp) else 0.0

    if board.ep_square is not None:
        c = canonical_square(board.ep_square, stm)
        obs[chess.square_rank(c), chess.square_file(c), 17] = 1.0

    obs[:, :, 18] = min(board.halfmove_clock, 100) / 100.0
    obs[:, :, 19] = 1.0 if board.is_repetition(2) else 0.0
    return obs


def search_key(board: chess.Board):
    """Transposition / cache key.

    Includes the position key *and* draw-relevant status so that states with
    different repetition or fifty-move status are never silently merged.
    """
    return (
        board._transposition_key(),
        board.halfmove_clock,
        board.is_repetition(2),
    )
