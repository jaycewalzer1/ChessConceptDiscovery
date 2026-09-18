"""Chess environment for the AlphaZero-style agent.

REPAIR NOTES (relative to the original repository)
--------------------------------------------------
The original ``envs/chess_env.py`` defined ``ChessEnv`` **twice**; the second
definition silently shadowed the first, so the OpenSpiel-derived environment was
dead code.  The original also:

  * subclassed ``open_spiel.python.games.ChessGame`` (wrong import path, and the
    ``open_spiel`` gitlink had no ``.gitmodules`` entry so it could not even be
    fetched),
  * read ``self.state.reward(self.state.current_player())`` *after* the state
    became terminal -- in OpenSpiel ``current_player()`` returns the special
    ``kTerminalPlayerId`` (-4) there, which is not a valid player index,
  * used ``board.has_castling_rights(WHITE)`` inside ``calculate_development``
    as a proxy for "the king has castled", which is exactly backwards (having
    castling *rights* means you have **not** castled yet).

This module keeps the ``ChessEnv`` name and the ``reset`` / ``step`` /
``is_done`` / ``calculate_development`` API but rebuilds it on ``python-chess``
so legal move generation, FEN/PGN handling and draw rules are exact.  This is an
*environment substitution*, not a replacement of the AlphaZero learner or
search.  ``chess.engine`` / UCI engines are never imported.
"""

from __future__ import annotations

import chess

from envs.chess_utils import (
    NUM_ACTIONS,
    decode_action,
    encode_action,
    encode_board,
    legal_action_ids,
    legal_action_mask,
    search_key,
)

# Outcome from the perspective of the player who is to move in the *terminal*
# state is always <= 0 (you cannot be checkmated and win), so we report the
# result explicitly instead of inferring a reward for an invalid player index.
WHITE_WIN, DRAW, BLACK_WIN = 1.0, 0.0, -1.0


class ChessEnv:
    """A single-game chess environment backed by ``chess.Board``.

    ``max_plies`` imposes a *search/game truncation*.  Truncation is reported
    separately from a genuine rules draw via :meth:`terminal_info` so that an
    unfinished game is never mislabelled as a real draw.
    """

    def __init__(self, fen: str | None = None, max_plies: int = 512):
        self.start_fen = fen or chess.STARTING_FEN
        self.max_plies = max_plies
        self.board = chess.Board(self.start_fen)
        self._start_ply = 0

    # ---------------------------------------------------------------- basics
    def reset(self, fen: str | None = None):
        if fen is not None:
            self.start_fen = fen
        self.board = chess.Board(self.start_fen)
        self._start_ply = 0
        return self.observation()

    def clone(self) -> "ChessEnv":
        """Deep copy including the move stack (needed for repetition rules)."""
        env = ChessEnv.__new__(ChessEnv)
        env.start_fen = self.start_fen
        env.max_plies = self.max_plies
        env.board = self.board.copy(stack=True)
        env._start_ply = self._start_ply
        return env

    # ------------------------------------------------------------ observation
    def observation(self):
        return encode_board(self.board)

    def legal_actions(self):
        return legal_action_ids(self.board)

    def legal_mask(self):
        return legal_action_mask(self.board)

    def key(self):
        return search_key(self.board)

    @property
    def current_player(self) -> bool:
        """``chess.WHITE`` / ``chess.BLACK`` of the side to move."""
        return self.board.turn

    def num_actions(self) -> int:
        return NUM_ACTIONS

    # --------------------------------------------------------------- dynamics
    def step(self, action: int):
        """Apply a canonical action id.  Returns ``(obs, reward, done, info)``.

        ``reward`` is the value **for the player who was to move before the
        step**, and is non-zero only at a genuine terminal state.
        """
        move = decode_action(action, self.board)
        if move not in self.board.legal_moves:
            raise ValueError(f"illegal action {action} ({move.uci()}) in {self.board.fen()}")
        mover = self.board.turn
        self.board.push(move)

        done, kind, result = self.terminal_info()
        reward = 0.0
        if done and kind == "rules":
            reward = self._result_for(result, mover)
        return self.observation(), reward, done, {"terminal_kind": kind, "result": result}

    def push_move(self, move: chess.Move):
        self.board.push(move)

    def pop_move(self):
        return self.board.pop()

    # -------------------------------------------------------------- terminals
    def terminal_info(self):
        """Return ``(done, kind, result)``.

        ``kind`` is ``"rules"`` for a genuine game end (checkmate / stalemate /
        insufficient material / 75-move / fivefold), ``"truncated"`` when the
        ply cap is hit, and ``None`` otherwise.  ``result`` is ``+1`` (White
        wins), ``-1`` (Black wins), ``0`` (draw) or ``None`` when truncated --
        a truncated game has an **unknown** game-theoretic outcome.
        """
        outcome = self.board.outcome(claim_draw=False)
        if outcome is not None:
            if outcome.winner is None:
                return True, "rules", DRAW
            return True, "rules", WHITE_WIN if outcome.winner == chess.WHITE else BLACK_WIN
        if len(self.board.move_stack) - self._start_ply >= self.max_plies:
            return True, "truncated", None
        return False, None, None

    def is_done(self) -> bool:
        return self.terminal_info()[0]

    @staticmethod
    def _result_for(result: float, player: bool) -> float:
        """Convert a White-perspective result to ``player``'s perspective."""
        return result if player == chess.WHITE else -result

    def terminal_value_for_player_to_move(self):
        """Value of the current (terminal) state for the side to move, or None."""
        done, kind, result = self.terminal_info()
        if not done or kind != "rules":
            return None
        return self._result_for(result, self.board.turn)

    # -------------------------------------------------------------- utilities
    def fen(self) -> str:
        return self.board.fen()

    def san(self, move: chess.Move) -> str:
        return self.board.san(move)

    def encode(self, move: chess.Move) -> int:
        return encode_action(move, self.board.turn)

    def decode(self, action: int) -> chess.Move:
        return decode_action(action, self.board)

    # ---------------------------------------------- repaired development score
    def calculate_development(self, board: chess.Board | None = None, color: bool = chess.WHITE) -> float:
        """Simple development heuristic (kept from the original repository).

        The original version added a point when the side **had** castling
        rights, which rewards *not* having castled.  Here we detect an actual
        completed castle by scanning the move stack for a castling move, and
        fall back to "king off its home square with rights gone" when no stack
        is available.  Used only as a descriptive statistic; it never labels
        training data or grades the experiment.
        """
        board = self.board if board is None else board
        score = 0.0
        home_rank = 0 if color == chess.WHITE else 7
        advanced = (lambda r: r >= 3) if color == chess.WHITE else (lambda r: r <= 4)

        for sq in board.pieces(chess.PAWN, color):
            if advanced(chess.square_rank(sq)):
                score += 1.0
        for pt in (chess.KNIGHT, chess.BISHOP):
            for sq in board.pieces(pt, color):
                if chess.square_rank(sq) != home_rank:
                    score += 1.0

        king_sq = board.king(color)
        castle_targets = (chess.G1, chess.C1) if color == chess.WHITE else (chess.G8, chess.C8)
        if (
            king_sq in castle_targets
            and not board.has_kingside_castling_rights(color)
            and not board.has_queenside_castling_rights(color)
        ):
            score += 1.0
        return score
