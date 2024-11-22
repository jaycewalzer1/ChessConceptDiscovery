import open_spiel
import numpy as np

class ChessEnv(open_spiel.python.games.ChessGame):
    def __init__(self):
        super(ChessEnv, self).__init__()

    def reset(self):
        self.state = self.new_initial_state()
        return np.array(self.state.observation())

    def step(self, action):
        self.state.apply_action(action)
        done = self.state.is_terminal()
        reward = self.state.reward(self.state.current_player())
        return np.array(self.state.observation()), reward, done

    def is_done(self):
        return self.state.is_terminal()

class ChessEnv:
    def calculate_development(self, board):
        # Example function to calculate development score
        development_score = 0
        # Add pawns in advanced ranks
        development_score += sum(1 for square in board.pieces(chess.PAWN, chess.WHITE) if square >= 24)
        # Add minor pieces in active squares
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        development_score += sum(1 for square in board.pieces(chess.KNIGHT, chess.WHITE) if square in center_squares)
        development_score += sum(1 for square in board.pieces(chess.BISHOP, chess.WHITE) if square in center_squares)
        # Check if the king has castled
        if board.has_castling_rights(chess.WHITE):
            development_score += 1
        return development_score
