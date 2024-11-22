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
