"""AlphaZero-style agent.

REPAIR NOTES (relative to the original repository)
--------------------------------------------------
The original ``agent.py`` could not be imported at all:

  * ``from mcts import MCTS`` -- no ``mcts`` module existed in the repository;
  * ``from neural_network import ...`` / ``from chess_env import ...`` -- flat
    imports that only work if those directories are on ``sys.path``, which the
    package layout never arranged;
  * ``select_action`` called ``self.model.predict(state)`` and then
    ``self.mcts.search(state, policy)``, i.e. it computed a policy, threw it
    away, and called a method that did not exist;
  * ``train`` looped over raw episode tuples and called a ``NeuralNetwork.train``
    that was a ``pass`` stub;
  * ``load_checkpoint`` delegated to a broken ``load_weights``.

The class name ``AlphaZeroAgent`` and the ``select_action`` / ``train`` /
``save_checkpoint`` / ``load_checkpoint`` surface are preserved.
"""

from __future__ import annotations

import numpy as np

from agents.alphazero.mcts import MCTS
from agents.alphazero.neural_network import NeuralNetwork
from envs.chess_env import ChessEnv
from envs.chess_utils import NUM_ACTIONS, masked_softmax


class AlphaZeroAgent:
    def __init__(
        self,
        board_size: int = 8,
        action_space: int = NUM_ACTIONS,
        model_checkpoint: str | None = None,
        model: NeuralNetwork | None = None,
        simulations: int = 200,
        c_puct: float = 1.5,
        max_plies: int = 40,
        arch: dict | None = None,
    ):
        self.board_size = board_size
        self.action_space = action_space
        if model is not None:
            self.model = model
        elif model_checkpoint:
            self.model = NeuralNetwork.from_checkpoint(model_checkpoint)
        else:
            self.model = NeuralNetwork(arch=arch)
        self.simulations = simulations
        self.mcts = MCTS(self.model, c_puct=c_puct, max_plies=max_plies)

    # ---------------------------------------------------------------- acting
    def raw_policy(self, env: ChessEnv):
        """The network's policy over legal moves *without* any search."""
        logits, value = self.model.predict(env.observation())
        mask = env.legal_mask()
        probs = masked_softmax(logits, mask)
        legal = np.flatnonzero(mask)
        return {int(a): float(probs[a]) for a in legal}, float(value)

    def select_action(self, env: ChessEnv, simulations: int | None = None,
                      temperature: float = 0.0, add_root_noise: bool = False,
                      rng=None, return_root: bool = False):
        """Run MCTS from ``env``'s current state and return an action id."""
        n = self.simulations if simulations is None else simulations
        root, stats = self.mcts.run(
            env, n, add_root_noise=add_root_noise, rng=rng
        )
        action = MCTS.select_move(root, temperature=temperature, rng=rng)
        if return_root:
            return action, root, stats
        return action

    # --------------------------------------------------------------- learning
    def train(self, batch, value_coef: float = 1.0):
        """Train on a prepared minibatch dict (see ``NeuralNetwork.train``)."""
        return self.model.train(batch, value_coef=value_coef)

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path, extra: dict | None = None):
        return self.model.save_weights(path, extra=extra)

    def load_checkpoint(self, path, restore_optimizer: bool = False):
        self.model.load_weights(path, restore_optimizer=restore_optimizer)
        self.mcts = MCTS(self.model, c_puct=self.mcts.c_puct, max_plies=self.mcts.max_plies)
        return self
