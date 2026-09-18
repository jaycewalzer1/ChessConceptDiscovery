"""Standalone self-play training entry point.

REPAIR NOTES: the original imported ``envs.chess.chess_env`` (a path that never
existed) and called ``env.num_actions()`` (never defined), so the module could
not be imported.  The full experiment uses ``scripts.run_experiment``; this
script is kept as a minimal way to run stage 2 (self-play) on its own.
"""

from __future__ import annotations

import argparse
import os

from agents.alphazero.agent import AlphaZeroAgent
from agents.alphazero.training import TrainingLoop
from envs.chess_env import ChessEnv
from envs.chess_utils import NUM_ACTIONS


def run_training(iterations: int = 10, checkpoint_dir: str = "checkpoints",
                 simulations: int = 48, games_per_iter: int = 4,
                 minutes: float | None = None, seed: int = 0):
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent = AlphaZeroAgent(board_size=8, action_space=NUM_ACTIONS, simulations=simulations)
    loop = TrainingLoop(agent, ChessEnv(), num_iterations=iterations,
                        checkpoint_dir=checkpoint_dir, simulations=simulations,
                        games_per_iter=games_per_iter)
    summary = loop.run(time_budget_s=None if minutes is None else minutes * 60)
    agent.save_checkpoint(os.path.join(checkpoint_dir, "selfplay"))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--simulations", type=int, default=48)
    p.add_argument("--games-per-iter", type=int, default=4)
    p.add_argument("--minutes", type=float, default=None)
    a = p.parse_args()
    run_training(a.iterations, a.checkpoint_dir, a.simulations,
                 a.games_per_iter, a.minutes)
