"""Evaluate a frozen checkpoint on a set of verified forced-mate tasks.

REPAIR NOTES: the original imported ``envs.chess.chess_env`` (nonexistent),
constructed the agent with a 64-wide action space, and called
``agent.select_action(state)`` / ``env.step(action)`` with mismatched
signatures and a 2-tuple unpack of a 3-tuple return.
"""

from __future__ import annotations

import argparse
import json

from agents.alphazero.agent import AlphaZeroAgent
from agents.alphazero.mcts import MCTS
from agents.alphazero.neural_network import NeuralNetwork
from envs.chess_env import ChessEnv


def eval_agent(checkpoint: str, positions_file: str, simulations: int = 400,
               limit: int | None = None):
    agent = AlphaZeroAgent(model=NeuralNetwork.from_checkpoint(checkpoint),
                           simulations=simulations)
    tasks = [json.loads(l) for l in open(positions_file) if l.strip()]
    if limit:
        tasks = tasks[:limit]

    solved = scored = 0
    for t in tasks:
        env = ChessEnv(fen=t["fen"])
        action, root, stats = agent.select_action(env, return_root=True)
        move = env.decode(action).uci()
        if "solution_moves" in t:  # labels present
            scored += 1
            solved += move in t["solution_moves"]
        print(f"{t['task_id']}: {move} "
              f"(sims={stats.simulations}, nn_evals={stats.nn_evals}, "
              f"{stats.elapsed_s:.2f}s)")
    if scored:
        print(f"\nsolved {solved}/{scored} = {100*solved/scored:.1f}%")
    return solved, scored


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--positions", required=True,
                   help="a *_positions.jsonl or *_labels-joined jsonl file")
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    eval_agent(a.checkpoint, a.positions, a.simulations, a.limit)
