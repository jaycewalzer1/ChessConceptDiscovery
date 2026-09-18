"""Training: supervised tactical bootstrapping + AlphaZero-style self-play.

REPAIR NOTES (relative to the original repository)
--------------------------------------------------
The original ``TrainingLoop.run_episode`` collected ``(state, action, reward,
next_state)`` tuples -- a *DQN-style* transition, not AlphaZero supervision.  It
had no MCTS visit-count policy target, no game-outcome value target with the
correct per-ply perspective, no replay buffer, no minibatching, and it called an
``agent.train`` that was a stub.  ``run`` also saved a checkpoint through a
broken ``save_weights``.

The class name ``TrainingLoop`` and its ``run`` / ``run_episode`` surface are
preserved.  What is added: a bounded replay buffer, proper AlphaZero records
(observation, legal mask, MCTS visit-count policy target, value target with an
explicit *known/unknown* weight), minibatch SGD, and an honest treatment of
truncated games.

TRUNCATION POLICY
-----------------
A game stopped by the ply cap has an **unknown** game-theoretic outcome.  Its
records are stored with ``v_weight = 0`` so the value head receives no
supervision from them, instead of pretending the game was a draw.  The policy
target from MCTS visit counts is still valid and is kept.

TACTICAL BOOTSTRAP
------------------
``SupervisedBootstrap`` trains the policy head on root moves that the exact
forced-mate verifier proved to be winning, and the value head on the verified
decisive value (+1 for the side to move in a position with a forced mate).
This is **supervised tactical bootstrapping**, not tabula-rasa AlphaZero, and
the report labels it as such.  The value target is only ever applied where the
verifier returned status ``ok``; positions with unknown game-theoretic value get
``v_weight = 0``.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from agents.alphazero.mcts import MCTS
from envs.chess_env import ChessEnv
from envs.chess_utils import NUM_ACTIONS, encode_board, legal_action_mask


class ReplayBuffer:
    def __init__(self, capacity: int = 20000, rng=None):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)
        self.rng = rng or np.random.default_rng(0)

    def add(self, record):
        self.data.append(record)

    def extend(self, records):
        for r in records:
            self.data.append(r)

    def __len__(self):
        return len(self.data)

    def sample_batch(self, batch_size: int):
        n = min(batch_size, len(self.data))
        idx = self.rng.choice(len(self.data), size=n, replace=False)
        items = [self.data[i] for i in idx]
        return collate(items)


def collate(items):
    obs = np.stack([it["obs"] for it in items]).astype(np.float32)
    mask = np.stack([it["mask"] for it in items]).astype(bool)
    pi = np.zeros((len(items), NUM_ACTIONS), dtype=np.float32)
    for i, it in enumerate(items):
        for a, p in it["pi"].items():
            pi[i, a] = p
        s = pi[i].sum()
        if s > 0:
            pi[i] /= s
    v = np.asarray([it["v"] for it in items], dtype=np.float32)
    vw = np.asarray([it["v_weight"] for it in items], dtype=np.float32)
    return {"obs": obs, "mask": mask, "pi": pi, "v": v, "v_weight": vw}


def make_record(board, pi_dict, v, v_weight):
    return {
        "obs": encode_board(board),
        "mask": legal_action_mask(board),
        "pi": pi_dict,
        "v": float(v),
        "v_weight": float(v_weight),
    }


# --------------------------------------------------------------------------- #
# stage 1: supervised tactical bootstrapping on verifier-labelled positions
# --------------------------------------------------------------------------- #
class SupervisedBootstrap:
    """Train the policy/value heads on exactly-verified forced-mate tasks."""

    def __init__(self, agent, rng=None, label_smoothing: float = 0.02):
        self.agent = agent
        self.rng = rng or np.random.default_rng(0)
        self.label_smoothing = label_smoothing

    def build_records(self, tasks):
        """``tasks`` are dicts with ``fen`` and ``solution_moves`` (UCI)."""
        import chess

        records = []
        for t in tasks:
            board = chess.Board(t["fen"])
            sols = [chess.Move.from_uci(u) for u in t["solution_moves"]]
            sols = [m for m in sols if m in board.legal_moves]
            if not sols:
                continue
            from envs.chess_utils import encode_action

            legal = list(board.legal_moves)
            pi = {}
            eps = self.label_smoothing
            for m in legal:
                pi[encode_action(m, board.turn)] = eps / max(len(legal), 1)
            for m in sols:
                pi[encode_action(m, board.turn)] += (1.0 - eps) / len(sols)
            # A verified forced mate is a decisive win for the side to move.
            records.append(make_record(board, pi, v=1.0, v_weight=1.0))
        return records

    def run(self, records, epochs: int = 8, batch_size: int = 64, lr: float = 1e-3,
            value_coef: float = 1.0, time_budget_s: float | None = None, log=print):
        if not records:
            return {"steps": 0, "history": [], "note": "no records"}
        self.agent.model.set_learning_rate(lr)
        history, steps = [], 0
        t0 = time.perf_counter()
        n = len(records)
        for ep in range(epochs):
            order = self.rng.permutation(n)
            ep_losses = []
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = collate([records[i] for i in idx])
                out = self.agent.train(batch, value_coef=value_coef)
                ep_losses.append(out)
                steps += 1
                if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
                    log(f"[bootstrap] time budget hit at epoch {ep}, step {steps}")
                    history.append(_avg(ep_losses, ep, steps))
                    return {"steps": steps, "history": history, "stopped_early": True,
                            "elapsed_s": time.perf_counter() - t0}
            rec = _avg(ep_losses, ep, steps)
            history.append(rec)
            log(f"[bootstrap] epoch {ep+1}/{epochs} loss={rec['loss']:.4f} "
                f"policy={rec['policy_loss']:.4f} value={rec['value_loss']:.4f}")
        return {"steps": steps, "history": history, "stopped_early": False,
                "elapsed_s": time.perf_counter() - t0}


def _avg(losses, epoch, steps):
    keys = ["loss", "policy_loss", "value_loss", "grad_norm"]
    out = {k: float(np.mean([l[k] for l in losses])) for k in keys}
    out["epoch"] = epoch
    out["steps"] = steps
    return out


# --------------------------------------------------------------------------- #
# stage 2: AlphaZero-style self-play fine-tuning
# --------------------------------------------------------------------------- #
class TrainingLoop:
    """Self-play + replay training loop (repaired)."""

    def __init__(self, agent, env: ChessEnv, num_iterations: int = 10,
                 checkpoint_dir: str = "checkpoints", simulations: int = 64,
                 games_per_iter: int = 4, max_game_plies: int = 40,
                 batch_size: int = 64, train_steps_per_iter: int = 20,
                 temperature_plies: int = 6, rng=None, log=print):
        self.agent = agent
        self.env = env
        self.num_iterations = num_iterations
        self.checkpoint_dir = checkpoint_dir
        self.simulations = simulations
        self.games_per_iter = games_per_iter
        self.max_game_plies = max_game_plies
        self.batch_size = batch_size
        self.train_steps_per_iter = train_steps_per_iter
        self.temperature_plies = temperature_plies
        self.rng = rng or np.random.default_rng(0)
        self.buffer = ReplayBuffer(capacity=20000, rng=self.rng)
        self.log = log

    def run(self, start_fens=None, time_budget_s: float | None = None):
        t0 = time.perf_counter()
        summary = {"iterations": [], "games": 0, "records": 0, "stopped_early": False}
        for it in range(self.num_iterations):
            games, recs = 0, 0
            for g in range(self.games_per_iter):
                fen = None
                if start_fens:
                    fen = start_fens[int(self.rng.integers(len(start_fens)))]
                records, info = self.run_episode(fen)
                self.buffer.extend(records)
                games += 1
                recs += len(records)
                if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
                    break
            losses = []
            if len(self.buffer) >= 8:
                for _ in range(self.train_steps_per_iter):
                    batch = self.buffer.sample_batch(self.batch_size)
                    losses.append(self.agent.train(batch))
            rec = {
                "iteration": it,
                "games": games,
                "new_records": recs,
                "buffer": len(self.buffer),
                "loss": float(np.mean([l["loss"] for l in losses])) if losses else None,
                "policy_loss": float(np.mean([l["policy_loss"] for l in losses])) if losses else None,
                "value_loss": float(np.mean([l["value_loss"] for l in losses])) if losses else None,
                "elapsed_s": time.perf_counter() - t0,
            }
            summary["iterations"].append(rec)
            summary["games"] += games
            summary["records"] += recs
            self.log(f"[selfplay] iter {it+1}/{self.num_iterations} games={games} "
                     f"buffer={len(self.buffer)} loss={rec['loss']}")
            if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
                summary["stopped_early"] = True
                self.log("[selfplay] time budget hit, stopping")
                break
        summary["elapsed_s"] = time.perf_counter() - t0
        return summary

    def run_episode(self, fen=None):
        """Play one self-play game, returning AlphaZero training records.

        Records carry the MCTS visit-count policy target and, once the game
        ends, the outcome value target from the perspective of the player to
        move at that record's state.  Truncated games leave ``v_weight = 0``.
        """
        env = ChessEnv(fen=fen, max_plies=self.max_game_plies)
        env.reset()
        mcts = MCTS(self.agent.model, c_puct=self.agent.mcts.c_puct, max_plies=self.agent.mcts.max_plies)
        pending, players = [], []
        ply = 0
        while not env.is_done():
            root, _ = mcts.run(
                env, self.simulations, add_root_noise=True, rng=self.rng
            )
            if not root.children:
                break
            total = sum(c.visit_count for c in root.children.values())
            if total == 0:
                break
            pi = {int(a): c.visit_count / total for a, c in root.children.items() if c.visit_count > 0}
            pending.append(make_record(env.board, pi, v=0.0, v_weight=0.0))
            players.append(env.current_player)
            temp = 1.0 if ply < self.temperature_plies else 0.0
            action = MCTS.select_move(root, temperature=temp, rng=self.rng)
            env.step(action)
            ply += 1

        done, kind, result = env.terminal_info()
        info = {"plies": ply, "terminal_kind": kind, "result": result, "fen": env.fen()}
        if kind == "rules" and result is not None:
            import chess

            for rec, player in zip(pending, players):
                rec["v"] = result if player == chess.WHITE else -result
                rec["v_weight"] = 1.0
        # else: truncated -> unknown outcome, v_weight stays 0 (masked)
        return pending, info
