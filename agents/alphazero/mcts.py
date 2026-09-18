"""Neural-network-guided PUCT Monte-Carlo tree search.

NEW module.  ``agents/alphazero/agent.py`` in the original repository did
``from mcts import MCTS`` but no ``mcts`` module existed anywhere in the tree,
so the agent could never be imported.

Design notes
------------
* Value convention: **value for the player to move** at every node.  Backing up
  an alternating-turn path therefore negates the value at each ply.
* Terminal handling is explicit: a genuine rules terminal contributes its exact
  value (``+1/-1/0`` for the side to move there); a *truncated* node (ply cap)
  contributes the network's value estimate and is counted separately.  A
  truncation is never reported as a real draw.
* Every search records real instrumentation: simulations, network-evaluated
  positions, inference batches, terminal expansions, cache hits, environment
  transitions, peak tree size and wall time.  Loop iterations are never
  reported as "equivalent compute".
* Dirichlet root noise and temperature are *training-only* knobs and default to
  off so that evaluation is deterministic given a seed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from envs.chess_utils import masked_softmax


@dataclass
class SearchStats:
    simulations: int = 0
    nn_evals: int = 0
    nn_batches: int = 0
    terminal_expansions: int = 0
    cache_hits: int = 0
    env_transitions: int = 0
    peak_tree_size: int = 0
    elapsed_s: float = 0.0
    truncations: int = 0

    def merge(self, other: "SearchStats") -> "SearchStats":
        return SearchStats(
            simulations=self.simulations + other.simulations,
            nn_evals=self.nn_evals + other.nn_evals,
            nn_batches=self.nn_batches + other.nn_batches,
            terminal_expansions=self.terminal_expansions + other.terminal_expansions,
            cache_hits=self.cache_hits + other.cache_hits,
            env_transitions=self.env_transitions + other.env_transitions,
            peak_tree_size=max(self.peak_tree_size, other.peak_tree_size),
            elapsed_s=self.elapsed_s + other.elapsed_s,
            truncations=self.truncations + other.truncations,
        )

    def as_dict(self):
        return dict(self.__dict__)


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "expanded", "terminal_value", "is_truncated")

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action_id -> Node
        self.expanded = False
        self.terminal_value = None  # value for the player to move at this node
        self.is_truncated = False

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    """PUCT search over a :class:`envs.chess_env.ChessEnv`."""

    def __init__(
        self,
        model,
        c_puct: float = 1.5,
        max_plies: int = 40,
        use_cache: bool = True,
        fpu_reduction: float = 0.25,
    ):
        self.model = model
        self.c_puct = c_puct
        self.max_plies = max_plies
        self.use_cache = use_cache
        self.fpu_reduction = fpu_reduction
        self._cache = {}
        self.stats = SearchStats()

    # -------------------------------------------------------------- utilities
    def clear_cache(self):
        self._cache.clear()

    def reset_stats(self):
        self.stats = SearchStats()

    def _evaluate(self, env, stats: SearchStats):
        """Network evaluation with an optional transposition cache.

        Returns ``(priors_over_legal_actions, value_for_player_to_move)``.
        Cache hits are counted but still charged as a logical evaluation by the
        caller's accounting (see ``nn_evals`` vs ``cache_hits``).
        """
        key = env.key() if self.use_cache else None
        if key is not None and key in self._cache:
            stats.cache_hits += 1
            return self._cache[key]
        obs = env.observation()
        logits, value = self.model.predict(obs)
        stats.nn_evals += 1
        stats.nn_batches += 1
        mask = env.legal_mask()
        priors = masked_softmax(logits, mask)
        legal = np.flatnonzero(mask)
        result = ({int(a): float(priors[a]) for a in legal}, float(value))
        if key is not None:
            self._cache[key] = result
        return result

    def _expand(self, node: Node, env, stats: SearchStats):
        done, kind, _ = env.terminal_info()
        if done and kind == "rules":
            node.terminal_value = env.terminal_value_for_player_to_move()
            node.expanded = True
            stats.terminal_expansions += 1
            return node.terminal_value
        if done and kind == "truncated":
            node.is_truncated = True
            node.expanded = True
            stats.truncations += 1
            _, value = self._evaluate(env, stats)
            return value
        priors, value = self._evaluate(env, stats)
        for a, p in priors.items():
            node.children[a] = Node(prior=p)
        node.expanded = True
        return value

    def _select_child(self, node: Node):
        total = max(node.visit_count, 1)
        sqrt_total = math.sqrt(total)
        parent_q = node.value
        best, best_score = None, -float("inf")
        for a, child in node.children.items():
            if child.visit_count == 0:
                q = -parent_q - self.fpu_reduction
            else:
                q = -child.value  # child's value is for the *child's* mover
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best, best_score = a, score
        return best, node.children[best]

    # ----------------------------------------------------------------- search
    def run(
        self,
        env,
        simulations: int,
        root: Node | None = None,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        noise_frac: float = 0.25,
        rng: np.random.Generator | None = None,
        stats: SearchStats | None = None,
    ):
        """Run ``simulations`` PUCT simulations from ``env``'s current state.

        ``root`` may be a previously built root to continue searching (used by
        the allocation conditions so screening work is reused *and charged*).
        """
        stats = stats if stats is not None else SearchStats()
        t0 = time.perf_counter()
        root = root if root is not None else Node()

        if not root.expanded:
            v = self._expand(root, env, stats)
            root.visit_count += 1
            root.value_sum += v

        if add_root_noise and root.children:
            rng = rng or np.random.default_rng()
            actions = list(root.children)
            noise = rng.dirichlet([dirichlet_alpha] * len(actions))
            for a, n in zip(actions, noise):
                c = root.children[a]
                c.prior = (1 - noise_frac) * c.prior + noise_frac * float(n)

        for _ in range(simulations):
            node = root
            search_env = env.clone()
            path = [node]
            depth = 0
            while node.expanded and node.children and node.terminal_value is None and not node.is_truncated:
                action, node = self._select_child(node)
                search_env.step(action)
                stats.env_transitions += 1
                path.append(node)
                depth += 1
                if depth >= self.max_plies:
                    break

            if node.terminal_value is not None:
                value = node.terminal_value
            elif not node.expanded:
                value = self._expand(node, search_env, stats)
            else:
                value = node.value

            # Back up, negating at every ply (value is always for the mover).
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value
            stats.simulations += 1
            stats.peak_tree_size = max(stats.peak_tree_size, self._tree_size(root))

        stats.elapsed_s += time.perf_counter() - t0
        self.stats = self.stats.merge(SearchStats(elapsed_s=0.0))
        return root, stats

    @staticmethod
    def _tree_size(root: Node) -> int:
        # Cheap upper-level estimate: root children + their children.
        n = 1 + len(root.children)
        for c in root.children.values():
            n += len(c.children)
        return n

    # ----------------------------------------------------- move selection APIs
    @staticmethod
    def visit_counts(root: Node):
        return {a: c.visit_count for a, c in root.children.items()}

    @staticmethod
    def select_move(root: Node, temperature: float = 0.0, rng: np.random.Generator | None = None):
        """Select a root action from visit counts.

        ``temperature == 0`` is deterministic argmax with a stable tie-break on
        (visit_count, mean value, action id).
        """
        if not root.children:
            return None
        items = list(root.children.items())
        if temperature <= 0:
            best = max(items, key=lambda kv: (kv[1].visit_count, -kv[1].value, -kv[0]))
            return best[0]
        rng = rng or np.random.default_rng()
        actions = np.array([a for a, _ in items])
        counts = np.array([c.visit_count for _, c in items], dtype=np.float64)
        if counts.sum() == 0:
            probs = np.ones_like(counts) / len(counts)
        else:
            adj = counts ** (1.0 / temperature)
            probs = adj / adj.sum()
        return int(rng.choice(actions, p=probs))

    @staticmethod
    def policy_target(root: Node, num_actions: int) -> np.ndarray:
        pi = np.zeros(num_actions, dtype=np.float32)
        total = sum(c.visit_count for c in root.children.values())
        if total == 0:
            return pi
        for a, c in root.children.items():
            pi[a] = c.visit_count / total
        return pi

    def principal_variation(self, env, root: Node, max_len: int = 6):
        """Extract the search's own PV (most-visited path) as SAN + boards."""
        work = env.clone()
        node = root
        sans, fens = [], [work.fen()]
        for _ in range(max_len):
            if not node.children:
                break
            action = max(node.children.items(), key=lambda kv: kv[1].visit_count)[0]
            if node.children[action].visit_count == 0:
                break
            move = work.decode(action)
            if move not in work.board.legal_moves:
                break
            sans.append(work.san(move))
            work.step(action)
            fens.append(work.fen())
            node = node.children[action]
            if work.is_done():
                break
        return sans, fens
