"""Correctness tests targeting the actual failure risks of this project."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import chess

from agents.alphazero.allocators import AllocationConfig, clone_tree, run_condition
from agents.alphazero.features import FEATURE_NAMES, continuation_descriptor
from agents.alphazero.mcts import MCTS
from agents.alphazero.neural_network import NeuralNetwork
from agents.alphazero.training import SupervisedBootstrap, collate, make_record
from envs.chess_env import ChessEnv
from envs.chess_utils import (
    NUM_ACTIONS,
    decode_action,
    encode_action,
    encode_board,
    legal_action_mask,
    masked_softmax,
    search_key,
)
from envs.mate_solver import (
    classify_mate_task,
    is_forced_mate_in,
    refutation_line,
    solve_root_moves,
    winning_continuation,
)

# Fixture positions. These are implementation test fixtures ONLY; they are never
# part of the independently sampled research benchmark.
MATE_IN_1 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"          # Ra8#
BACK_RANK_2 = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"      # no mate (luft-free but blocked)
STALEMATE_TRAP = "k7/8/K7/8/8/8/8/1Q6 w - - 0 1"          # Qb7# mates; Qb4 stalemates
PROMO_POS = "8/P7/8/8/8/8/8/K6k w - - 0 1"
UNDERPROMO = "8/1P6/8/8/8/7k/8/K7 w - - 0 1"


# --------------------------------------------------------------------------- #
# action encoding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
        "8/PPPPPPPP/8/8/8/8/pppppppp/K6k w - - 0 1",
        "K6k/pppppppp/8/8/8/8/PPPPPPPP/8 b - - 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    ],
)
def test_action_roundtrip(fen):
    board = chess.Board(fen)
    seen = set()
    for move in board.legal_moves:
        a = encode_action(move, board.turn)
        assert 0 <= a < NUM_ACTIONS
        assert a not in seen, "action ids must be unique per legal move"
        seen.add(a)
        assert decode_action(a, board) == move


def test_underpromotion_ids_are_distinct():
    board = chess.Board(UNDERPROMO)
    promos = [m for m in board.legal_moves if m.promotion]
    assert len({m.promotion for m in promos}) == 4  # N, B, R, Q
    ids = {encode_action(m, board.turn) for m in promos}
    assert len(ids) == len(promos)


def test_legal_mask_and_masked_softmax():
    board = chess.Board(chess.STARTING_FEN)
    mask = legal_action_mask(board)
    assert mask.sum() == board.legal_moves.count() == 20
    rng = np.random.default_rng(0)
    logits = rng.normal(size=NUM_ACTIONS) * 50.0
    p = masked_softmax(logits, mask)
    assert np.isfinite(p).all()
    assert p[~mask].sum() == 0.0
    assert abs(p.sum() - 1.0) < 1e-9


def test_masked_softmax_handles_extreme_logits():
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[1, 2, 3]] = True
    logits = np.full(NUM_ACTIONS, -np.inf)
    logits[[1, 2, 3]] = [1e9, 1e9, 1e9]
    p = masked_softmax(logits, mask)
    assert np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# observation encoding
# --------------------------------------------------------------------------- #
def test_encoding_is_canonical_for_side_to_move():
    w = chess.Board("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    ow, ob = encode_board(w), encode_board(b)
    # Same symmetric position, opposite side to move -> identical canonical view
    assert np.allclose(ow[:, :, :12], ob[:, :, :12])


def test_encoding_records_ep_and_castling():
    b = chess.Board("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1")
    obs = encode_board(b)
    assert obs[:, :, 17].sum() == 1.0
    assert obs[0, 0, 13] == 1.0 and obs[0, 0, 15] == 1.0


# --------------------------------------------------------------------------- #
# environment / terminal handling
# --------------------------------------------------------------------------- #
def test_checkmate_vs_stalemate_values():
    mate = ChessEnv(fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    a = mate.encode(chess.Move.from_uci("a1a8"))
    _, reward, done, info = mate.step(a)
    assert done and info["terminal_kind"] == "rules"
    assert reward == 1.0  # White (the mover) wins
    assert mate.terminal_value_for_player_to_move() == -1.0  # Black is mated

    stale = ChessEnv(fen=STALEMATE_TRAP)
    a = stale.encode(chess.Move.from_uci("b1b4"))  # stalemates Black
    _, reward, done, info = stale.step(a)
    assert done and info["terminal_kind"] == "rules" and reward == 0.0
    assert stale.terminal_value_for_player_to_move() == 0.0


def test_truncation_is_not_a_draw():
    env = ChessEnv(max_plies=2)
    env.step(env.encode(chess.Move.from_uci("e2e4")))
    obs, reward, done, info = env.step(env.encode(chess.Move.from_uci("e7e5")))
    assert done and info["terminal_kind"] == "truncated"
    assert info["result"] is None and reward == 0.0
    assert env.terminal_value_for_player_to_move() is None


def test_clone_preserves_history_and_isolates_mutation():
    env = ChessEnv()
    env.step(env.encode(chess.Move.from_uci("e2e4")))
    clone = env.clone()
    assert len(clone.board.move_stack) == len(env.board.move_stack)
    clone.step(clone.encode(chess.Move.from_uci("e7e5")))
    assert len(env.board.move_stack) == 1
    assert clone.fen() != env.fen()


def test_search_key_separates_repetition_status():
    env = ChessEnv(fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    k0 = env.key()
    for uci in ["e1d1", "e8d8", "d1e1", "d8e8"]:
        env.step(env.encode(chess.Move.from_uci(uci)))
    k1 = env.key()
    assert k0 != k1, "repetition/fifty-move status must not be merged"


def test_illegal_action_rejected():
    env = ChessEnv()
    with pytest.raises(ValueError):
        env.step(encode_action(chess.Move.from_uci("e2e5"), chess.WHITE))


def test_development_heuristic_rewards_actual_castling():
    not_castled = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    castled = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R4RK1 w kq - 0 1")
    env = ChessEnv()
    assert env.calculate_development(castled) > env.calculate_development(not_castled)


# --------------------------------------------------------------------------- #
# forced-mate verifier
# --------------------------------------------------------------------------- #
def test_verifier_accepts_real_mate_in_one():
    board = chess.Board(MATE_IN_1)
    res = solve_root_moves(board, 1)
    assert res.status == "ok"
    assert "a1a8" in res.solution_moves


def test_verifier_returns_complete_solution_set():
    """The solver must return EVERY mating root move, not just the first one."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/Q5RK w - - 0 1")
    res = solve_root_moves(board, 1)
    assert res.status == "ok"
    # Independent brute-force ground truth for n = 1.
    truth = set()
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            truth.add(m.uci())
        board.pop()
    assert set(res.solution_moves) == truth
    assert len(truth) >= 2, "fixture must have multiple solutions to be meaningful"


def test_verifier_rejects_line_with_defensive_escape():
    # White has a check but Black escapes; no mate in 1.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1")
    res = solve_root_moves(board, 1)
    assert res.status == "ok" and res.solution_moves == []


def test_verifier_requires_universal_quantification():
    # White can check with Ra7 but Black has many replies; not mate in 1.
    board = chess.Board("6k1/8/8/8/8/8/8/R5K1 w - - 0 1")
    res1 = solve_root_moves(board, 1)
    assert "a1a7" not in res1.solution_moves


def test_verifier_stalemate_refutes_mate_claim():
    # If the attacker's "mating" try stalemates, it is not a mate.
    board = chess.Board(STALEMATE_TRAP)
    res = solve_root_moves(board, 1)
    assert res.status == "ok"
    assert "b1b4" not in res.solution_moves  # stalemate, not mate
    assert "b1b7" in res.solution_moves  # genuine mate


def test_classify_mate_task_finds_shortest():
    board = chess.Board(MATE_IN_1)
    res = classify_mate_task(board, max_n=3)
    assert res.status == "ok" and res.mate_in == 1


def test_solver_reports_unknown_on_tiny_budget():
    board = chess.Board(chess.STARTING_FEN)
    res = solve_root_moves(board, 3, node_budget=50)
    assert res.status == "unknown" and res.solution_moves == []


def test_refutation_line_for_non_solution():
    board = chess.Board(MATE_IN_1)
    line = refutation_line(board, "a1a7", 1)
    assert len(line) == 2  # White's try + a Black escape


def test_winning_continuation_is_legal():
    board = chess.Board(MATE_IN_1)
    line = winning_continuation(board, "a1a8", 1)
    work = chess.Board(MATE_IN_1)
    for san in line:
        work.push_san(san)
    assert work.is_checkmate()


# --------------------------------------------------------------------------- #
# network + checkpoints
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_model():
    return NeuralNetwork(arch={"channels": 16, "blocks": 1, "value_hidden": 16})


def test_network_shapes_and_finiteness(tiny_model):
    env = ChessEnv()
    logits, value = tiny_model.predict(env.observation())
    assert logits.shape == (NUM_ACTIONS,)
    assert np.isfinite(logits).all() and -1.0 <= value <= 1.0
    batch_logits, batch_v = tiny_model.predict(np.stack([env.observation()] * 3))
    assert batch_logits.shape == (3, NUM_ACTIONS) and batch_v.shape == (3,)


def test_training_changes_parameters_and_is_finite(tiny_model):
    env = ChessEnv()
    rec = make_record(env.board, {env.legal_actions()[0]: 1.0}, v=1.0, v_weight=1.0)
    batch = collate([rec] * 4)
    before = [w.numpy().copy() for w in tiny_model.trainable_variables]
    out = tiny_model.train(batch)
    assert np.isfinite(out["loss"]) and np.isfinite(out["policy_loss"])
    after = [w.numpy() for w in tiny_model.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(before, after)), "parameters must change"


def test_value_loss_is_masked_for_unknown_outcomes(tiny_model):
    env = ChessEnv()
    rec = make_record(env.board, {env.legal_actions()[0]: 1.0}, v=0.0, v_weight=0.0)
    out = tiny_model.train(collate([rec] * 4))
    assert out["value_loss"] == 0.0


def test_checkpoint_roundtrip_reproduces_predictions(tiny_model):
    env = ChessEnv()
    obs = env.observation()
    l0, v0 = tiny_model.predict(obs)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt")
        tiny_model.save_weights(path, extra={"note": "test"})
        restored = NeuralNetwork.from_checkpoint(path)
        l1, v1 = restored.predict(obs)
    assert np.allclose(l0, l1, atol=1e-5) and abs(v0 - v1) < 1e-5


# --------------------------------------------------------------------------- #
# MCTS
# --------------------------------------------------------------------------- #
def test_mcts_finds_mate_in_one_despite_bad_network(tiny_model):
    """Terminal facts must override a misleading network prediction."""

    class Misleading:
        """Uniform policy, and a value head that insists the position is lost."""

        def predict(self, obs):
            return np.zeros(NUM_ACTIONS, dtype=np.float32), -0.9

    env = ChessEnv(fen=MATE_IN_1)
    mcts = MCTS(Misleading(), c_puct=1.5)
    root, stats = mcts.run(env, 200)
    action = MCTS.select_move(root, temperature=0.0)
    assert env.decode(action).uci() == "a1a8"
    assert stats.simulations == 200
    assert stats.terminal_expansions > 0
    assert stats.env_transitions > 0


def test_mcts_stats_are_real(tiny_model):
    env = ChessEnv()
    mcts = MCTS(tiny_model)
    _, stats = mcts.run(env, 30)
    assert stats.simulations == 30
    assert stats.nn_evals >= 1
    assert stats.nn_evals + stats.cache_hits >= 1
    assert stats.elapsed_s > 0


def test_mcts_does_not_mutate_the_caller_env(tiny_model):
    env = ChessEnv()
    fen_before = env.fen()
    MCTS(tiny_model).run(env, 20)
    assert env.fen() == fen_before


def test_clone_tree_is_deep(tiny_model):
    env = ChessEnv()
    root, _ = MCTS(tiny_model).run(env, 20)
    copy = clone_tree(root)
    a = next(iter(copy.children))
    copy.children[a].visit_count += 1000
    assert root.children[a].visit_count != copy.children[a].visit_count


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_descriptor_shape_and_determinism():
    board = chess.Board(MATE_IN_1)
    moves = [chess.Move.from_uci("a1a8")]
    d1 = continuation_descriptor(board, moves)
    d2 = continuation_descriptor(board, moves)
    assert d1.shape == (len(FEATURE_NAMES),)
    assert np.allclose(d1, d2)
    assert np.isfinite(d1).all()


def test_descriptor_flags_checks_and_captures():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    d = continuation_descriptor(board, [chess.Move.from_uci("a1a8")])
    i = FEATURE_NAMES.index("gives_check_immediate")
    assert d[i] == 1.0
    board2 = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    d2 = continuation_descriptor(board2, [chess.Move.from_uci("e4d5")])
    assert d2[FEATURE_NAMES.index("is_capture_immediate")] == 1.0
    assert d2[FEATURE_NAMES.index("d_material_pawn")] == 1.0


# --------------------------------------------------------------------------- #
# conditions / budget accounting / isolation
# --------------------------------------------------------------------------- #
def test_all_conditions_run_and_account_budget(tiny_model):
    env = ChessEnv(fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 4 3")
    cfg = AllocationConfig(simulations=80, screen_frac=0.25, num_candidates=3)
    seen = {}
    for cond in ["standard", "quality", "random", "feature"]:
        res = run_condition(cond, tiny_model, env, cfg, seed=1)
        assert res.action in env.legal_actions(), "selected move must be legal"
        assert res.stats.simulations >= cfg.simulations * 0.9
        assert res.stats.elapsed_s > 0 and res.stats.nn_evals > 0
        seen[cond] = res
    # conditions 2-4 use the same nominal budget as each other
    sims = [seen[c].stats.simulations for c in ["quality", "random", "feature"]]
    assert max(sims) - min(sims) <= 1
    for c in ["quality", "random", "feature"]:
        assert len(seen[c].candidates) == 3


def test_screen_cache_is_copied_not_shared(tiny_model):
    from agents.alphazero.allocators import run_screening

    env = ChessEnv(fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 4 3")
    cfg = AllocationConfig(simulations=60, screen_frac=0.3, num_candidates=3)
    cache = run_screening(tiny_model, env, cfg)
    visits_before = {a: c.visit_count for a, c in cache[0].children.items()}
    run_condition("quality", tiny_model, env, cfg, seed=0, screen_cache=cache)
    visits_after = {a: c.visit_count for a, c in cache[0].children.items()}
    assert visits_before == visits_after, "a condition must not mutate the shared screen"


def test_screening_cost_charged_to_each_condition(tiny_model):
    from agents.alphazero.allocators import run_screening

    env = ChessEnv(fen=MATE_IN_1)
    cfg = AllocationConfig(simulations=60, screen_frac=0.5, num_candidates=2)
    cache = run_screening(tiny_model, env, cfg)
    res = run_condition("quality", tiny_model, env, cfg, seed=0, screen_cache=cache)
    assert res.stats.simulations >= cfg.simulations - 1
    assert res.stats.elapsed_s >= cache[2]  # screening wall time included


def test_conditions_are_deterministic_given_seed(tiny_model):
    env = ChessEnv(fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 4 3")
    cfg = AllocationConfig(simulations=60, screen_frac=0.3, num_candidates=3)
    a = run_condition("feature", tiny_model, env, cfg, seed=7).action
    b = run_condition("feature", tiny_model, env, cfg, seed=7).action
    assert a == b


# --------------------------------------------------------------------------- #
# dataset / oracle boundary
# --------------------------------------------------------------------------- #
def test_agent_facing_positions_exclude_solutions(tmp_path):
    import json

    from data.generate import Task, write_split_files

    t = Task(
        task_id="t0", family="f0", source="endgame", fen=MATE_IN_1, history_uci=[],
        reachable_from_start=False, mate_in=1, horizon_plies=1,
        solution_moves=["a1a8"], num_legal_moves=20, solver_status="ok",
        solver_nodes=10, position_hash="abc",
    )
    paths = write_split_files(str(tmp_path), {"test": [t]})
    raw = open(paths["test"]["positions"]).read()
    assert "a1a8" not in raw and "solution" not in raw
    row = json.loads(raw.strip())
    assert set(row) == {
        "task_id", "family", "source", "fen", "history_uci",
        "horizon_plies", "mate_in", "num_legal_moves", "reachable_from_start",
    }


def test_split_is_by_family():
    from data.generate import Task, split_by_family

    tasks = [
        Task(f"t{i}", f"fam{i//3}", "endgame", MATE_IN_1, [], False, 1, 1, ["a1a8"], 20, "ok", 1, f"h{i}")
        for i in range(30)
    ]
    splits, fam_split = split_by_family(tasks, seed=0)
    for name, ts in splits.items():
        for t in ts:
            assert fam_split[t.family] == name
    fam_to_split = {}
    for name, ts in splits.items():
        for t in ts:
            fam_to_split.setdefault(t.family, name)
            assert fam_to_split[t.family] == name


def test_bootstrap_records_use_verified_solutions_only(tiny_model):
    from agents.alphazero.agent import AlphaZeroAgent

    agent = AlphaZeroAgent(model=tiny_model)
    boot = SupervisedBootstrap(agent)
    recs = boot.build_records([{"fen": MATE_IN_1, "solution_moves": ["a1a8"]}])
    assert len(recs) == 1
    r = recs[0]
    board = chess.Board(MATE_IN_1)
    a = encode_action(chess.Move.from_uci("a1a8"), board.turn)
    assert r["pi"][a] > 0.9 and r["v"] == 1.0 and r["v_weight"] == 1.0
