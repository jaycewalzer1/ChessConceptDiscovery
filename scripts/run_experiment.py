"""Single documented entry point for the chess concept-discovery experiment.

    python -m scripts.run_experiment --profile laptop --max-minutes 120 --seed 42

Stages, in order:

  1. hardware      hardware / accelerator detection, thread and memory caps
  2. dependencies  import + version checks for every runtime dependency
  3. data          generate exactly-verified forced-mate tasks, split BY FAMILY
  4. model         supervised tactical bootstrap, then AlphaZero-style self-play
  5. checks        correctness test suite (pytest) run against the live code
  6. calibration   development-set grid over the diversity weight
  7. protocol      freeze the machine-readable protocol and hash it
  8. evaluation    frozen held-out test evaluation -> results.jsonl
  9. analysis      family-clustered statistics, verdict
 10. report        figures, case studies, HTML + Markdown reports

The run is resumable (``--resume``): completed stages and completed
``results.jsonl`` rows are detected and skipped.  When the wall-time budget is
reached the run stops cleanly, records which stages did not finish, and still
produces an honest partial report.

No chess engine, engine API, or language model is used anywhere.  ``python-chess``
supplies the rules only; all move quality judgements come from the learned
network, the search, and the self-implemented exact forced-mate verifier.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #
PROFILES = {
    "smoke": {
        "description": "fast end-to-end smoke test; not a research result",
        "data_target": 60,
        "data_minutes": 0.6,
        "arch": {"channels": 32, "blocks": 2},
        "bootstrap_epochs": 2,
        "bootstrap_minutes": 1.0,
        "selfplay_iterations": 1,
        "selfplay_minutes": 0.6,
        "selfplay_sims": 24,
        "selfplay_games": 2,
        "budgets": [96],
        "primary_budget": 96,
        "diversity_grid": [0.0, 0.75],
        "random_seeds": 2,
        "dev_cap": 12,
        "test_cap": 24,
        "calibration_minutes": 1.5,
        "eval_minutes": 4.0,
        "n_boot": 400,
        "run_tests": True,
    },
    "laptop": {
        "description": "resumable comparative experiment for a laptop CPU",
        # Budgets were chosen from a pre-registration probe on the SMOKE model:
        # mate-in-2 solve rate ran 4% -> 46% across 128 -> 3072 simulations, so
        # this ladder sits inside the sensitive region rather than at a floor
        # or a ceiling.  Mate-in-1 saturates by ~384 and acts as a sanity check.
        # A first laptop run showed the bootstrap converging in seconds on a few
        # hundred positions, i.e. the wall-time budget was being spent on epochs
        # over a tiny set rather than on data.  Generation runs at ~13 verified
        # tasks/s, so the budget is now weighted toward a much larger corpus.
        "data_target": 6000,
        "data_minutes": 10.0,
        "arch": {"channels": 64, "blocks": 4},
        "bootstrap_epochs": 60,
        "bootstrap_minutes": 12.0,
        "selfplay_iterations": 40,
        "selfplay_minutes": 10.0,
        "selfplay_sims": 48,
        "selfplay_games": 6,
        "budgets": [256, 1024, 3072],
        "primary_budget": 1024,
        "diversity_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "random_seeds": 2,
        "dev_cap": 100,
        "test_cap": 200,
        "calibration_minutes": 12.0,
        "eval_minutes": 58.0,
        "n_boot": 5000,
        "run_tests": True,
    },
}

STAGES = ["hardware", "dependencies", "data", "model", "checks",
          "calibration", "protocol", "evaluation", "analysis", "report"]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
class Clock:
    """Wall-time budget keeper shared by every stage."""

    def __init__(self, max_minutes: float):
        self.t0 = time.perf_counter()
        self.limit = max_minutes * 60.0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @property
    def remaining(self) -> float:
        return self.limit - self.elapsed

    def expired(self, reserve: float = 0.0) -> bool:
        return self.remaining <= reserve

    def budget(self, minutes: float, reserve: float) -> float:
        """Grant ``minutes`` unless that would eat into the closing reserve."""
        return max(0.0, min(minutes * 60.0, self.remaining - reserve))


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def _read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fh = open(path, "a")

    def __call__(self, *parts):
        msg = " ".join(str(p) for p in parts)
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# --------------------------------------------------------------------------- #
# stage 1: hardware
# --------------------------------------------------------------------------- #
def detect_hardware(args) -> dict:
    logical = os.cpu_count() or 1
    thread_cap = args.threads if args.threads else max(1, logical // 2)

    total_ram = None
    try:
        total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1 << 30)
    except (ValueError, OSError, AttributeError):
        pass

    cpu_brand = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu_brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or cpu_brand
        except Exception:
            pass

    mem_cap = args.memory_gib if args.memory_gib else min(8.0, (total_ram or 16.0) / 2)

    # Thread and memory caps must be set BEFORE TensorFlow is imported.
    os.environ.setdefault("OMP_NUM_THREADS", str(thread_cap))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(thread_cap))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(max(1, thread_cap // 2)))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_brand": cpu_brand,
        "logical_cores": logical,
        "thread_cap": thread_cap,
        "total_ram_gib": float(total_ram) if total_ram else float("nan"),
        "memory_cap_gib": float(mem_cap),
        "device_request": args.device,
        "python": platform.python_version(),
        "detected_at": _now(),
    }


# --------------------------------------------------------------------------- #
# stage 2: dependencies
# --------------------------------------------------------------------------- #
def check_dependencies(hardware: dict, log) -> dict:
    import numpy as np
    import chess
    import matplotlib
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(hardware["thread_cap"])
    tf.config.threading.set_inter_op_parallelism_threads(max(1, hardware["thread_cap"] // 2))

    gpus = tf.config.list_physical_devices("GPU")
    accelerator = "CPU only (no GPU/Metal device exposed to TensorFlow)"
    if gpus and hardware["device_request"] != "cpu":
        accelerator = f"{len(gpus)} GPU device(s): {[g.name for g in gpus]}"

    hardware.update({
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "python_chess": chess.__version__,
        "matplotlib": matplotlib.__version__,
        "accelerator": accelerator,
        "tf_visible_devices": [d.name for d in tf.config.list_physical_devices()],
    })
    log(f"[deps] python {hardware['python']} | tensorflow {tf.__version__} | "
        f"numpy {np.__version__} | python-chess {chess.__version__}")
    log(f"[deps] accelerator: {accelerator}")
    log(f"[deps] threads capped at {hardware['thread_cap']} of {hardware['logical_cores']}")
    return hardware


# --------------------------------------------------------------------------- #
# stage 3: data
# --------------------------------------------------------------------------- #
def prepare_data(args, cfg, out_dir, clock, log, reserve):
    from data.generate import (generate_dataset, split_by_family, write_manifest,
                               write_split_files, load_split)

    data_dir = args.data_path or os.path.join(out_dir, "data")
    paths_file = os.path.join(data_dir, "paths.json")

    if args.resume and os.path.exists(paths_file):
        paths = _read_json(paths_file)
        if all(os.path.exists(p["positions"]) for p in paths.values()):
            log(f"[data] reusing existing dataset at {data_dir}")
            manifest = _read_json(os.path.join(out_dir, "data_manifest.json"))
            return paths, manifest, {"reused": True}

    budget = clock.budget(cfg["data_minutes"], reserve)
    tasks, gen_info = generate_dataset(
        seed=args.seed, target=cfg["data_target"], time_budget_s=budget, log=log
    )
    if not tasks:
        raise RuntimeError("dataset generation produced no verified tasks")

    splits, fam_split = split_by_family(tasks, seed=args.seed)
    paths = write_split_files(data_dir, splits)
    _write_json(paths_file, paths)
    manifest = write_manifest(
        os.path.join(out_dir, "data_manifest.json"), splits, gen_info, fam_split,
        extra={"data_dir": data_dir, "written_at": _now()},
    )
    for name in ("train", "dev", "test"):
        s = manifest["splits"][name]
        log(f"[data] {name}: {s['n']} positions / {s['families']} families, "
            f"mate-in {s['mate_in_counts']}")
    return paths, manifest, gen_info


# --------------------------------------------------------------------------- #
# stage 4: model
# --------------------------------------------------------------------------- #
def prepare_model(args, cfg, out_dir, paths, clock, log, reserve):
    import numpy as np
    from agents.alphazero.agent import AlphaZeroAgent
    from agents.alphazero.neural_network import NeuralNetwork
    from agents.alphazero.training import SupervisedBootstrap, TrainingLoop
    from data.generate import load_split
    from envs.chess_env import ChessEnv

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    ckpt_path = args.model_path or os.path.join(ckpt_dir, "frozen")
    meta_path = os.path.join(out_dir, "checkpoint_metadata.json")

    if args.resume and os.path.exists(ckpt_path + ".pkl") and os.path.exists(meta_path):
        log(f"[model] reusing frozen checkpoint {ckpt_path}")
        agent = AlphaZeroAgent(model=NeuralNetwork.from_checkpoint(ckpt_path))
        meta = _read_json(meta_path)
        return agent, meta, _read_json(os.path.join(out_dir, "training_summary.json"))

    train_tasks = load_split(paths, "train", with_labels=True)
    rng = np.random.default_rng(args.seed)
    agent = AlphaZeroAgent(simulations=cfg["primary_budget"], arch=cfg["arch"])

    # ---- stage 4a: supervised tactical bootstrap on VERIFIED solutions -----
    boot = SupervisedBootstrap(agent, rng=rng)
    records = boot.build_records(train_tasks)
    log(f"[model] bootstrap on {len(records)} verified training positions "
        f"({agent.model.parameter_count():,} parameters)")
    before = float(agent.model.weights[0].numpy().sum())
    boot_summary = boot.run(
        records, epochs=cfg["bootstrap_epochs"], batch_size=64, lr=1e-3,
        time_budget_s=clock.budget(cfg["bootstrap_minutes"], reserve), log=log,
    )
    after = float(agent.model.weights[0].numpy().sum())
    boot_summary["weights_changed"] = bool(abs(after - before) > 0)
    boot_summary["n_records"] = len(records)

    # ---- stage 4b: AlphaZero-style self-play fine-tuning -------------------
    sp_budget = clock.budget(cfg["selfplay_minutes"], reserve)
    if sp_budget < 10:
        sp_summary = {"iterations": [], "games": 0, "records": 0,
                      "note": "skipped: insufficient remaining wall-time budget"}
        log("[model] self-play skipped (no remaining budget)")
    else:
        loop = TrainingLoop(
            agent, ChessEnv(), num_iterations=cfg["selfplay_iterations"],
            checkpoint_dir=ckpt_dir, simulations=cfg["selfplay_sims"],
            games_per_iter=cfg["selfplay_games"], max_game_plies=24,
            batch_size=64, train_steps_per_iter=16, rng=rng, log=log,
        )
        start_fens = [t["fen"] for t in train_tasks]
        sp_summary = loop.run(start_fens=start_fens, time_budget_s=sp_budget)

    training_summary = {
        "procedure": ("supervised tactical bootstrapping on exactly-verified forced-mate "
                      "solutions, followed by an AlphaZero-style self-play/MCTS fine-tuning "
                      "stage; this is NOT tabula-rasa AlphaZero training"),
        "bootstrap": boot_summary,
        "selfplay": sp_summary,
    }
    _write_json(os.path.join(out_dir, "training_summary.json"), training_summary)

    # ---- freeze -----------------------------------------------------------
    agent.save_checkpoint(ckpt_path, extra={"run_seed": args.seed, "profile": args.profile})
    probe = _probe_predictions(agent.model, train_tasks)
    reloaded = NeuralNetwork.from_checkpoint(ckpt_path)
    probe2 = _probe_predictions(reloaded, train_tasks)
    reproduces = bool(np.allclose(probe, probe2, atol=1e-6))
    log(f"[model] checkpoint reload reproduces predictions: {reproduces}")

    meta = {
        "path": ckpt_path,
        "weights_sha256": _sha256_file(ckpt_path + ".pkl"),
        "meta_sha256": _sha256_file(ckpt_path + ".meta.json"),
        "parameter_count": agent.model.parameter_count(),
        "arch": agent.model.arch,
        "procedure": training_summary["procedure"],
        "bootstrap_steps": boot_summary.get("steps", 0),
        "bootstrap_records": len(records),
        "selfplay_note": (f"{sp_summary.get('games', 0)} self-play games, "
                          f"{sp_summary.get('records', 0)} records"
                          if sp_summary.get("games") else
                          sp_summary.get("note", "not run")),
        "reload_reproduces_predictions": reproduces,
        "frozen_at": _now(),
    }
    _write_json(meta_path, meta)
    return agent, meta, training_summary


def _probe_predictions(model, tasks, n=4):
    import chess
    import numpy as np
    from envs.chess_utils import encode_board

    obs = [encode_board(chess.Board(t["fen"])) for t in tasks[:n]]
    if not obs:
        return np.zeros(1)
    logits, value = model.predict(np.stack(obs))
    return np.concatenate([logits.ravel()[:64], np.atleast_1d(value).ravel()])


# --------------------------------------------------------------------------- #
# stage 5: correctness checks
# --------------------------------------------------------------------------- #
def run_correctness_checks(out_dir, log, enabled=True):
    if not enabled:
        return {"ran": False, "note": "disabled by --skip-tests"}
    log("[checks] running tests/test_core.py")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_core.py", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    with open(os.path.join(out_dir, "logs", "pytest.txt"), "w") as fh:
        fh.write(proc.stdout + "\n" + proc.stderr)
    log(f"[checks] {tail} (exit {proc.returncode})")
    return {"ran": True, "passed": proc.returncode == 0, "summary": tail,
            "elapsed_s": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# evaluation primitives shared by calibration and the frozen test run
# --------------------------------------------------------------------------- #
def raw_policy_facts(agent, env, solutions):
    """Raw network policy facts for one position, shared by every condition."""
    probs, value = agent.raw_policy(env)
    order = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = {a: i + 1 for i, (a, _) in enumerate(order)}
    sol_actions = []
    for uci in solutions:
        try:
            mv = env.board.parse_uci(uci)
        except ValueError:
            continue
        if mv in env.board.legal_moves:
            sol_actions.append(env.encode(mv))
    sol_ranks = [rank[a] for a in sol_actions if a in rank]
    best_rank = min(sol_ranks) if sol_ranks else None
    best_prob = max((probs[a] for a in sol_actions if a in probs), default=None)
    return {
        "probs": probs,
        "rank": rank,
        "value": value,
        "best_solution_policy_rank": best_rank,
        "best_solution_policy_prob": best_prob,
        # Eligibility uses the COMMON raw policy, so the denominator is
        # identical for every method (frozen in the protocol).
        "overlooked_eligible": bool(best_rank is not None and best_rank > 3),
    }


def evaluate_position(agent, task, conditions, budget, cfg_kwargs, scaler,
                      seeds, run_id, share_screen=True):
    """Run every condition on one position.  Returns a list of result rows."""
    from agents.alphazero.allocators import AllocationConfig, run_condition, run_screening
    from envs.chess_env import ChessEnv

    solutions = task["solution_moves"]
    base_env = ChessEnv(fen=task["fen"])
    facts = raw_policy_facts(agent, base_env, solutions)

    cfg = AllocationConfig(simulations=budget, **cfg_kwargs)
    screen = None
    if share_screen:
        # One physical screening search is reused, but its FULL logical cost is
        # charged to every condition that needs it (see allocators docstring).
        screen = run_screening(agent.model, ChessEnv(fen=task["fen"]), cfg)

    rows = []
    for cond in conditions:
        cond_seeds = seeds if cond == "random" else [0]
        for sd in cond_seeds:
            env = ChessEnv(fen=task["fen"])
            res = run_condition(cond, agent.model, env, cfg, scaler=scaler,
                                seed=sd, screen_cache=None if cond == "standard" else screen)
            legal = True
            try:
                mv = base_env.board.parse_uci(res.move_uci)
                legal = mv in base_env.board.legal_moves
                san = base_env.board.san(mv) if legal else "(illegal)"
                action = res.action
            except ValueError:
                san, action = "(illegal)", res.action
                legal = False
            rows.append({
                "run_id": run_id,
                "result_id": f"{task['task_id']}|{cond}|{budget}|{sd}",
                "task_id": task["task_id"],
                "family": task["family"],
                "source": task["source"],
                "fen": task["fen"],
                "mate_in": task["mate_in"],
                "horizon_plies": task["horizon_plies"],
                "num_legal_moves": task["num_legal_moves"],
                "condition": cond,
                "budget": budget,
                "seed": sd,
                "chosen_move_uci": res.move_uci,
                "chosen_move_san": san,
                "move_was_legal": legal,
                "correct": bool(res.move_uci in solutions),
                "solution_moves": solutions,
                "solver_status": task["solver_status"],
                "policy_rank_of_chosen": facts["rank"].get(action),
                "policy_prob_of_chosen": facts["probs"].get(action),
                "best_solution_policy_rank": facts["best_solution_policy_rank"],
                "best_solution_policy_prob": facts["best_solution_policy_prob"],
                "overlooked_eligible": facts["overlooked_eligible"],
                "raw_value": facts["value"],
                "stats": res.stats.as_dict(),
                "candidates": res.candidates,
                "feature_mode": res.feature_mode,
                "notes": res.notes,
                "search_pv_san": res.pv_san,
            })
    return rows


# --------------------------------------------------------------------------- #
# stage 6: development calibration
# --------------------------------------------------------------------------- #
def calibrate(agent, cfg, paths, out_dir, args, clock, log, reserve):
    """Fit the feature scaler and choose the diversity weight on DEV data only."""
    import numpy as np
    from agents.alphazero.allocators import AllocationConfig, run_screening, eligible_pool, most_visited_continuation
    from agents.alphazero.features import FeatureScaler, continuation_descriptor
    from data.generate import load_split
    from envs.chess_env import ChessEnv

    dev_file = os.path.join(out_dir, "dev_summary.json")
    scaler_file = os.path.join(out_dir, "feature_scaler.json")
    if args.resume and os.path.exists(dev_file) and os.path.exists(scaler_file):
        log("[calib] reusing existing development calibration")
        return _read_json(dev_file), FeatureScaler.from_dict(_read_json(scaler_file))

    train = load_split(paths, "train", with_labels=True)
    dev = load_split(paths, "dev", with_labels=True)[: cfg["dev_cap"]]
    budget = cfg["primary_budget"]
    base_cfg = dict(screen_frac=0.25, num_candidates=4, pool_cap=48,
                    continuation_len=4, c_puct=1.5, max_plies=40)

    # ---- fit the standardiser on TRAIN + DEV descriptors only -------------
    acfg = AllocationConfig(simulations=budget, **base_cfg)
    descs = []
    for t in (train + dev)[: min(len(train) + len(dev), 120)]:
        env = ChessEnv(fen=t["fen"])
        root, _, _ = run_screening(agent.model, env, acfg)
        for a in eligible_pool(env, root, acfg):
            moves, mode = most_visited_continuation(env, root, a, acfg.continuation_len)
            descs.append(continuation_descriptor(env.board, moves, mode=mode))
        if clock.expired(reserve):
            break
    scaler = FeatureScaler().fit(np.asarray(descs)) if descs else FeatureScaler()
    _write_json(scaler_file, scaler.to_dict())
    log(f"[calib] feature scaler fitted on {len(descs)} TRAIN/DEV descriptors")

    # ---- grid over the diversity weight on DEV ----------------------------
    deadline = time.perf_counter() + clock.budget(cfg["calibration_minutes"], reserve)
    grid, std_hits, std_n = [], 0, 0
    for lam in cfg["diversity_grid"]:
        hits = n = 0
        kw = dict(base_cfg, diversity_weight=lam)
        for t in dev:
            if time.perf_counter() > deadline or clock.expired(reserve):
                break
            conds = ["feature", "standard"] if lam == cfg["diversity_grid"][0] else ["feature"]
            rows = evaluate_position(agent, t, conds, budget, kw, scaler,
                                     [args.seed], "calibration")
            for r in rows:
                if r["condition"] == "feature":
                    hits += int(r["correct"])
                    n += 1
                else:
                    std_hits += int(r["correct"])
                    std_n += 1
        grid.append({"diversity_weight": lam, "n": n,
                     "success_rate": hits / n if n else float("nan"), "hits": hits})
        log(f"[calib] lambda={lam}: {hits}/{n} dev positions solved")
        if time.perf_counter() > deadline or clock.expired(reserve):
            log("[calib] calibration budget reached, stopping grid early")
            break

    scored = [g for g in grid if g["n"] > 0]
    chosen = max(scored, key=lambda g: (g["success_rate"], -g["diversity_weight"]))["diversity_weight"] \
        if scored else 0.5
    dev_summary = {
        "n_dev": len(dev),
        "budget": budget,
        "diversity_grid": grid,
        "chosen_diversity_weight": chosen,
        "standard_success_rate": std_hits / std_n if std_n else None,
        "standard_n": std_n,
        "selection_rule": "highest DEV success rate; ties broken toward the smaller lambda",
        "scaler_descriptors": len(descs),
        "base_config": base_cfg,
    }
    _write_json(dev_file, dev_summary)
    log(f"[calib] frozen diversity weight lambda = {chosen}")
    return dev_summary, scaler


# --------------------------------------------------------------------------- #
# stage 7: protocol freeze
# --------------------------------------------------------------------------- #
def freeze_protocol(args, cfg, out_dir, dev_summary, manifest, ckpt_meta, scaler):
    from agents.alphazero.allocators import CONDITIONS
    from agents.alphazero.features import FEATURE_NAMES

    alloc = dict(dev_summary["base_config"],
                 diversity_weight=dev_summary["chosen_diversity_weight"])
    protocol = {
        "research_question": (
            "At a fixed search budget, does explicitly exploring candidate moves with "
            "different chess features help a small AlphaZero-style agent recover correct "
            "moves that its own initial policy ranks low?"),
        "creativity_definition": (
            "Operational and narrow: discovering a solution that was initially unlikely "
            "under the agent's own policy but is verified correct by an exact oracle. "
            "No claim of novelty to humanity, artistic merit, or general creative intelligence."),
        "primary_endpoint": ("fraction of held-out test tasks where the selected root move is a "
                             "member of the exactly-verified complete solution set"),
        "primary_comparison": "feature-diverse allocation vs standard PUCT, paired on positions",
        "verdict_rule": ("positive if the 95% family-clustered bootstrap CI of the paired "
                         "difference lies entirely above zero; negative if entirely below; "
                         "otherwise inconclusive"),
        "secondary_endpoints": [
            "success restricted to positions whose every verified solution falls outside the "
            "common raw-policy top three (same denominator for all methods)",
            "policy rank and probability of the selected correct move",
            "success as a function of search budget",
            "search time and network evaluations per solved task",
            "candidate coverage and feature dispersion (mechanism diagnostics only)",
        ],
        "conditions": CONDITIONS,
        "budgets": cfg["budgets"],
        "primary_budget": cfg["primary_budget"],
        "allocation": alloc,
        "random_seeds": list(range(cfg["random_seeds"])),
        "seed": args.seed,
        "max_minutes": args.max_minutes,
        "profile": args.profile,
        "feature_definitions": FEATURE_NAMES,
        "feature_normalisation": {
            "fitted_on": "TRAIN + DEV descriptors only",
            "stats": scaler.to_dict(),
        },
        "budget_accounting": (
            "Every condition is charged the full logical cost of the screening search it "
            "requires, even when one physical screen is reused; feature-computation wall "
            "time is added to the feature-diverse condition. No condition inherits another's "
            "search tree or any oracle result."),
        "inclusion_criteria": "solver_status == 'ok' and a legal selected move",
        "exclusion_criteria": "tasks whose exact solution set could not be completely enumerated",
        "statistics": {
            "resampling_unit": "source family (independent playout game or endgame seed)",
            "bootstrap_replicates": cfg["n_boot"],
            "interval": "95% percentile cluster bootstrap",
            "seed_aggregation": "repeated seeds averaged within a position before comparison",
        },
        "example_selection_rule": (
            "feature_win -> standard_win -> both_solved -> both_failed -> vs_random, "
            "up to 2 per category and 10 total, task-id order within a category"),
        "dataset": {
            "split_unit": manifest["leakage_controls"]["split_unit"],
            "test_split_hash": manifest["splits"]["test"]["split_hash"],
            "dev_split_hash": manifest["splits"]["dev"]["split_hash"],
            "train_split_hash": manifest["splits"]["train"]["split_hash"],
            "n_test": manifest["splits"]["test"]["n"],
        },
        "checkpoint": {
            "weights_sha256": ckpt_meta["weights_sha256"],
            "parameter_count": ckpt_meta["parameter_count"],
            "frozen": "the identical checkpoint is used for every condition",
        },
        "development_calibration": dev_summary,
        "no_test_tuning": ("all hyper-parameters were fixed from development data before "
                           "any test score was computed; no seed selection was performed"),
        "frozen_at": _now(),
    }
    payload = json.dumps(protocol, sort_keys=True, default=str).encode()
    protocol["protocol_sha256"] = hashlib.sha256(payload).hexdigest()
    _write_json(os.path.join(out_dir, "protocol.json"), protocol)
    return protocol


# --------------------------------------------------------------------------- #
# stage 8: frozen test evaluation
# --------------------------------------------------------------------------- #
def evaluate_test(agent, protocol, cfg, paths, out_dir, args, scaler, clock, log, reserve):
    from data.generate import load_split

    results_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if args.resume and os.path.exists(results_path):
        with open(results_path) as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["result_id"])
        log(f"[eval] resuming with {len(done)} rows already complete")

    test = load_split(paths, "test", with_labels=True)
    test = [t for t in test if t["solver_status"] == "ok"][: cfg["test_cap"]]
    budgets = protocol["budgets"]
    conds = protocol["conditions"]
    seeds = protocol["random_seeds"]
    alloc_kwargs = {k: v for k, v in protocol["allocation"].items()}

    deadline = time.perf_counter() + clock.budget(cfg["eval_minutes"], reserve)
    written = 0
    unfinished = []
    with open(results_path, "a") as fh:
        for bi, budget in enumerate(budgets):
            for ti, task in enumerate(test):
                key = f"{task['task_id']}|standard|{budget}|0"
                if key in done:
                    continue
                if time.perf_counter() > deadline or clock.expired(reserve):
                    unfinished.append(f"budget {budget}: stopped after {ti}/{len(test)} positions")
                    log(f"[eval] wall-time limit reached at budget {budget}, "
                        f"position {ti}/{len(test)}")
                    return results_path, {"written": written, "unfinished": unfinished,
                                          "n_test_attempted": len(test)}
                rows = evaluate_position(agent, task, conds, budget, alloc_kwargs,
                                         scaler, seeds, args.run_id)
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
                    written += 1
                fh.flush()
                if (ti + 1) % 10 == 0:
                    log(f"[eval] budget {budget}: {ti+1}/{len(test)} positions, "
                        f"{written} rows, {clock.remaining/60:.1f} min left")
            log(f"[eval] budget {budget} complete")
    return results_path, {"written": written, "unfinished": unfinished,
                          "n_test_attempted": len(test)}


# --------------------------------------------------------------------------- #
# stages 9-10: analysis and reporting
# --------------------------------------------------------------------------- #
def analyse_and_report(args, cfg, out_dir, results_path, protocol, hardware,
                       manifest, training_summary, ckpt_meta, dev_summary,
                       stages, clock, log):
    from analysis import case_studies, plots, report, stats

    rows = stats.load_results(results_path) if os.path.exists(results_path) else []
    if not rows:
        log("[analysis] no result rows were produced; writing an infrastructure-only report")
        summary = {"budgets": [], "conditions": protocol["conditions"],
                   "primary_budget": protocol["primary_budget"], "n_test_positions": 0,
                   "n_families": 0, "overlooked_denominator": 0, "by_budget": {},
                   "paired": {}, "overlooked": {}, "cost": {}, "seeds_per_condition": {},
                   "policy_diagnostics": {}, "exclusions": {}}
        v = ("infrastructure-only", "No test rows were produced within the wall-time budget.")
    else:
        summary, _agg, _overlooked = stats.build_summary(
            rows, protocol, n_boot=cfg["n_boot"], seed=args.seed)
        v = stats.verdict(summary)
    summary["verdict"] = {"kind": v[0], "text": v[1]}
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    log(f"[analysis] verdict: {v[0]} - {v[1]}")

    fig_dir = os.path.join(out_dir, "figures")
    overhead = {
        "data_s": stages.get("data", {}).get("elapsed_s", 0.0),
        "training_s": stages.get("model", {}).get("elapsed_s", 0.0),
        "calibration_s": stages.get("calibration", {}).get("elapsed_s", 0.0),
    }
    figures = plots.generate_all(rows, summary, fig_dir, training_summary,
                                 dev_summary, overhead)
    log(f"[report] wrote {len(figures)} figures to {fig_dir}")

    cases, index = case_studies.generate(rows, summary, out_dir, args.run_id)
    log(f"[report] built {len(cases)} case studies")

    ctx = {
        "summary": summary,
        "figures": figures,
        "hardware": hardware,
        "protocol": protocol,
        "verdict": v,
        "cases": cases,
        "case_index": index,
        "data_manifest": manifest,
        "training_summary": training_summary,
        "checkpoint_metadata": ckpt_meta,
        "stages": stages,
        "run_id": args.run_id,
        "generated_at": _now(),
        "runtime_s": clock.elapsed,
        "command": args.command,
        "literature_note": LITERATURE_NOTE,
    }
    for name, builder in (("report.html", report.build_html),
                          ("case_studies.html", report.build_case_studies_html),
                          ("report.md", report.build_markdown)):
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write(builder(ctx))
        log(f"[report] wrote {os.path.join(out_dir, name)}")
    return summary, v


LITERATURE_NOTE = (
    "Related work was identified from the two references supplied with the task: "
    "Schut et al., 'Bridging the Human-AI Knowledge Gap' (arXiv:2310.16410), which extracts "
    "concepts an AlphaZero-style chess agent has learned; and Zahavy et al., 'Diversifying AI' "
    "(arXiv:2308.09175), which trains a population of diverse chess agents. No systematic "
    "literature search was performed in this session, and no claim beyond the titles and "
    "general subject matter of those two papers is asserted here."
)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m scripts.run_experiment",
        description="AlphaZero chess feature-diverse allocation experiment.",
    )
    p.add_argument("--profile", choices=sorted(PROFILES), default="laptop")
    p.add_argument("--max-minutes", type=float, default=120.0,
                   help="total wall-time budget for data, training and experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None, help="defaults to outputs/<run_id>")
    p.add_argument("--run-id", default=None)
    p.add_argument("--model-path", default=None, help="reuse an existing frozen checkpoint")
    p.add_argument("--data-path", default=None, help="reuse an existing dataset directory")
    p.add_argument("--device", choices=["cpu", "auto"], default="auto")
    p.add_argument("--threads", type=int, default=None, help="logical-core cap")
    p.add_argument("--memory-gib", type=float, default=None, help="soft memory target")
    p.add_argument("--resume", action="store_true", help="reuse completed stages and rows")
    p.add_argument("--skip-tests", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = PROFILES[args.profile]
    args.run_id = args.run_id or f"{args.profile}-{_dt.datetime.now():%Y%m%d-%H%M%S}"
    out_dir = args.output_dir or os.path.join(REPO_ROOT, "outputs", args.run_id)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    args.command = "python -m scripts.run_experiment " + " ".join(
        sys.argv[1:] if argv is None else argv)

    log = Logger(os.path.join(out_dir, "logs", "run.log"))
    clock = Clock(args.max_minutes)
    reserve = min(180.0, 0.12 * args.max_minutes * 60.0)  # closing reserve for reporting
    stages = {}
    state_path = os.path.join(out_dir, "state.json")

    def stage(name, fn, *a, **kw):
        t0 = time.perf_counter()
        log(f"=== stage {name} ===")
        try:
            out = fn(*a, **kw)
            note = "completed"
        except Exception as exc:
            stages[name] = {"status": "failed", "elapsed_s": time.perf_counter() - t0,
                            "note": repr(exc)}
            _write_json(state_path, {"stages": stages, "run_id": args.run_id})
            raise
        stages[name] = {"status": "completed", "elapsed_s": time.perf_counter() - t0,
                        "note": note}
        _write_json(state_path, {"stages": stages, "run_id": args.run_id})
        return out

    log(f"run_id={args.run_id} profile={args.profile} seed={args.seed} "
        f"budget={args.max_minutes} min")
    log(f"output directory: {out_dir}")

    hardware = stage("hardware", detect_hardware, args)
    hardware = stage("dependencies", check_dependencies, hardware, log)
    _write_json(os.path.join(out_dir, "hardware.json"), hardware)

    paths, manifest, _gen = stage("data", prepare_data, args, cfg, out_dir,
                                  clock, log, reserve)
    agent, ckpt_meta, training_summary = stage("model", prepare_model, args, cfg,
                                               out_dir, paths, clock, log, reserve)
    stages["checks"] = {"status": "completed", "elapsed_s": 0.0, "note": "pending"}
    checks = stage("checks", run_correctness_checks, out_dir, log, not args.skip_tests)
    stages["checks"]["note"] = checks.get("summary", checks.get("note", ""))

    dev_summary, scaler = stage("calibration", calibrate, agent, cfg, paths, out_dir,
                                args, clock, log, reserve)
    protocol = stage("protocol", freeze_protocol, args, cfg, out_dir, dev_summary,
                     manifest, ckpt_meta, scaler)
    log(f"[protocol] frozen, sha256={protocol['protocol_sha256'][:16]}...")

    results_path, eval_info = stage("evaluation", evaluate_test, agent, protocol, cfg,
                                    paths, out_dir, args, scaler, clock, log, reserve)
    stages["evaluation"]["note"] = (
        f"{eval_info['written']} rows over {eval_info['n_test_attempted']} test positions"
        + ("; UNFINISHED: " + "; ".join(eval_info["unfinished"])
           if eval_info["unfinished"] else ""))

    summary, v = stage("report", analyse_and_report, args, cfg, out_dir, results_path,
                       protocol, hardware, manifest, training_summary, ckpt_meta,
                       dev_summary, stages, clock, log)
    stages["analysis"] = {"status": "completed", "elapsed_s": 0.0,
                          "note": f"verdict: {v[0]}"}
    _write_json(state_path, {"stages": stages, "run_id": args.run_id,
                             "finished_at": _now(), "runtime_s": clock.elapsed})

    log(f"=== done in {clock.elapsed/60:.1f} min ===")
    log(f"report: {os.path.join(out_dir, 'report.html')}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
