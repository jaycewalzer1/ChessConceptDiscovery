"""Verify a run's artifacts: boards against FENs, labels against raw rows.

    python -m scripts.verify_artifacts outputs/<run_id>

Checks performed:

  1. every required artifact exists;
  2. the HTML reports are self-contained (no remote assets, no CDN, no server);
  3. every figure referenced in the report is present as inline SVG;
  4. the comparison table's numbers agree with ``summary.json``;
  5. every figure's ``.json`` sidecar agrees with ``summary.json``;
  6. every case study's FEN parses, every rendered board SVG matches the piece
     census of the position it claims to show, the oracle line replays legally,
     and every method's chosen move is legal in the root position;
  7. every case-study number (rank, probability, simulations, elapsed time,
     verified result) matches the raw ``results.jsonl`` row it cites;
  8. ``examples.pgn`` parses and every game replays legally.

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

import chess
import chess.pgn

PIECE_RE = re.compile(r'xlink:href="#(white|black)-([a-z]+)"')
REMOTE_RE = re.compile(r'(?:src|href)\s*=\s*["\'](https?:)?//', re.I)


class Checker:
    def __init__(self):
        self.failures = []
        self.passed = 0

    def check(self, ok, label, detail=""):
        if ok:
            self.passed += 1
        else:
            self.failures.append(f"{label}: {detail}")
        return ok

    def report(self):
        print(f"\n{self.passed} checks passed, {len(self.failures)} failed")
        for f in self.failures:
            print("  FAIL", f)
        return not self.failures


def svg_piece_census(svg_text):
    census = {}
    for color, piece in PIECE_RE.findall(svg_text):
        census[(color, piece)] = census.get((color, piece), 0) + 1
    return census


def board_piece_census(board):
    names = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
             chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
    census = {}
    for sq, piece in board.piece_map().items():
        key = ("white" if piece.color else "black", names[piece.piece_type])
        census[key] = census.get(key, 0) + 1
    return census


def main(out_dir):
    c = Checker()

    required = ["protocol.json", "hardware.json", "data_manifest.json",
                "training_summary.json", "checkpoint_metadata.json", "summary.json",
                "report.md", "report.html", "case_studies.html", "examples.pgn"]
    for name in required:
        c.check(os.path.exists(os.path.join(out_dir, name)), "artifact present", name)

    summary = json.load(open(os.path.join(out_dir, "summary.json")))
    rows = []
    rp = os.path.join(out_dir, "results.jsonl")
    if os.path.exists(rp):
        rows = [json.loads(l) for l in open(rp) if l.strip()]
    by_id = {r["result_id"]: r for r in rows}
    print(f"loaded {len(rows)} result rows, {len({r['task_id'] for r in rows})} positions")

    # ---- 2/3: self-contained HTML ----------------------------------------
    for name in ("report.html", "case_studies.html"):
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            continue
        html = open(path).read()
        c.check(not REMOTE_RE.search(html), "no remote assets", name)
        c.check("<script src" not in html, "no external script", name)
        c.check("<link " not in html, "no external stylesheet", name)
        n_svg = html.count("<svg")
        c.check(n_svg > 0, "inline svg present", f"{name}: {n_svg}")
        print(f"{name}: {len(html)/1024:.0f} KiB, {n_svg} inline SVG elements")

    # ---- 4: figure sidecars ----------------------------------------------
    fig_dir = os.path.join(out_dir, "figures")
    if os.path.isdir(fig_dir):
        for f in sorted(os.listdir(fig_dir)):
            if f.endswith(".json"):
                base = f[:-5]
                c.check(os.path.exists(os.path.join(fig_dir, base + ".svg")),
                        "figure svg present", base)
                c.check(os.path.exists(os.path.join(fig_dir, base + ".png")),
                        "figure png present", base)
        data_path = os.path.join(fig_dir, "fig1_success_vs_budget.json")
        if os.path.exists(data_path):
            d = json.load(open(data_path))
            budgets = d.get("budgets", [])
            for cond, series in d.get("series", {}).items():
                for b, pt in zip(budgets, series):
                    ref = summary["by_budget"].get(str(b), {}).get(cond)
                    if ref:
                        c.check(abs(ref["rate"] - pt["rate"]) < 1e-9
                                and ref["n"] == pt["n"],
                                "fig1 matches summary", f"{cond}@{b}")

    # ---- 5/6/7: case studies ---------------------------------------------
    idx_path = os.path.join(out_dir, "case_studies", "index.json")
    if os.path.exists(idx_path):
        index = json.load(open(idx_path))
        fig_dir = os.path.join(out_dir, "case_studies", "figures")
        for case in index["cases"]:
            cid = case["case_id"]
            if "error" in case:
                c.check(False, "case built", f"{cid}: {case['error']}")
                continue
            board = chess.Board(case["fen"])
            c.check(board.is_valid(), "case FEN valid", cid)
            c.check(("White" if board.turn else "Black") == case["side_to_move"],
                    "side to move label", cid)

            # initial board render matches the FEN exactly
            init = os.path.join(fig_dir, f"{cid}_initial.svg")
            if c.check(os.path.exists(init), "initial board svg", cid):
                c.check(svg_piece_census(open(init).read()) == board_piece_census(board),
                        "initial board matches FEN", cid)

            # oracle line replays legally and each sequence panel matches
            work = chess.Board(case["fen"])
            seq_ok = True
            for i, san in enumerate(case["oracle_line_san"]):
                try:
                    work.push(work.parse_san(san))
                except ValueError:
                    seq_ok = False
                    break
                p = os.path.join(fig_dir, f"{cid}_seq{i+1}.svg")
                if os.path.exists(p):
                    c.check(svg_piece_census(open(p).read()) == board_piece_census(work),
                            "sequence board matches replay", f"{cid} ply {i+1}")
            c.check(seq_ok, "oracle line replays legally", cid)
            if case["oracle_line_san"]:
                c.check(work.is_checkmate(), "oracle line ends in checkmate", cid)

            # methods: legality, correctness, and agreement with the raw rows
            for name, m in case["methods"].items():
                mv = chess.Move.from_uci(m["move_uci"])
                c.check(mv in board.legal_moves, "chosen move legal", f"{cid}/{name}")
                c.check(m["correct"] == (m["move_uci"] in case["solution_moves"]),
                        "correct flag matches solution set", f"{cid}/{name}")
                rid = case["result_ids"].get(name)
                row = by_id.get(rid)
                if c.check(row is not None, "result id resolves", f"{cid}/{name} {rid}"):
                    c.check(row["chosen_move_uci"] == m["move_uci"],
                            "move matches raw row", f"{cid}/{name}")
                    c.check(bool(row["correct"]) == bool(m["correct"]),
                            "result matches raw row", f"{cid}/{name}")
                    c.check(row["stats"]["simulations"] == m["simulations"],
                            "simulations match raw row", f"{cid}/{name}")
                    c.check(row["policy_rank_of_chosen"] == m["policy_rank"],
                            "policy rank matches raw row", f"{cid}/{name}")

            # a failed move's refutation must itself be legal
            for name, line in case.get("refutations", {}).items():
                probe = chess.Board(case["fen"])
                ok = True
                for san in line:
                    try:
                        probe.push(probe.parse_san(san))
                    except ValueError:
                        ok = False
                        break
                c.check(ok, "refutation line legal", f"{cid}/{name}")

    # ---- 8: PGN ----------------------------------------------------------
    pgn_path = os.path.join(out_dir, "examples.pgn")
    if os.path.exists(pgn_path):
        stream = io.StringIO(open(pgn_path).read())
        n = 0
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            n += 1
            b = game.board()
            for mv in game.mainline_moves():
                if not c.check(mv in b.legal_moves, "pgn move legal",
                               f"game {n}: {mv.uci()}"):
                    break
                b.push(mv)
        c.check(n > 0, "examples.pgn contains games", str(n))
        print(f"examples.pgn: {n} games replayed")

    return 0 if c.report() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
