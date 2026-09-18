"""Self-contained local HTML + Markdown report generation.

NEW module.  The HTML opens from the filesystem with no server and no CDN:
figures are embedded as inline SVG, and the board viewer is a few lines of
vanilla JavaScript over already-embedded SVG frames.  If scripting is disabled
the static board-sequence panels remain fully readable, and the report says so.
"""

from __future__ import annotations

import base64
import html
import json
import os

from analysis.stats import CONDITION_COLORS, CONDITION_LABELS

CSS = """
:root{--bg:#ffffff;--fg:#16181d;--muted:#5b6270;--line:#d9dde5;--accent:#2E7D32;--warn:#b25000;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1140px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:30px;line-height:1.25;margin:0 0 6px}
h2{font-size:22px;margin:40px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:17px;margin:26px 0 8px}
h4{font-size:15px;margin:16px 0 6px;color:var(--muted)}
p{margin:10px 0}
code,kbd{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;
 background:#f2f4f8;padding:1px 5px;border-radius:4px}
pre{background:#f7f8fa;border:1px solid var(--line);border-radius:8px;padding:12px;
 overflow-x:auto;font-size:12.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#f2f4f8;font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.sub{color:var(--muted);font-size:13.5px}
.verdict{border:2px solid var(--line);border-radius:10px;padding:16px 18px;margin:18px 0;background:#fafbfd}
.verdict .tag{display:inline-block;padding:3px 12px;border-radius:999px;font-weight:700;
 font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:#fff}
.tag.positive{background:#2E7D32}.tag.negative{background:#b3261e}
.tag.inconclusive{background:#b25000}.tag.infrastructure-only{background:#5b6270}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:14px 0}
.card{border:1px solid var(--line);border-radius:9px;padding:12px 14px;background:#fafbfd}
.card .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:19px;font-weight:650;margin-top:3px}
figure{margin:18px 0;border:1px solid var(--line);border-radius:9px;padding:12px;background:#fff}
figure svg{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{color:var(--muted);font-size:13px;margin-top:8px}
.case{border:2px solid var(--line);border-radius:12px;padding:18px;margin:26px 0;background:#fcfdff}
.case h3{margin-top:0}
.badge{display:inline-block;padding:2px 9px;border-radius:5px;font-size:12px;font-weight:650;
 border:1px solid var(--line)}
.badge.ok{background:#e6f4ea;color:#1b5e20;border-color:#a5d6a7}
.badge.no{background:#fdecea;color:#8c1d18;border-color:#f5b5b0}
.boards{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0}
.board{border:1px solid var(--line);border-radius:8px;padding:9px;background:#fff}
.board .lbl{font-size:12.5px;color:var(--muted);margin-bottom:6px;font-weight:600}
.seq{display:flex;flex-wrap:wrap;gap:12px}
.seq .board{width:274px}
.viewer{border:1px dashed var(--line);border-radius:8px;padding:12px;margin:12px 0;background:#fff}
.viewer .frame svg{max-width:340px}
.viewer button{font:inherit;padding:5px 14px;margin-right:8px;border:1px solid var(--line);
 background:#f2f4f8;border-radius:6px;cursor:pointer}
.viewer .status{font-size:13px;color:var(--muted);margin-left:8px}
.note{border-left:4px solid var(--warn);background:#fff8f0;padding:10px 14px;margin:14px 0;font-size:13.5px}
.ok-note{border-left:4px solid var(--accent);background:#f2f9f3;padding:10px 14px;margin:14px 0;font-size:13.5px}
ul{margin:8px 0 8px 20px;padding:0}
li{margin:4px 0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
@media print{
 .viewer{display:none}
 .case{break-inside:avoid;border-color:#888}
 body{font-size:11.5px}
 .seq .board{width:220px}
}
"""

VIEWER_JS = """
document.querySelectorAll('[data-viewer]').forEach(function(v){
  var frames = JSON.parse(v.getAttribute('data-frames'));
  var box = v.querySelector('.frame');
  var status = v.querySelector('.status');
  var i = 0;
  function render(){
    box.innerHTML = document.getElementById(frames[i]).innerHTML;
    status.textContent = 'Position ' + (i+1) + ' of ' + frames.length +
      ' - ' + v.getAttribute('data-labels').split('|')[i];
  }
  v.querySelector('.prev').addEventListener('click', function(){ i=(i-1+frames.length)%frames.length; render(); });
  v.querySelector('.next').addEventListener('click', function(){ i=(i+1)%frames.length; render(); });
  render();
});
"""


def _e(x):
    return html.escape(str(x))


def _pct(v):
    return "n/a" if v is None or v != v else f"{100*v:.1f}%"


def _ci(s):
    if s is None:
        return "n/a"
    if s["ci_low"] != s["ci_low"]:
        return f"{_pct(s['rate'])} (CI n/a)"
    return f"{_pct(s['rate'])} [{100*s['ci_low']:.1f}, {100*s['ci_high']:.1f}]"


def _inline_svg(path):
    try:
        with open(path) as fh:
            s = fh.read()
    except OSError:
        return "<p class='sub'>figure missing</p>"
    i = s.find("<svg")
    return s[i:] if i >= 0 else s


def _figure(fig, caption):
    if not fig:
        return ""
    return (f"<figure>{_inline_svg(fig['svg'])}"
            f"<figcaption>{caption} &nbsp;|&nbsp; data: "
            f"<code>{_e(os.path.basename(fig['data']))}</code></figcaption></figure>")


# --------------------------------------------------------------------------- #
def _comparison_table(summary):
    b = summary["primary_budget"]
    rows = []
    for c in summary["conditions"]:
        s = summary["by_budget"].get(str(b), {}).get(c)
        cost = summary["cost"].get(c, {})
        if not s:
            continue
        rows.append(
            f"<tr><td><span style='color:{CONDITION_COLORS[c]};font-weight:700'>&#9632;</span> "
            f"{_e(CONDITION_LABELS[c])}</td>"
            f"<td class='num'>{s['successes']:.0f} / {s['n']}</td>"
            f"<td class='num'>{_ci(s)}</td>"
            f"<td class='num'>{s['n_families']}</td>"
            f"<td class='num'>{cost.get('mean_simulations', float('nan')):.0f}</td>"
            f"<td class='num'>{cost.get('mean_nn_evals', float('nan')):.0f}</td>"
            f"<td class='num'>{cost.get('mean_elapsed_s', float('nan')):.3f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Condition</th><th class='num'>Solved / n</th>"
        "<th class='num'>Success rate [95% CI]</th><th class='num'>Families</th>"
        "<th class='num'>Mean sims</th><th class='num'>Mean NN evals</th>"
        "<th class='num'>Mean search time (s)</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _paired_table(summary):
    rows = []
    label = {"standard": "Standard PUCT", "quality": "Quality-only", "random": "Random"}
    for other in ["standard", "quality", "random"]:
        p = summary["paired"].get(f"feature_vs_{other}")
        if not p:
            continue
        ci = ("n/a" if p["ci_low_pp"] != p["ci_low_pp"]
              else f"[{p['ci_low_pp']:+.1f}, {p['ci_high_pp']:+.1f}]")
        rows.append(
            f"<tr><td>Feature-diverse &minus; {label[other]}</td>"
            f"<td class='num'>{p['diff_pp']:+.1f} pp</td><td class='num'>{ci}</td>"
            f"<td class='num'>{p['n_paired']}</td><td class='num'>{p['n_families']}</td>"
            f"<td class='num'>{p['a_only']}</td><td class='num'>{p['b_only']}</td>"
            f"<td class='num'>{p['both_solved']}</td><td class='num'>{p['both_failed']}</td></tr>"
        )
    if not rows:
        return "<p class='note'>No paired comparisons were produced.</p>"
    return (
        "<table><thead><tr><th>Paired comparison</th><th class='num'>Difference</th>"
        "<th class='num'>95% CI (pp)</th><th class='num'>n paired</th><th class='num'>Families</th>"
        "<th class='num'>Feature only</th><th class='num'>Baseline only</th>"
        "<th class='num'>Both solved</th><th class='num'>Both failed</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _case_html(c, idx):
    if "error" in c:
        return (f"<div class='case'><h3>{_e(c['case_id'])} &mdash; generation failed</h3>"
                f"<pre>{_e(c['error'])}</pre></div>")
    a = c["assets"]
    meth = c["methods"]

    def badge(ok):
        return ("<span class='badge ok'>VERIFIED SOLVED</span>" if ok
                else "<span class='badge no'>NOT A FORCED MATE</span>")

    panels = []
    for name, title, key in (("standard", "Standard PUCT", "standard_move"),
                             ("feature", "Feature-diverse", "feature_move")):
        if key in a and name in meth:
            m = meth[name]
            panels.append(
                f"<div class='board'><div class='lbl'>{title}: "
                f"<strong>{_e(m['move_san'])}</strong> ({_e(m['move_uci'])}) {badge(m['correct'])}</div>"
                f"{a[key]['svg_inline']}"
                f"<div class='sub'>raw policy rank #{m['policy_rank']}, p={m['policy_prob']:.4f}"
                f" &middot; {m['simulations']} sims &middot; {m['nn_evals']} NN evals"
                f" &middot; {m['elapsed_s']:.3f}s</div></div>"
            )

    mrows = "".join(
        f"<tr><td>{_e(CONDITION_LABELS[n])}</td><td class='mono'>{_e(m['move_san'])} ({_e(m['move_uci'])})</td>"
        f"<td class='num'>#{m['policy_rank']}</td><td class='num'>{m['policy_prob']:.4f}</td>"
        f"<td class='num'>{m['simulations']}</td><td class='num'>{m['nn_evals']}</td>"
        f"<td class='num'>{m['elapsed_s']:.3f}</td><td>{badge(m['correct'])}</td></tr>"
        for n, m in meth.items()
    )

    seq = a.get("sequence", [])
    frame_ids = [f"frm-{c['case_id']}-{i}" for i in range(len(seq))]
    hidden = "".join(f"<div id='{fid}' style='display:none'>{s['svg_inline']}</div>"
                     for fid, s in zip(frame_ids, seq))
    static = "".join(
        f"<div class='board'><div class='lbl'>{i+1}. {_e(s['label'])}</div>{s['svg_inline']}"
        f"<div class='sub mono'>{_e(s['fen'])}</div></div>"
        for i, s in enumerate(seq)
    )
    labels = "|".join(s["label"].replace("|", "/") for s in seq)
    viewer = ""
    if len(seq) > 1:
        viewer = (
            f"<div class='viewer' data-viewer data-frames='{_e(json.dumps(frame_ids))}' "
            f"data-labels=\"{_e(labels)}\">"
            f"<div class='frame'></div>"
            f"<div style='margin-top:8px'><button class='prev'>&#8592; Previous</button>"
            f"<button class='next'>Next &#8594;</button>"
            f"<span class='status'></span></div></div>"
        )

    refut = ""
    if c["refutations"]:
        items = []
        for name, line in c["refutations"].items():
            if line:
                items.append(f"<li><strong>{_e(name)}</strong>: {_e(' '.join(line))} "
                             f"&mdash; a legal defence that survives the {c['mate_in']}-move horizon.</li>")
            else:
                items.append(f"<li><strong>{_e(name)}</strong>: fails the mate-within-horizon "
                             f"objective; no explicit escape established inside the verifier budget.</li>")
        refut = f"<h4>Refutation of failed choices</h4><ul>{''.join(items)}</ul>"

    solset = ", ".join(f"<code>{_e(u)}</code>" for u in c["solution_moves"])
    return f"""
<div class='case'>
  <h3>{_e(c['case_id'])} &mdash; {_e(c['category_title'])}</h3>
  <p class='sub'>Task <code>{_e(c['task_id'])}</code> &middot; family <code>{_e(c['family'])}</code>
   &middot; generator <code>{_e(c['source'])}</code> &middot; <strong>{_e(c['side_to_move'])} to move</strong>
   &middot; objective: force mate in {c['mate_in']} ({c['horizon_plies']} plies)
   &middot; {c['num_legal_moves']} legal moves
   &middot; board shown from {_e(c['orientation'])}'s side.</p>
  <div class='boards'>
    <div class='board'><div class='lbl'>Initial position</div>{a['initial']['svg_inline']}
      <div class='sub mono'>{_e(c['fen'])}</div></div>
    {''.join(panels)}
  </div>
  <p><strong>Verified solution set</strong> (exact oracle, all legal defences checked): {solset}
    {'&nbsp;&mdash;&nbsp;SAN: ' + _e(', '.join(c['solution_sans'])) if c['solution_sans'] else ''}.
    Best verified solution&rsquo;s raw-policy rank: <strong>#{c['best_solution_policy_rank']}</strong>
    (p = {c['best_solution_policy_prob']:.4f}).</p>
  <h4>All four conditions on this position</h4>
  <table><thead><tr><th>Condition</th><th>Chosen move</th><th class='num'>Policy rank</th>
    <th class='num'>Policy p</th><th class='num'>Sims</th><th class='num'>NN evals</th>
    <th class='num'>Time (s)</th><th>Verified result</th></tr></thead><tbody>{mrows}</tbody></table>
  <h4>Verified winning continuation (from the oracle, with a most-stubborn legal defence)</h4>
  <p class='sub'>This line <em>illustrates</em> the oracle's proof. The proof itself quantifies over
   <em>every</em> legal defending reply, not just the one shown. It is distinct from any method's own
   search principal variation.</p>
  {hidden}{viewer}
  <div class='seq'>{static}</div>
  {refut}
  <h4>What happened</h4>
  <p>{_e(c['explanation'])}</p>
  <p class='sub'>PGN: <code>case_studies/pgn/{_e(c['case_id'])}.pgn</code> &middot;
   raw result ids: <span class='mono'>{_e(json.dumps(c['result_ids']))}</span></p>
</div>"""


# --------------------------------------------------------------------------- #
def build_html(ctx):
    s = ctx["summary"]
    figs = ctx["figures"]
    hw = ctx["hardware"]
    proto = ctx["protocol"]
    v_kind, v_text = ctx["verdict"]
    cases = ctx["cases"]
    cidx = ctx["case_index"]
    dm = ctx["data_manifest"]
    tr = ctx["training_summary"]
    ck = ctx["checkpoint_metadata"]
    stages = ctx["stages"]

    b = s["primary_budget"]
    feat_s = s["by_budget"].get(str(b), {}).get("feature")
    std_s = s["by_budget"].get(str(b), {}).get("standard")

    cards = f"""
    <div class='grid'>
      <div class='card'><div class='k'>Held-out test positions</div><div class='v'>{s['n_test_positions']}</div>
        <div class='sub'>{s['n_families']} independent source families</div></div>
      <div class='card'><div class='k'>Standard PUCT success</div><div class='v'>{_pct(std_s['rate']) if std_s else 'n/a'}</div>
        <div class='sub'>{f"{std_s['successes']:.0f}/{std_s['n']}" if std_s else ''}</div></div>
      <div class='card'><div class='k'>Feature-diverse success</div><div class='v'>{_pct(feat_s['rate']) if feat_s else 'n/a'}</div>
        <div class='sub'>{f"{feat_s['successes']:.0f}/{feat_s['n']}" if feat_s else ''}</div></div>
      <div class='card'><div class='k'>Primary budget</div><div class='v'>{b}</div>
        <div class='sub'>MCTS simulations per position</div></div>
      <div class='card'><div class='k'>Overlooked-solution subset</div><div class='v'>{s['overlooked_denominator']}</div>
        <div class='sub'>all solutions outside raw-policy top 3</div></div>
      <div class='card'><div class='k'>Total run time</div><div class='v'>{ctx['runtime_s']/60:.1f} min</div>
        <div class='sub'>budget {proto['max_minutes']} min</div></div>
    </div>"""

    stage_rows = "".join(
        f"<tr><td>{_e(k)}</td><td>{'&#10003; completed' if v.get('completed') else '&#10007; not completed'}</td>"
        f"<td class='num'>{v.get('elapsed_s', 0):.1f}</td><td>{_e(v.get('note',''))}</td></tr>"
        for k, v in stages.items()
    )

    case_html = "".join(_case_html(c, i) for i, c in enumerate(cases))
    empty_note = ""
    if cidx["empty_categories"]:
        empty_note = ("<p class='note'><strong>Empty case-study categories:</strong> "
                      + _e(", ".join(cidx["empty_categories"]))
                      + ". No examples exist in the completed data for these categories; "
                        "nothing has been invented to fill them.</p>")

    excl = s["exclusions"]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feature-guided exploration in a small AlphaZero-style chess agent</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>Does exploring candidates with different chess features help a small
AlphaZero-style agent find moves its own policy overlooked?</h1>
<p class="sub">Local research prototype built on the <code>ChessConceptDiscovery</code> repository.
Run <code>{_e(ctx['run_id'])}</code> &middot; generated {_e(ctx['generated_at'])}.</p>

<div class="verdict">
  <span class="tag {v_kind}">{_e(v_kind)}</span>
  <p style="margin-bottom:0"><strong>Plain-language question.</strong> At a fixed search budget, a chess
  agent usually spends its thinking time on the moves its network already likes. We asked whether
  deliberately spending part of that same budget on moves that are <em>different in character</em>
  (sacrifices, checks, promotions, king-side vs queen-side play) helps the agent find winning moves
  its policy initially ranked low.</p>
  <p style="margin-bottom:0"><strong>Verdict.</strong> {_e(v_text)}</p>
</div>

{cards}

<h2>1. What this study is and is not</h2>
<p>This is a small, fully local, reproducible experiment about <em>search-budget allocation</em>.
The "creativity" component under test is operationalised narrowly: <strong>discovering a useful
solution that was initially unlikely under the agent's own policy</strong>. Success is judged by an
exact forced-mate verifier, never by the network's own value head and never by an external engine.</p>
<div class="note"><strong>Claim boundaries.</strong> No claim is made about novelty to humanity,
artistic ability, subjective experience, or general creative intelligence. Novelty here is
<em>relative to the frozen network's initial policy only</em>; withholding a task from training does
not prove the underlying concept was absent from all training experience. This study does not
demonstrate concept recombination. It does not reproduce DeepMind's models or results.
The chess-feature distance used in condition 4 is the hypothesis being tested &mdash; it is
<em>not</em> an established measure of creativity.</div>

<h2>2. Setup actually used</h2>
<h3>Hardware and runtime</h3>
<table><tbody>
<tr><th>Platform</th><td>{_e(hw['platform'])}</td></tr>
<tr><th>Processor</th><td>{_e(hw.get('cpu_brand','?'))}, {hw['logical_cores']} logical cores
  (capped at {hw['thread_cap']} for this run)</td></tr>
<tr><th>Memory</th><td>{hw['total_ram_gib']:.1f} GiB total, run capped at {hw['memory_cap_gib']:.1f} GiB</td></tr>
<tr><th>Accelerator</th><td>{_e(hw['accelerator'])}</td></tr>
<tr><th>Python / TensorFlow / python-chess</th><td>{_e(hw['python'])} / {_e(hw['tensorflow'])} / {_e(hw['python_chess'])}</td></tr>
<tr><th>Total wall time</th><td>{ctx['runtime_s']/60:.1f} min (budget {proto['max_minutes']} min)</td></tr>
</tbody></table>

<h3>Dataset actually generated</h3>
<table><thead><tr><th>Split</th><th class='num'>Positions</th><th class='num'>Families</th>
<th>Mate-in counts</th><th>Generators</th><th class='num'>Median legal moves</th></tr></thead><tbody>
{''.join(f"<tr><td>{_e(k)}</td><td class='num'>{v['n']}</td><td class='num'>{v['families']}</td>"
         f"<td>{_e(json.dumps(v['mate_in_counts']))}</td><td>{_e(json.dumps(v['sources']))}</td>"
         f"<td class='num'>{v['median_legal_moves']}</td></tr>"
         for k, v in dm['splits'].items())}
</tbody></table>
<p class="sub">Splits are by <strong>source family</strong> (one random playout game or one endgame
construction seed), never by adjacent row. Exact FEN duplicates and duplicate piece placements are
removed globally before splitting. Board transformations are deliberately not applied because
mirroring breaks castling and pawn-direction semantics; this leakage-detection limit is recorded in
<code>data_manifest.json</code>. Every task stores the legal move sequence that produced it, so
<code>board.is_valid()</code> is never used as a stand-in for reachability.</p>

<h3>Checkpoint provenance</h3>
<table><tbody>
<tr><th>Training procedure</th><td>{_e(ck['procedure'])}</td></tr>
<tr><th>Parameters</th><td>{ck['parameter_count']:,}</td></tr>
<tr><th>Architecture</th><td class='mono'>{_e(json.dumps(ck['arch']))}</td></tr>
<tr><th>Weights SHA-256</th><td class='mono'>{_e(ck['weights_sha256'])}</td></tr>
<tr><th>Bootstrap steps</th><td>{ck.get('bootstrap_steps', 0)} gradient steps on the TRAIN split</td></tr>
<tr><th>Self-play</th><td>{_e(ck.get('selfplay_note','not run'))}</td></tr>
</tbody></table>
<div class="note">This is <strong>supervised tactical bootstrapping followed by an AlphaZero-style
procedure</strong>, not tabula-rasa AlphaZero. The policy head was trained on root moves that the
exact verifier proved winning on the TRAIN split; the value head received supervision only where the
verifier returned a decisive, complete result. Positions with unknown game-theoretic value, and all
truncated self-play games, carry a masked (zero-weight) value target rather than a fabricated draw
label.</div>

<h2>3. The four conditions</h2>
<p>All conditions share the same frozen checkpoint, environment, legal action space,
observation encoding and nominal simulation budget.</p>
<table><thead><tr><th>Condition</th><th>What it does</th></tr></thead><tbody>
<tr><td><strong>Standard PUCT</strong></td><td>Ordinary neural-guided MCTS. All
{b} simulations from the root; move chosen by root visit count. No diversity intervention.</td></tr>
<tr><td><strong>Quality-only</strong></td><td>Screening search of
{int(round(proto['allocation']['screen_frac']*b))} simulations, then the remaining budget split evenly over the
top-{proto['allocation']['num_candidates']} candidates by screening visit share.</td></tr>
<tr><td><strong>Random</strong></td><td>Same screening and candidate count; the top screening move is
kept as an anchor and the other {proto['allocation']['num_candidates']-1} candidates are sampled uniformly
from the eligible pool with recorded seeds.</td></tr>
<tr><td><strong>Feature-diverse</strong></td><td>Same screening and candidate count; candidates chosen
greedily to maximise <code>normalised screening quality + {proto['allocation']['diversity_weight']} &times;
minimum standardised feature distance</code> to the already-selected candidates.</td></tr>
</tbody></table>
<p class="sub"><strong>Eligible pool:</strong> every legal root move (cap
{proto['allocation']['pool_cap']}, never binding on this benchmark), so low-prior alternatives are always
reachable. <strong>Final move rule</strong> (identical for conditions 2&ndash;4): the candidate with the
highest post-allocation root-player value estimate, tie-broken by visits then action id. Every
candidate receives at least one simulation, so there are no unvisited candidates at selection time.
<strong>Budget accounting:</strong> the screening search is run once per position, deep-copied for each
condition, and its full simulation, network-evaluation and wall-time cost is charged to every
condition that uses it. Feature computation time is measured and added to the feature-diverse
condition even though it only reads already-produced states.</p>

<h2>4. Results on the held-out test set</h2>
{_comparison_table(summary=s)}
{_figure(figs.get('fig1_success_vs_budget'), 'Figure 1. Task success versus search budget.')}
<h3>Primary comparison and controls</h3>
{_paired_table(s)}
{_figure(figs.get('fig2_paired_differences'), 'Figure 2. Paired differences with a zero reference line.')}
{_figure(figs.get('fig5_win_tie_loss'), 'Figure 5. Per-position win / both-solved / both-failed / loss counts.')}

<h3>Recovery of initially overlooked solutions</h3>
<p>Eligibility was fixed in the protocol <em>before</em> test scoring and is computed from the single
common raw network policy, so every method is scored on the same {s['overlooked_denominator']} positions:
a position is eligible when <strong>all</strong> verified solution moves fall outside the raw policy's
top three legal moves.</p>
{_figure(figs.get('fig3_overlooked_recovery'), 'Figure 3. Success on the fixed eligible subset, with denominators.')}

<h3>Success versus computation</h3>
{_figure(figs.get('fig4_success_vs_compute'), 'Figure 4. Success against measured per-position time and network evaluations.')}
<h3>Case-level policy ranks</h3>
{_figure(figs.get('fig6_candidate_ranks'), 'Figure 6. Initial policy rank of each method&rsquo;s selected move versus the verified solution. Ranks are MODEL ESTIMATES; the green ring is exact verification.')}

<h2>5. Training and development diagnostics</h2>
<p class="sub">These plots describe the training and development phases. They are <strong>not</strong>
held-out test performance and must not be read as evidence of chess competence.</p>
{_figure(figs.get('fig7_training'), 'Figure 7. Training losses (train split only).')}
{_figure(figs.get('fig8_dev_calibration'), 'Figure 8. Development calibration of the diversity weight. All hyper-parameters were frozen from this grid before the test set was opened.')}

<h2>6. Illustrated chess case studies</h2>
<p class="sub">Selection rule: {_e(cidx['selection_rule'])}
Category counts in the completed data: <span class='mono'>{_e(json.dumps(cidx['category_counts']))}</span>.</p>
{empty_note}
<div class="ok-note">The interactive previous/next viewer below each case requires browser scripting.
The static board-sequence panels underneath carry the same information and remain complete when
printed or exported. A printable version is at <code>case_studies.html</code>.</div>
{case_html}

<h2>7. Frozen protocol</h2>
<p>The protocol was written and hashed <strong>after</strong> development calibration and
<strong>before</strong> any test score was computed. No hyper-parameter was adjusted on test data,
and no seed or variant was selected for the headline.</p>
<table><tbody>
<tr><th>Protocol SHA-256</th><td class='mono'>{_e(proto['protocol_sha256'])}</td></tr>
<tr><th>Frozen at</th><td>{_e(proto['frozen_at'])}</td></tr>
<tr><th>Primary endpoint</th><td>{_e(proto['primary_endpoint'])}</td></tr>
<tr><th>Primary comparison</th><td>{_e(proto['primary_comparison'])}</td></tr>
<tr><th>Verdict rule</th><td>{_e(proto['verdict_rule'])}</td></tr>
<tr><th>Master seed</th><td>{proto['seed']}</td></tr>
<tr><th>Test split hash</th><td class='mono'>{_e(dm['splits']['test']['split_hash'])}</td></tr>
</tbody></table>
<details><summary>Full protocol JSON</summary><pre>{_e(json.dumps(proto, indent=2))}</pre></details>

<h2>8. Stages, limitations and exclusions</h2>
<table><thead><tr><th>Stage</th><th>Status</th><th class='num'>Seconds</th><th>Note</th></tr></thead>
<tbody>{stage_rows}</tbody></table>
<h3>Limitations</h3>
<ul>
<li><strong>Benchmark scope.</strong> The tasks are short bounded forced mates with modest material.
Performance here says nothing about general chess strength.</li>
<li><strong>Model strength.</strong> The network is a small convolutional model trained for minutes on
a laptop CPU with supervised tactical bootstrapping. It is weak, and both its policy ranks and value
estimates should be read as model estimates only.</li>
<li><strong>Observation is not fully Markov.</strong> The network input carries a single repetition bit
and no move history. The search state keeps the full board with its move stack, so game rules are
applied exactly, but the network cannot see history.</li>
<li><strong>Feature space is a hypothesis.</strong> Distances between descriptors are not a validated
measure of strategic difference, and a feature named <code>sacrifice_proxy</code> records an observed
material deficit along a short line &mdash; it does not assert a sound sacrifice.</li>
<li><strong>Leakage detection is partial.</strong> Near-duplicate removal is piece-placement based.
Positions differing only in en-passant or halfmove counters are retained.</li>
<li><strong>Statistical power.</strong> Confidence intervals resample {s['n_families']} independent source
families. With this many clusters the intervals are wide; read them, not the point estimates.</li>
<li><strong>Excluded / anomalous rows.</strong> Illegal selected moves: {excl['invalid_moves']}.
Rows whose oracle status was not <code>ok</code>: {excl['solver_unknown']} (such tasks are filtered out
at generation time and excluded from verified scoring).</li>
</ul>

<h2>9. Reproduce this run</h2>
<pre>git clone https://github.com/jaycewalzer1/ChessConceptDiscovery
cd ChessConceptDiscovery
python -m venv .venv &amp;&amp; . .venv/bin/activate
pip install -r requirements.txt

# smoke test (about a minute)
python -m scripts.run_experiment --profile smoke --seed 42

# the run shown in this report
{_e(ctx['command'])}

# resume an interrupted laptop run
python -m scripts.run_experiment --profile laptop --resume --output-dir outputs/{_e(ctx['run_id'])}</pre>
<p class="sub">Artifacts for this run live in <code>outputs/{_e(ctx['run_id'])}/</code>:
<code>protocol.json</code>, <code>hardware.json</code>, <code>data_manifest.json</code>,
<code>training_summary.json</code>, <code>checkpoint_metadata.json</code>, <code>results.jsonl</code>,
<code>summary.json</code>, <code>report.md</code>, <code>report.html</code>,
<code>case_studies.html</code>, <code>examples.pgn</code>, <code>figures/</code>,
<code>case_studies/</code>, <code>logs/</code>.</p>

<h2>10. Related work</h2>
<p>This prototype is a small local study inspired by two published lines of work, which it does not
reproduce:</p>
<ul>
<li>Schut et al., <em>Bridging the Human&ndash;AI Knowledge Gap: Concept Discovery and Transfer in
AlphaZero</em>, arXiv:2310.16410.</li>
<li>Zahavy et al., <em>Diversifying AI: Towards Creative Chess with AlphaZero</em>, arXiv:2308.09175.</li>
</ul>
<p class="sub">{_e(ctx['literature_note'])}</p>

<script>{VIEWER_JS}</script>
</div></body></html>"""


def build_case_studies_html(ctx):
    cases = ctx["cases"]
    cidx = ctx["case_index"]
    empty_note = ""
    if cidx["empty_categories"]:
        empty_note = ("<p class='note'><strong>Empty categories:</strong> "
                      + _e(", ".join(cidx["empty_categories"])) + ".</p>")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Case studies &mdash; {_e(ctx['run_id'])}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>Illustrated chess case studies</h1>
<p class="sub">Run <code>{_e(ctx['run_id'])}</code>. Selection rule: {_e(cidx['selection_rule'])}
Category counts: <span class='mono'>{_e(json.dumps(cidx['category_counts']))}</span>.
This page is designed to stay complete when printed; the interactive viewers are hidden in print.</p>
{empty_note}
{''.join(_case_html(c, i) for i, c in enumerate(cases))}
<script>{VIEWER_JS}</script>
</div></body></html>"""


def build_markdown(ctx):
    s = ctx["summary"]
    v_kind, v_text = ctx["verdict"]
    b = s["primary_budget"]
    lines = [
        "# Feature-guided exploration in a small AlphaZero-style chess agent",
        "",
        f"Run `{ctx['run_id']}` | generated {ctx['generated_at']} | "
        f"total wall time {ctx['runtime_s']/60:.1f} min",
        "",
        f"## Verdict: {v_kind.upper()}",
        "",
        v_text,
        "",
        "## Question",
        "",
        "At a fixed search budget, does explicitly exploring candidate moves with different chess",
        "features help a small AlphaZero-style agent recover correct moves its initial policy",
        "overlooked? Creativity is operationalised narrowly as *discovering a useful solution that",
        "was initially unlikely under the agent's own policy*. No claim is made about novelty to",
        "humanity, artistic ability, or general creative intelligence.",
        "",
        "## Setup",
        "",
        f"- Hardware: {ctx['hardware'].get('cpu_brand','?')}, "
        f"{ctx['hardware']['logical_cores']} logical cores "
        f"(capped at {ctx['hardware']['thread_cap']}), "
        f"{ctx['hardware']['total_ram_gib']:.0f} GiB RAM, accelerator: {ctx['hardware']['accelerator']}",
        f"- Test set: {s['n_test_positions']} positions across {s['n_families']} independent families",
        f"- Primary budget: {b} MCTS simulations per position",
        f"- Checkpoint: {ctx['checkpoint_metadata']['parameter_count']:,} parameters, "
        f"SHA-256 `{ctx['checkpoint_metadata']['weights_sha256'][:16]}...`",
        f"- Training: {ctx['checkpoint_metadata']['procedure']}",
        "",
        "## Results at the primary budget",
        "",
        "| Condition | Solved / n | Success rate [95% CI] | Mean sims | Mean NN evals | Mean time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in s["conditions"]:
        st = s["by_budget"].get(str(b), {}).get(c)
        cost = s["cost"].get(c, {})
        if not st:
            continue
        lines.append(
            f"| {CONDITION_LABELS[c]} | {st['successes']:.0f}/{st['n']} | {_ci(st)} | "
            f"{cost.get('mean_simulations',0):.0f} | {cost.get('mean_nn_evals',0):.0f} | "
            f"{cost.get('mean_elapsed_s',0):.3f} |"
        )
    lines += ["", "## Paired comparisons", "",
              "| Comparison | Difference | 95% CI (pp) | n | Feature only | Baseline only | Both solved | Both failed |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    label = {"standard": "Standard PUCT", "quality": "Quality-only", "random": "Random"}
    for other in ["standard", "quality", "random"]:
        p = s["paired"].get(f"feature_vs_{other}")
        if not p:
            continue
        ci = ("n/a" if p["ci_low_pp"] != p["ci_low_pp"]
              else f"[{p['ci_low_pp']:+.1f}, {p['ci_high_pp']:+.1f}]")
        lines.append(
            f"| Feature-diverse - {label[other]} | {p['diff_pp']:+.1f} pp | {ci} | {p['n_paired']} | "
            f"{p['a_only']} | {p['b_only']} | {p['both_solved']} | {p['both_failed']} |"
        )
    lines += [
        "",
        f"## Overlooked-solution subset (n = {s['overlooked_denominator']})",
        "",
        "Eligibility (fixed before test scoring): all verified solutions fall outside the raw",
        "policy's top three legal moves. The denominator is identical for every method.",
        "",
        "| Condition | Solved / n | Rate [95% CI] |",
        "|---|---:|---:|",
    ]
    for c in s["conditions"]:
        st = s["overlooked"].get(c)
        if not st:
            continue
        lines.append(f"| {CONDITION_LABELS[c]} | {st['successes']:.0f}/{st['n']} | {_ci(st)} |")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- `outputs/{ctx['run_id']}/report.html` - full illustrated report",
        f"- `outputs/{ctx['run_id']}/case_studies.html` - printable case-study gallery",
        f"- `outputs/{ctx['run_id']}/figures/` - standalone SVG + PNG + plotted-data JSON",
        f"- `outputs/{ctx['run_id']}/case_studies/figures/` - standalone board SVGs",
        f"- `outputs/{ctx['run_id']}/examples.pgn` - case-study PGNs",
        f"- `outputs/{ctx['run_id']}/results.jsonl` - raw per-position rows",
        "",
        "## Reproduce",
        "",
        "```",
        ctx["command"],
        "```",
    ]
    return "\n".join(lines)
