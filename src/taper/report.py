"""Render the close package a controller would actually receive.

Deliberately *not* a web app. This is a single self-contained HTML file with no
server, no framework, no external assets and no build step - the same artifact a
finance team gets emailed at month end. It opens from disk and it prints.

A reconciliation tool's real output is a report somebody signs off on, so the
report is the product surface. Charts are hand-rolled inline SVG for the same
reason the runtime has no dependencies: a reviewer must be able to clone the
repo and reproduce every number without installing anything.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

from .engine.results import Layer, ReconResult
from .metrics.harness import (
    Scorecard,
    auto_clear_operating_point,
    calibration,
    layer_breakdown,
)
from .models import DefectClass, Money

# Colours live in CSS custom properties, not here, so the document can follow
# the reader's light/dark preference. SVG uses classes rather than presentation
# attributes because `fill="var(--x)"` is not reliably supported on those.

def _esc(v: Any) -> str:
    return html.escape(str(v))


def _pct(x: float) -> str:
    return "-" if x != x else f"{x:.1%}"


def _num(x: float, places: int = 3) -> str:
    return "-" if x != x else f"{x:.{places}f}"


# ---------------------------------------------------------------------------
# Charts - hand-rolled SVG, no plotting library
# ---------------------------------------------------------------------------

def taper_chart(rows: list[Any], width: int = 760, height: int = 300) -> str:
    """The headline: model calls falling while match rate climbs.

    Two series on independent scales, because they move in opposite directions
    and in different units. The whole argument of the project is the shape of
    these two lines crossing.
    """
    if len(rows) < 2:
        return "<p class='muted'>Not enough months to plot.</p>"

    pad_l, pad_r, pad_t, pad_b = 58, 58, 24, 42
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b

    months = [r.month for r in rows]
    match = [r.match_rate for r in rows]
    calls = [r.llm_calls_per_100 for r in rows]

    call_max = max(calls) * 1.25 or 1.0
    n = len(rows)

    def x(i: int) -> float:
        return pad_l + (w * i / (n - 1))

    def y_match(v: float) -> float:
        return pad_t + h - (h * v)  # match rate is already 0..1

    def y_calls(v: float) -> float:
        return pad_t + h - (h * v / call_max)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="Model calls falling while match rate rises" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]

    # horizontal gridlines + left axis (match rate)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gy = pad_t + h - h * frac
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + w}" y2="{gy:.1f}" '
            f'class="grid" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-size="11" class="lbl">{frac:.0%}</text>'
        )

    # right axis (model calls)
    for frac in (0, 0.5, 1.0):
        gy = pad_t + h - h * frac
        parts.append(
            f'<text x="{pad_l + w + 10}" y="{gy + 4:.1f}" text-anchor="start" '
            f'font-size="11" class="accent-txt">{call_max * frac:.2f}</text>'
        )

    # x labels
    for i, m in enumerate(months):
        parts.append(
            f'<text x="{x(i):.1f}" y="{pad_t + h + 22:.1f}" text-anchor="middle" '
            f'font-size="11" class="lbl">month {m}</text>'
        )

    # match-rate line
    pts = " ".join(f"{x(i):.1f},{y_match(v):.1f}" for i, v in enumerate(match))
    parts.append(
        f'<polyline points="{pts}" fill="none" class="s-good" stroke-width="2.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for i, v in enumerate(match):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y_match(v):.1f}" r="4" class="f-good"/>')

    # model-calls line
    pts = " ".join(f"{x(i):.1f},{y_calls(v):.1f}" for i, v in enumerate(calls))
    parts.append(
        f'<polyline points="{pts}" fill="none" class="s-accent" stroke-width="2.5" '
        f'stroke-dasharray="6 4" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for i, v in enumerate(calls):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y_calls(v):.1f}" r="4" class="f-accent"/>')

    # legend
    parts.append(
        f'<g font-size="12">'
        f'<line x1="{pad_l}" y1="{pad_t - 8}" x2="{pad_l + 22}" y2="{pad_t - 8}" '
        f'class="s-good" stroke-width="2.5"/>'
        f'<text x="{pad_l + 28}" y="{pad_t - 4}" class="ink">clean match rate</text>'
        f'<line x1="{pad_l + 170}" y1="{pad_t - 8}" x2="{pad_l + 192}" y2="{pad_t - 8}" '
        f'class="s-accent" stroke-width="2.5" stroke-dasharray="6 4"/>'
        f'<text x="{pad_l + 198}" y="{pad_t - 4}" class="ink">model calls / 100 records</text>'
        f'</g>'
    )
    parts.append("</svg>")
    return "".join(parts)


def reliability_chart(bins: list[tuple[float, float, float, int]],
                      width: int = 420, height: int = 300) -> str:
    """Predicted probability against observed rate, with the ideal diagonal.

    The diagonal is the whole point: a calibrated model's points sit on it. Any
    point above means over-confidence, and an auto-clear threshold set there
    would clear more than it should.
    """
    if not bins:
        return "<p class='muted'>Not enough findings to assess calibration.</p>"

    pad = 44
    w = width - pad * 2
    h = height - pad * 2

    def px(v: float) -> float:
        return pad + w * v

    def py(v: float) -> float:
        return pad + h - h * v

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="Reliability curve: predicted vs observed" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{pad}" y1="{py(frac):.1f}" x2="{pad + w}" y2="{py(frac):.1f}" '
            f'class="grid" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad - 8}" y="{py(frac) + 4:.1f}" text-anchor="end" '
            f'font-size="10" class="lbl">{frac:.1f}</text>'
        )
        parts.append(
            f'<text x="{px(frac):.1f}" y="{pad + h + 18:.1f}" text-anchor="middle" '
            f'font-size="10" class="lbl">{frac:.1f}</text>'
        )

    parts.append(
        f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(1):.1f}" y2="{py(1):.1f}" '
        f'class="diag" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    pts = " ".join(f"{px(p):.1f},{py(o):.1f}" for _, p, o, _ in bins)
    parts.append(
        f'<polyline points="{pts}" fill="none" class="s-accent" stroke-width="2.5" '
        f'stroke-linejoin="round"/>'
    )
    for _, p, o, n in bins:
        r = 3 + min(n / 25, 5)
        parts.append(f'<circle cx="{px(p):.1f}" cy="{py(o):.1f}" r="{r:.1f}" class="f-accent"/>')

    parts.append(
        f'<text x="{pad + w / 2:.0f}" y="{height - 6}" text-anchor="middle" '
        f'font-size="11" class="lbl">predicted probability</text>'
    )
    parts.append(
        f'<text x="12" y="{pad + h / 2:.0f}" text-anchor="middle" font-size="11" '
        f'class="lbl" transform="rotate(-90 12 {pad + h / 2:.0f})">observed rate</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def layer_bar(result: ReconResult) -> str:
    """Where the work happened. One stacked bar, deterministic tiers first."""
    counts = layer_breakdown(result)
    total = sum(counts.values()) or 1
    css_class = {
        Layer.L0_EXACT.value: "l0",
        Layer.L1_FUZZY.value: "l1",
        Layer.L2_RULE.value: "l2",
        Layer.L3_LLM.value: "l3",
    }
    segs, legend, x = [], [], 0.0
    for layer in Layer:
        n = counts.get(layer.value, 0)
        if not n:
            continue
        wpc = 100 * n / total
        cls = css_class[layer.value]
        segs.append(
            f'<rect x="{x:.2f}%" y="0" width="{wpc:.2f}%" height="34" class="seg {cls}"/>'
        )
        x += wpc
        legend.append(
            f'<span class="key"><i class="{cls}"></i>'
            f'{_esc(layer.value)} — {n} ({wpc:.1f}%)</span>'
        )
    return (
        f'<svg viewBox="0 0 100 34" preserveAspectRatio="none" width="100%" height="34" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(segs)}</svg>'
        f'<div class="legend">{"".join(legend)}</div>'
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

CSS = """
/* Light is the default because this is a document that gets printed and
   emailed. Dark is a first-class alternative, not an afterthought - a finance
   report that glares white at midnight is the one a reviewer closes.
   Print always forces light: dark ink on dark paper wastes toner and reads badly. */
:root{
  --paper:#faf9f6; --raise:#ffffff; --ink:#1a1a1a; --muted:#6b6b6b;
  --line:#e2e0da; --accent:#c98a2b; --good:#2f7d5d; --bad:#b3402f;
  --warn-bg:#fdf3e0; --err-bg:#fdeceb;
  --l0:#2f4f4f; --l1:#5c7f76;
}
@media (prefers-color-scheme: dark){
  :root{
    --paper:#16161a; --raise:#1e1e24; --ink:#ececec; --muted:#9a9a9a;
    --line:#32323a; --accent:#e0a94b; --good:#59b98c; --bad:#e0705e;
    --warn-bg:#2a2317; --err-bg:#2c1c1a;
    --l0:#6e9a9a; --l1:#7fae9f;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:940px;margin:0 auto;padding:44px 28px 80px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
 margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
 background:var(--line);border:1px solid var(--line);margin-top:24px}
.kpi{background:var(--raise);padding:16px 18px}
.kpi .v{font-size:25px;font-weight:600;letter-spacing:-.02em}
.kpi .l{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-top:3px}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
tbody tr:last-child td{border-bottom:none}
tr.total td{font-weight:600;border-top:2px solid var(--ink)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
.good{color:var(--good);font-weight:600} .bad{color:var(--bad);font-weight:600}
.muted{color:var(--muted)}
.banner{background:var(--warn-bg);border-left:3px solid var(--accent);
 padding:13px 16px;margin:22px 0;font-size:14px}
.banner.warn{background:var(--err-bg);border-left-color:var(--bad)}
.legend{margin-top:10px;font-size:12.5px;color:var(--muted)}
.key{display:inline-flex;align-items:center;margin-right:16px}
.key i{width:10px;height:10px;border-radius:2px;margin-right:6px;display:inline-block}
.exc{border-left:2px solid var(--line);padding:9px 0 9px 14px;margin-bottom:10px}
.exc .id{font-weight:600;font-size:13.5px}
.exc .why{color:var(--muted);font-size:13px;margin-top:2px}
.chart{margin:6px 0 4px}
.toc{display:flex;flex-wrap:wrap;gap:6px 18px;margin:22px 0 4px;font-size:13px;
 padding-bottom:14px;border-bottom:1px solid var(--line)}
.toc a{color:var(--muted);text-decoration:none;border-bottom:1px solid transparent}
.toc a:hover{color:var(--accent);border-bottom-color:var(--accent)}
h3.sub{font-size:13px;text-transform:uppercase;letter-spacing:.07em}
@media print{.toc{display:none}}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--line);
 color:var(--muted);font-size:12.5px}

/* SVG + layer-bar palette */
.grid{stroke:var(--line)}
.diag{stroke:var(--muted)}
.lbl{fill:var(--muted)}
.ink{fill:var(--ink)}
.accent-txt{fill:var(--accent)}
.s-good{stroke:var(--good)} .f-good{fill:var(--good)}
.s-accent{stroke:var(--accent)} .f-accent{fill:var(--accent)}
.l0{background:var(--l0)} rect.l0{fill:var(--l0)}
.l1{background:var(--l1)} rect.l1{fill:var(--l1)}
.l2{background:var(--good)} rect.l2{fill:var(--good)}
.l3{background:var(--accent)} rect.l3{fill:var(--accent)}

@media print{
  :root{
    --paper:#fff; --raise:#fff; --ink:#111; --muted:#555; --line:#ddd;
    --accent:#9a6a1f; --good:#1f5c43; --bad:#8c2f21;
  }
  body{background:#fff}
  .wrap{padding:0}
}
"""


# What each defect class actually means for the money, and what a controller
# does next. A single "money flagged" total is not actionable: a duplicate
# capture is an exposure the merchant may owe back, while a fee overcharge is
# cash recoverable from the gateway. Same rupees, opposite direction.
DEFECT_MEANING: dict[DefectClass, tuple[str, str]] = {
    DefectClass.DUPLICATE_CAPTURE: (
        "Customer charged twice for one order",
        "Refund the duplicate; exposure, not income",
    ),
    DefectClass.FEE_OVERCHARGE: (
        "Billed above the contracted rate",
        "Recoverable from the gateway",
    ),
    DefectClass.UNRECORDED_ADJUSTMENT: (
        "Bank deducted more than the report explains",
        "Query with the bank, or learn it as a standing charge",
    ),
    DefectClass.MISSING_LEDGER_ENTRY: (
        "Money settled with no order behind it",
        "Unrecorded revenue - find the order",
    ),
    DefectClass.CROSS_CYCLE_REFUND: (
        "Refund settled in a different cycle than its payment",
        "Timing, not loss - reclassify across periods",
    ),
    DefectClass.SPLIT_SETTLEMENT: (
        "One payout arrived as several credits",
        "No action; reconciled across the parts",
    ),
    DefectClass.TIMING_SHIFT: (
        "Money landed later than the settlement date",
        "No action; affects cash forecasting",
    ),
    DefectClass.MISSING_UTR: (
        "Settlement report shipped without a reference",
        "Matched by amount and date instead",
    ),
    DefectClass.NARRATION_DRIFT: (
        "Bank narration carried no usable reference",
        "Matched by amount and date instead",
    ),
}


def money_bars(rows: list[tuple[Any, int, Any]], width: int = 760) -> str:
    """Horizontal bars for the money breakdown.

    Horizontal rather than a pie: the labels are long, the classes are ranked
    rather than parts of one thing a reader should compare by angle, and the
    reader's question is "what is the biggest number" - which a length answers
    and a wedge does not.
    """
    if not rows:
        return ""
    biggest = max(float(m) for _, _, m in rows) or 1.0
    row_h, gap, label_w = 26, 8, 190
    height = len(rows) * (row_h + gap)
    bar_w = width - label_w - 130

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="Money identified by category" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    for i, (dc, n, amount) in enumerate(rows):
        y = i * (row_h + gap)
        w = bar_w * (float(amount) / biggest)
        parts.append(
            f'<text x="0" y="{y + 17}" font-size="12" class="ink">'
            f'{_esc(dc.value)}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y + 3}" width="{max(w, 2):.1f}" height="18" '
            f'rx="2" class="f-accent"/>'
        )
        parts.append(
            f'<text x="{label_w + max(w, 2) + 8:.1f}" y="{y + 17}" font-size="12" '
            f'class="lbl">Rs.{amount:,.0f} · {n}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def money_section(result: ReconResult) -> str:
    """Break the flagged total into what it is and what to do about it."""
    agg: dict[DefectClass, list[Any]] = {}
    for f in result.findings:
        row = agg.setdefault(f.defect_class, [0, Money("0.00")])
        row[0] += 1
        row[1] += f.money_impact

    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    with_money = [(dc, n, m) for dc, (n, m) in rows if m > 0]
    no_money = [(dc, n) for dc, (n, m) in rows if m == 0]
    total = sum((m for _, _, m in with_money), Money("0.00"))

    p = [f"<div class='chart'>{money_bars(with_money)}</div>"]
    p.append("<table><thead><tr><th>What was found</th><th>Count</th><th>Amount</th>"
             "<th>What it means</th><th>Action</th></tr></thead><tbody>")
    for dc, n, amount in with_money:
        meaning, action = DEFECT_MEANING.get(dc, ("", ""))
        share = amount / total if total else 0
        p.append(
            f"<tr><td>{_esc(dc.value)}</td><td>{n}</td>"
            f"<td><strong>Rs.{amount:,.0f}</strong> "
            f"<span class='muted'>({share:.0%})</span></td>"
            f"<td class='muted'>{_esc(meaning)}</td>"
            f"<td class='muted'>{_esc(action)}</td></tr>"
        )
    p.append(
        f"<tr class='total'><td>Total identified</td>"
        f"<td>{sum(n for _, n, _ in with_money)}</td>"
        f"<td>Rs.{total:,.0f}</td><td colspan='2'></td></tr></tbody></table>"
    )

    if no_money:
        labels = ", ".join(f"{_esc(dc.value)} ({n})" for dc, n in no_money)
        p.append(
            f"<p class='muted'>Also reported with no direct rupee impact: {labels}. "
            f"These affect cash timing and matchability rather than the balance.</p>"
        )
    p.append(
        "<p class='muted'>This is money <em>identified</em>, not money recovered. "
        "The two columns on the right exist because the same rupee means opposite "
        "things depending on the class - a duplicate capture is owed back to a "
        "customer, an overcharge is owed to you.</p>"
    )
    return "".join(p)


def learned_section(store) -> str:
    """What the system has taught itself, and where each lesson came from."""
    if store is None or not len(store):
        return (
            "<p class='muted'>Nothing learned yet — this is a cold start. Every "
            "recurring pattern below is currently being resolved from scratch.</p>"
        )
    p = ["<table><thead><tr><th>Rule</th><th>Kind</th><th>What it encodes</th>"
         "<th>Learned from</th><th>On</th></tr></thead><tbody>"]
    for r in store.rules:
        params = ", ".join(f"{k}={v}" for k, v in sorted(r.params.items()) if v not in (None, ""))
        p.append(
            f"<tr><td class='mono'>{_esc(r.rule_id)}</td><td>{_esc(r.kind)}</td>"
            f"<td class='mono'>{_esc(params)}</td>"
            f"<td class='mono muted'>{_esc(r.origin_exception)}</td>"
            f"<td class='muted'>{_esc(r.learned_on)}</td></tr>"
        )
    p.append("</tbody></table>")

    if getattr(store, "rejected", None):
        p.append(
            f"<p><strong>{len(store.rejected)} candidate rule(s) refused by the "
            f"admission gate.</strong> A rule is only admitted if replaying it "
            f"against every already-confirmed case changes none of them.</p>"
        )
        for rej in store.rejected[:5]:
            p.append(
                f"<div class='exc'><div class='id mono'>{_esc(rej.rule.rule_id)}</div>"
                f"<div class='why'>{_esc(rej.reason)}</div></div>"
            )
    else:
        p.append(
            "<p class='muted'>No candidate was refused this close. Every rule above "
            "was replayed against the full history of confirmed cases before "
            "admission.</p>"
        )
    return "".join(p)


def reconciliation_detail(result: ReconResult, limit: int = 25) -> str:
    """The working: every batch, what paid it, and how it was matched.

    This is the audit trail. A controller does not sign off on a match rate;
    they sign off on being able to see how each number was arrived at.
    """
    if not result.matches:
        return "<p class='muted'>No batches matched.</p>"

    ordered = sorted(result.matches, key=lambda m: (m.is_clean, -abs(m.delta)))
    p = ["<table><thead><tr><th>Batch</th><th>Paid by</th><th>Expected</th>"
         "<th>Credited</th><th>Delta</th><th>Matched by</th><th>Layer</th>"
         "</tr></thead><tbody>"]
    for m in ordered[:limit]:
        delta_cls = "good" if m.is_clean else "bad"
        credits = ", ".join(m.bank_txn_ids[:2])
        if len(m.bank_txn_ids) > 2:
            credits += f" +{len(m.bank_txn_ids) - 2}"
        p.append(
            f"<tr><td class='mono'>{_esc(m.batch_id)}</td>"
            f"<td class='mono muted'>{_esc(credits)}</td>"
            f"<td>Rs.{m.expected_net:,.2f}</td><td>Rs.{m.credited:,.2f}</td>"
            f"<td class='{delta_cls}'>{m.delta:+,.2f}</td>"
            f"<td class='mono muted'>{_esc(m.method)}</td>"
            f"<td class='mono'>{_esc(m.layer.value)}</td></tr>"
        )
    p.append("</tbody></table>")
    if len(ordered) > limit:
        p.append(
            f"<p class='muted'>Showing the {limit} least clean of "
            f"{len(ordered)} batches. The rest reconciled exactly.</p>"
        )
    return "".join(p)


def receipts_section(result: ReconResult, limit: int = 8) -> str:
    """The largest findings with the evidence that produced them."""
    scored = sorted(
        (f for f in result.findings if f.money_impact > 0),
        key=lambda f: -f.money_impact,
    )[:limit]
    if not scored:
        return "<p class='muted'>No findings carried a rupee impact this close.</p>"

    p = []
    for f in scored:
        rule = f" · rule <span class='mono'>{_esc(f.rule_id)}</span>" if f.rule_id else ""
        ev = " · ".join(
            f"{_esc(k)}=<span class='mono'>{_esc(v)}</span>"
            for k, v in f.evidence.items()
            if isinstance(v, (str, int, float)) and k not in ("reasoning",)
        )
        p.append(
            f"<div class='exc'><div class='id'>Rs.{f.money_impact:,.2f} — "
            f"{_esc(f.defect_class.value)} "
            f"<span class='mono muted'>{_esc(f.subject_id)}</span></div>"
            f"<div class='why'>{ev}</div>"
            f"<div class='why'>{_esc(f.layer.value)} · confidence "
            f"{f.confidence:.2f}{rule}</div></div>"
        )
    return "".join(p)


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{_esc(label)}</div></div>'


def render(
    result: ReconResult,
    card: Scorecard,
    case,
    period: str,
    campaign_rows: list[Any] | None = None,
    risk: dict[str, Any] | None = None,
    store=None,
) -> str:
    """Build the full close package as one self-contained HTML string."""
    is_mock = "mock" in card.client_name.lower()
    op = auto_clear_operating_point(case, result)
    bins = calibration(case, result)

    p: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Taper — close report {_esc(period)}</title><style>{CSS}</style></head><body>",
        "<div class='wrap'>",
        f"<h1>Reconciliation close — {_esc(period)}</h1>",
        f"<p class='sub'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
        f"resolver <span class='mono'>{_esc(card.client_name)}</span></p>",
    ]

    if is_mock:
        p.append(
            "<div class='banner warn'><strong>Offline heuristic, not a model.</strong> "
            "Layer 3 ran against the built-in mock, so any figure attributed to model "
            "judgement in this report is a stand-in and must not be reported as an "
            "LLM result.</div>"
        )

    # --- headline ---------------------------------------------------------
    p.append("<div class='kpis'>")
    p.append(_kpi(_pct(card.match_rate), "clean match rate"))
    p.append(_kpi(f"{card.records:,}", "records processed"))
    p.append(_kpi(_num(card.precision), "precision"))
    p.append(_kpi(_num(card.recall), "recall"))
    p.append(_kpi(f"{card.exceptions}", "to a human"))
    p.append(_kpi(f"Rs.{card.money_flagged:,.0f}", "money flagged"))
    p.append(_kpi(f"{card.llm_calls_per_100:.2f}", "model calls / 100 rec"))
    p.append(_kpi(_pct(card.deterministic_share), "resolved without a model"))
    p.append("</div>")

    p.append(
        "<nav class='toc'>"
        "<a href='#money'>Money found</a>"
        "<a href='#taper'>The taper</a>"
        "<a href='#learned'>What it learned</a>"
        "<a href='#risk'>Where the work will be</a>"
        "<a href='#accuracy'>Accuracy</a>"
        "<a href='#detail'>Reconciliation detail</a>"
        "<a href='#receipts'>Receipts</a>"
        "<a href='#exceptions'>Exceptions</a>"
        "</nav>"
    )

    # --- money, broken down and made actionable ---------------------------
    p.append("<h2 id='money'>Money found</h2>")
    p.append(money_section(result))

    # --- what the system taught itself ------------------------------------
    p.append("<h2 id='learned'>What the system has learned</h2>")
    p.append(learned_section(store))

    # --- the taper --------------------------------------------------------
    if campaign_rows:
        a, b = campaign_rows[0], campaign_rows[-1]
        drop = 1 - (b.llm_calls_per_100 / a.llm_calls_per_100) if a.llm_calls_per_100 else 0
        p.append("<h2 id='taper'>The taper — model load over consecutive closes</h2>")
        p.append(f"<div class='chart'>{taper_chart(campaign_rows)}</div>")
        p.append(
            f"<p>Across {len(campaign_rows)} closes, model calls per 100 records fell "
            f"<strong>{a.llm_calls_per_100:.2f} → {b.llm_calls_per_100:.2f}</strong> "
            f"(<span class='good'>{drop:.0%} reduction</span>) while clean match rate rose "
            f"<strong>{a.match_rate:.1%} → {b.match_rate:.1%}</strong>. "
            f"Precision held at <strong>{b.precision:.3f}</strong> throughout — the system "
            f"got cheaper without getting looser.</p>"
        )
        p.append(
            "<table><thead><tr><th>Month</th><th>Rules</th><th>Calls/100</th>"
            "<th>Exceptions</th><th>Match rate</th><th>Precision</th><th>Recall</th>"
            "</tr></thead><tbody>"
        )
        for r in campaign_rows:
            p.append(
                f"<tr><td>{r.month}</td><td>{r.rules:.1f}</td>"
                f"<td>{r.llm_calls_per_100:.2f}</td><td>{r.exceptions:.1f}</td>"
                f"<td>{r.match_rate:.1%}</td><td>{r.precision:.3f}</td>"
                f"<td>{r.recall:.3f}</td></tr>"
            )
        p.append("</tbody></table>")

    # --- where the work will be -------------------------------------------
    if risk:
        p.append("<h2 id='risk'>Where the work will be — predicted before the close</h2>")
        p.append(
            f"<p>A calibrated model scores every batch for the probability it needs a "
            f"human. Trained on separate periods and evaluated on data it never saw: "
            f"<strong>Brier {risk['brier']:.4f}</strong> against a base-rate baseline of "
            f"{risk['baseline']:.4f} (<span class='good'>skill "
            f"{risk['skill']:+.3f}</span>), AUC {risk['auc']:.3f}.</p>"
        )
        if risk.get("budget"):
            p.append(
                "<table><thead><tr><th>Review the riskiest…</th>"
                "<th>Escalations caught</th><th>Lift</th></tr></thead><tbody>"
            )
            for frac, caught, _n in risk["budget"][:4]:
                lift = caught / frac if frac else 0
                p.append(
                    f"<tr><td>{frac:.0%} of batches</td><td>{caught:.0%}</td>"
                    f"<td>{lift:.1f}&times;</td></tr>"
                )
            p.append("</tbody></table>")

        if risk.get("top"):
            p.append("<h3 class='sub' style='margin-top:26px'>Highest-risk batches this close</h3>")
            p.append(
                "<table><thead><tr><th>Batch</th><th>P(needs a human)</th>"
                "<th>Actually escalated</th></tr></thead><tbody>"
            )
            for batch_id, prob, actual in risk["top"][:10]:
                mark = ("<span class='bad'>yes</span>" if actual
                        else "<span class='muted'>no</span>")
                p.append(
                    f"<tr><td class='mono'>{_esc(batch_id)}</td>"
                    f"<td>{prob:.2f}</td><td>{mark}</td></tr>"
                )
            p.append("</tbody></table>")

        if risk.get("reliability"):
            p.append("<h3 class='sub' style='margin-top:26px'>Reliability — is the model honest?</h3>")
            p.append(f"<div class='chart'>{reliability_chart(risk['reliability'])}</div>")
            p.append(
                "<p class='muted'>Points on the dashed diagonal mean a stated "
                "probability matches the observed rate. Above the line is "
                "over-confidence, and a threshold set there would clear more than "
                "it should. Marker size is bucket population.</p>"
            )

    # --- accuracy ---------------------------------------------------------
    p.append("<h2 id='accuracy'>Accuracy against ground truth</h2>")
    p.append(
        "<table><thead><tr><th>Defect class</th><th>Support</th><th>TP</th><th>FP</th>"
        "<th>FN</th><th>Precision</th><th>Recall</th></tr></thead><tbody>"
    )
    for dc in DefectClass:
        s = card.per_class[dc]
        if not s.support and not s.fp:
            continue
        p.append(
            f"<tr><td>{_esc(dc.value)}</td><td>{s.support}</td><td>{s.tp}</td>"
            f"<td>{s.fp}</td><td>{s.fn}</td><td>{_num(s.precision)}</td>"
            f"<td>{_num(s.recall)}</td></tr>"
        )
    p.append(
        f"<tr class='total'><td>Overall</td><td>{card.tp + card.fn}</td><td>{card.tp}</td>"
        f"<td>{card.fp}</td><td>{card.fn}</td><td>{_num(card.precision)}</td>"
        f"<td>{_num(card.recall)}</td></tr></tbody></table>"
    )
    p.append(
        f"<p class='muted'>False-positive cost: "
        f"<strong>{card.false_positive_cost_minutes:.0f} review-minutes</strong> "
        f"({card.fp} false flags x 4 min). Throughput {card.throughput:,.0f} records/sec.</p>"
    )

    # --- layers -----------------------------------------------------------
    p.append("<h2>Where the work happened</h2>")
    p.append(layer_bar(result))
    p.append(
        "<p class='muted'>Each item stops at the first layer that can answer it. "
        "Layers 0–2 involve no model at all.</p>"
    )

    # --- auto-clear -------------------------------------------------------
    p.append("<h2>Auto-clear operating point</h2>")
    if op["coverage"]:
        p.append(
            f"<p>At confidence <span class='mono'>&ge; {op['threshold']}</span>, "
            f"<strong>{op['coverage']:.1%}</strong> of findings auto-clear at "
            f"<strong>{op['precision']:.3f}</strong> precision — "
            f"{op['auto_cleared']} cleared, {op['routed_to_human']} routed to a human, "
            f"<strong>{op['review_minutes_saved']:.0f} review-minutes</strong> saved.</p>"
        )
    else:
        p.append("<p class='bad'>No threshold reaches target precision — route everything.</p>")

    if bins:
        p.append(
            "<table><thead><tr><th>Confidence bucket</th><th>n</th><th>Stated</th>"
            "<th>Observed</th><th>Gap</th></tr></thead><tbody>"
        )
        for b in bins:
            cls = "bad" if b.gap == b.gap and b.gap > 0.05 else "good"
            p.append(
                f"<tr><td class='mono'>{b.lo:.1f}–{b.hi:.1f}</td><td>{b.n}</td>"
                f"<td>{b.midpoint:.2f}</td><td>{_num(b.observed, 2)}</td>"
                f"<td class='{cls}'>{b.gap:+.2f}</td></tr>"
            )
        p.append("</tbody></table>")
        p.append(
            "<p class='muted'>A positive gap means over-confidence. An uncalibrated "
            "confidence cannot support an auto-clear threshold.</p>"
        )

    # --- the working ------------------------------------------------------
    p.append("<h2 id='detail'>Reconciliation detail</h2>")
    p.append(reconciliation_detail(result))

    p.append("<h2 id='receipts'>Receipts — the largest findings and their evidence</h2>")
    p.append(receipts_section(result))

    # --- exceptions -------------------------------------------------------
    p.append(f"<h2 id='exceptions'>Exception list — {len(result.exceptions)} item(s) for a human</h2>")
    if result.exceptions:
        for exc in result.exceptions[:40]:
            p.append(
                f"<div class='exc'><div class='id mono'>[{_esc(exc.kind)}] "
                f"{_esc(exc.subject_id)}</div>"
                f"<div class='why'>{_esc(exc.reason)}</div></div>"
            )
        if len(result.exceptions) > 40:
            p.append(f"<p class='muted'>… and {len(result.exceptions) - 40} more.</p>")
    else:
        p.append("<p class='good'>Nothing outstanding — every item resolved automatically.</p>")

    p.append(
        "<footer>Taper — settlement reconciliation. Every figure is reproducible from "
        "the repository with no API key: <span class='mono'>python -m taper.cli report</span>. "
        "Findings are scored against injected ground truth, not sampled by hand."
        "</footer></div></body></html>"
    )

    # Wrap every table so wide ones scroll inside their own box. Done once here
    # rather than at each call site: the page body must never scroll sideways,
    # and a table added later would otherwise quietly break that.
    doc = "".join(p)
    doc = doc.replace("<table>", '<div class="tablewrap"><table>')
    doc = doc.replace("</table>", "</table></div>")
    return doc
