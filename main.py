#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case file per run (schema makefun/gta6-trophies-case/v1) from the
input directory and writes, for every trophy, five quantiles of its Exophase
earn rate at the case target, as {"trophies": [{"id": n, "q": [...]}]}.

Model (standard library only, deterministic, no network):
  * The page the house records is a step function: frozen between Exophase
    refreshes, jumping at each refresh.  The informative points are the
    refresh endpoints, so every trend is fitted on the reads where a trophy's
    rate (or the players-tracked count) actually changed.
  * Earners and players are forecast separately, then divided.  For trophy i,
    earners E_i(t) = rate_i(t) * players(t) / 100 only ever grow, so
    log E_i is fitted against x = ln(1 + hours since launch) with a
    recency-weighted least squares line; its slope is shrunk toward the
    tier's mean slope (platinum/gold/silver/bronze).  Players tracked N(t)
    is fitted the same way once, globally.  The median rate at the target
    is E_i*/N*, anchored on the last read's values.
  * Spread, in logit space, combines a random-walk term growing with the
    horizon, the fit's residual, and the observed refresh-jump dispersion.
  * Flat history (no refresh ever seen, e.g. the practice cases): the page
    has been frozen for hours, so a refresh is overdue.  The forecast is a
    two-component mixture: with probability p a refresh lands before the
    target and dilutes every non-100% rate (newly tracked players hold far
    fewer trophies than the launch-night core), with probability 1-p the
    page stays put.  p grows with the staleness of the page; the dilution
    size uses the launch-night refresh the kit documents (638 -> 3,868
    players tracked) when the tracked base is still small.
  * A trophy at 100% stays 100% (every tracked player holds it); a trophy
    at 0% is anchored on one hypothetical earner with its tier's prior
    slope, so its upper quantiles can still move.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at the five levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
# Prior log-time slope of log earners, per tier, used only where a trophy
# has no observed movement at all.
TIER_PRIOR_SLOPE = {"platinum": 0.30, "gold": 0.50,
                    "silver": 0.60, "bronze": 0.80}
# Launch-night refresh the kit documents: the page sat unchanged from the
# first probe (2026-11-18T23:48Z) until the read at 2026-11-19T04:30Z, when
# players tracked went 638 -> 3,868.  Refreshes are "hours apart", so the
# prior mean interval is ~4.7 h.
PRIOR_REFRESH_INTERVAL_H = 4.7
PRIOR_LAUNCH_GROWTH = 3868.0 / 638.0
# Newly tracked players held, beyond the 100% starter trophy, about 103
# trophies among 3,230 of them, vs 305 among the 638 launch-night players:
# a per-trophy earn fraction of roughly (103/3230) / (305/638) ~= 0.067.
PRIOR_NEW_PLAYER_EARN_FRAC = 0.067


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lo):
    if lo > 20.0:
        return 100.0
    if lo < -20.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lo))


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def weighted_line(xs, ys, ws):
    """Recency-weighted least squares line; returns (slope, resid_std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    resid = [y - (inter + slope * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return slope, math.sqrt(max(var, 0.0))


def mixture_quantiles(median_logit, sigma, shock_prob, shock_down):
    """Quantiles of a two-component normal mixture in logit space.

    With probability (1 - shock_prob) the logit is N(median_logit, sigma);
    with probability shock_prob it is N(median_logit - shock_down, sigma)
    (a refresh diluting the rate).  shock_prob 0 or shock_down 0 reduces to
    the single normal, whose quantiles are median + z * sigma.
    """
    if shock_prob <= 0.0 or shock_down <= 0.0:
        return [median_logit + z * sigma for z in Z]
    lo = median_logit - shock_down - 6.0 * sigma
    hi = median_logit + 6.0 * sigma

    def cdf(x):
        return ((1.0 - shock_prob) * norm_cdf((x - median_logit) / sigma)
                + shock_prob * norm_cdf((x - (median_logit - shock_down)) / sigma))

    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(80):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append(0.5 * (a + b))
    return out


def hours_since_launch(moment):
    return max(0.0, (moment - LAUNCH).total_seconds() / 3600.0)


def x_of(moment):
    return math.log1p(hours_since_launch(moment))


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    x_target = x_of(target)

    reads = []
    for rec in history:
        moment = parse_time(rec["at"])
        rates = rec.get("rates") or []
        reads.append({"t": moment, "x": x_of(moment),
                      "n": rec.get("players_tracked"), "rates": rates})
    reads.sort(key=lambda r: r["t"])

    # --- refresh events: reads where anything on the page changed -------
    events = []
    for j in range(1, len(reads)):
        prev, cur = reads[j - 1], reads[j]
        changed = prev["n"] != cur["n"]
        if not changed:
            for a, b in zip(prev["rates"], cur["rates"]):
                if a != b:
                    changed = True
                    break
        if changed:
            events.append(j)
    n_events = len(events)
    last_change_t = reads[events[-1]]["t"] if events else (
        reads[0]["t"] if reads else checkpoint)
    staleness_h = max(0.0, (checkpoint - last_change_t).total_seconds() / 3600.0)

    # --- players tracked: one global log-time fit on change points ------
    n_points = []
    for j, rec in enumerate(reads):
        if rec["n"] and (j == 0 or j in events or j == len(reads) - 1):
            n_points.append(rec)
    n_last = reads[-1]["n"] if reads else None
    x_last = reads[-1]["x"] if reads else x_of(checkpoint)
    if len(n_points) >= 2 and n_last:
        xs = [r["x"] for r in n_points]
        ys = [math.log(max(r["n"], 1)) for r in n_points]
        ws = [0.5 ** ((x_last - x) / 24.0) for x in xs]
        slope_n, _ = weighted_line(xs, ys, ws)
        slope_n = min(max(slope_n, 0.0), 1.2)
    else:
        slope_n = 0.0
    if n_last:
        n_star = n_last * math.exp(slope_n * (x_target - x_last))
        n_star = min(max(n_star, n_last), 30.0 * n_last)
    else:
        n_star = None

    # --- per-trophy earners fits ----------------------------------------
    tier_slopes = {}
    per_trophy = {}
    for idx, tr in enumerate(trophies):
        pts = []  # (x, log(earners + 0.5)) at change points + first/last read
        for j, rec in enumerate(reads):
            if idx >= len(rec["rates"]):
                continue
            rate = rec["rates"][idx]
            if rate is None or rec["n"] is None:
                continue
            changed = (j == 0 or j == len(reads) - 1 or j in events
                       or rec["rates"][idx] != reads[j - 1]["rates"][idx])
            if not changed:
                continue
            earners = rate / 100.0 * rec["n"]
            pts.append((rec["x"], math.log(earners + 0.5)))
        last_rate = None
        for rec in reversed(reads):
            if idx < len(rec["rates"]) and rec["rates"][idx] is not None:
                last_rate = rec["rates"][idx]
                break
        if last_rate is None:
            last_rate = 0.01
        slope_i, resid_i = 0.0, 0.0
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [0.5 ** ((x_last - x) / 24.0) for x in xs]
            slope_i, resid_i = weighted_line(xs, ys, ws)
            slope_i = min(max(slope_i, -0.10), 2.0)
        per_trophy[idx] = {
            "tier": tr.get("tier", "bronze"), "last": last_rate,
            "slope": slope_i, "resid": resid_i, "npts": len(pts),
        }

    for tier in ("platinum", "gold", "silver", "bronze"):
        moved = [v["slope"] for v in per_trophy.values()
                 if v["tier"] == tier and v["npts"] >= 3]
        if moved:
            tier_slopes[tier] = sum(moved) / len(moved)
        else:
            # No trophy of this tier moved at all: the data says flat, so
            # the tier mean is 0.  (The fixed TIER_PRIOR_SLOPE is reserved
            # for trophies that sit at 0% and can only move up.)
            tier_slopes[tier] = 0.0

    # --- refresh-jump dispersion, from observed events -------------------
    shock_sigma = 0.0
    if n_events >= 1:
        resids = []
        for j in events:
            prev, cur = reads[j - 1], reads[j]
            dx = cur["x"] - prev["x"]
            for idx in range(len(trophies)):
                if idx >= len(prev["rates"]) or idx >= len(cur["rates"]):
                    continue
                r0, r1 = prev["rates"][idx], cur["rates"][idx]
                if r0 is None or r1 is None or not (0 < r0 < 100) \
                        or not (0 < r1 < 100):
                    continue
                expect = per_trophy[idx]["slope"] * dx
                resids.append(math.log(r1 / r0) - expect)
        if len(resids) >= 5:
            med = sorted(resids)[len(resids) // 2]
            mad = sorted(abs(r - med) for r in resids)[len(resids) // 2]
            shock_sigma = min(max(1.4826 * mad, 0.05), 0.5)

    # --- pending-refresh shock, only when no refresh was ever seen ------
    shock_prob, shock_down = 0.0, 0.0
    if n_events == 0 and reads and horizon_h > 0:
        remaining = max(0.5, PRIOR_REFRESH_INTERVAL_H - staleness_h)
        shock_prob = 1.0 - math.exp(-horizon_h / remaining)
        if n_last and n_last < 2000:
            growth = PRIOR_LAUNCH_GROWTH
            alpha = PRIOR_NEW_PLAYER_EARN_FRAC
        else:
            growth, alpha = 1.5, 0.20
        factor = (1.0 + alpha * (growth - 1.0)) / growth
        shock_down = -math.log(max(factor, 0.02))

    sigma_rw = 0.45 * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for idx, tr in enumerate(trophies):
        info = per_trophy[idx]
        last = info["last"]
        if last >= 99.99:
            rows.append({"id": tr["id"], "q": [round(float(last), 4)] * 5})
            continue
        # Shrink the per-trophy slope toward its tier mean.
        w = info["npts"] / (info["npts"] + 8.0) if info["npts"] else 0.0
        if info["npts"] >= 2:
            slope_star = w * info["slope"] + (1.0 - w) * tier_slopes[info["tier"]]
        else:
            slope_star = tier_slopes[info["tier"]]
        dx = x_target - x_last
        if last <= 0.0:
            anchor = 1.0
            slope_star = TIER_PRIOR_SLOPE.get(info["tier"], 0.6)
        else:
            anchor = max(last / 100.0 * (n_last or 1), 0.0)
        if n_star:
            e_star = anchor * math.exp(slope_star * dx)
            median_rate = 100.0 * e_star / n_star
        else:
            median_rate = last * math.exp(slope_star * dx)
        # Sanity caps on the move from the last read (dilution can pull the
        # displayed rate down, growth can lift it, neither without limit).
        if last > 0:
            median_rate = min(max(median_rate, 0.05 * last), 8.0 * last)
        median_rate = clamp_pct(median_rate)
        sigma_fit = min(max(info["resid"], 0.12), 0.6)
        sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + shock_sigma ** 2)
        sigma = min(max(sigma, 0.15), 1.5)
        logits = mixture_quantiles(logit_pct(median_rate), sigma,
                                   shock_prob, shock_down)
        qs = []
        prev_q = 0.0
        for lo in logits:
            q = round(inv_logit_pct(lo), 4)
            q = min(max(q, 0.01), 100.0)
            q = max(q, prev_q)
            qs.append(q)
            prev_q = q
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    if not names:
        print("no case files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
