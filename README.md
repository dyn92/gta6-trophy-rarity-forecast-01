# GTA VI trophy rarity forecast

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast, for
every Grand Theft Auto VI trophy on PlayStation 5, the percent of Exophase-tracked
players who will hold it at 2026-11-26T00:00:00Z, as five quantiles per trophy
(0.05, 0.25, 0.5, 0.75, 0.95). Scored by pinball loss on the logit scale.

## Program

`main.py` (Python 3.13, standard library only, no network, deterministic) reads
every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Model

- The house's recording is a step function: the page is frozen between Exophase
  refreshes and jumps at each refresh. Trends are fitted on the reads where a
  rate (or the players-tracked count) actually changed.
- Earners and players are forecast separately and divided: log earners per trophy
  is fitted against `ln(1 + hours since launch)` with a recency-weighted least
  squares line, the slope shrunk toward the tier's mean slope
  (platinum/gold/silver/bronze); players tracked is fitted once, globally. The
  median rate at the target is `E*/N*`, anchored on the last read.
- Spread (logit space) combines a random-walk term growing with the horizon,
  the fit residual, and the observed refresh-jump dispersion.
- Flat history (no refresh ever seen, as in the practice cases): the page has
  been frozen for hours, so a refresh is overdue. The forecast is a two-component
  mixture — with probability `p` a refresh lands before the target and dilutes
  every non-100% rate (newly tracked players hold far fewer trophies than the
  launch-night core), with probability `1-p` the page stays put.
- A trophy at 100% stays 100%; a trophy at 0% is anchored on one hypothetical
  earner with its tier's prior slope.

## Practice results (diagnostic only)

The kit holds no practice answers. The answers were reconstructed from the
refresh the kit documents (638 → 3,868 players tracked, 943 → 4,276 trophies
earned at the 04:30Z read; the 100% trophy stays 100%, the other 54 trophies'
counts go 305 → 408, i.e. a ~0.22× proportional dilution). Against that
reconstruction, pinball loss on the logit scale:

| case | this entry | house baseline |
|------|-----------|----------------|
| p00  | 0.213     | 1.405          |
| p01  | 0.095     | 1.405          |

(House baseline ~1.51 per the rules; the small gap is the reconstruction's
proportional assumption vs the true per-trophy variation.)

## Run locally

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```
