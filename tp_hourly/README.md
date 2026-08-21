# ERA5-Land daily precipitation, rebuilt from hourly data

Daily total precipitation (`tp`) for 2020-2024 on the ERA5-Land 0.1° grid,
aggregated here from hourly downloads rather than taken from a CDS daily
product. This note explains why, how, and — the part that actually matters for
interpreting results — **which 24-hour window each file represents**.

Everything below was verified against data in this repository. Where a number is
quoted, the command that produced it is given.

---

## 1. Layout

The daily files live in **two** places, deliberately:

```
tp_hourly/                              <- tracked in git (portable, travels with a clone)
  README.md                             this file
  tp-<year>.nc                          daily, 00-00 UTC window  ("next00", default)
  w0606/tp-<year>.nc                    daily, 06-06 UTC window  ("w0606")

datasets/ERA5_Land/tp_hourly/           <- NOT tracked: datasets/** is gitignored
  raw/tp_hourly-<year>-<month>.nc       hourly downloads, 61 files (~110 MB)
  tp-<year>.nc, w0606/tp-<year>.nc      the same daily files, byte-identical
```

`datasets/` is where the cloud sync puts data and where the hourly `raw/`
monthlies live, so that copy is what `scripts/reeval_precip_w0606.sh` reads
(`TP_DIR`, overridable). But `datasets/**` is gitignored, so a clone would arrive
with no reference at all — hence the tracked copy at the repo root, which is what
the figure scripts default to (`PRECIP_BASELINE_GLOB`). The two are verified
byte-identical; regenerate either with the commands in §9.

Both daily sets are built from the *same* raw hourly files. They differ only in
how the 24 hours are cut into days. Units are **metres**, matching
`datasets/ERA5_Land/precipitation/` — multiply by 1000 for mm/day. Grid is
**29 × 61** (lat 48.2 → 45.4, lon 5.0 → 11.0, 0.1° spacing).

---

## 2. Why the hourly route

The obvious source, `derived-era5-land-daily-statistics`, **cannot serve
precipitation at all**:

```
Daily statistics of accumulated variables are not supported for this dataset,
skipping: total_precipitation.
The job failed with: InvalidRequest
```

That is the CDS response to job `00b3d563` (2026-07-31), and to `5ded18e9`
(2026-07-23) before it. The rejection happens before any time-zone or statistic
setting is considered, so **no `time_zone` value makes that dataset work for
`tp`** — including `utc-01:00`.

`reanalysis-era5-land` (the hourly set) does work: 13 of 13 requests succeeded on
2026-07-23. So the daily field must be assembled locally, which is what
`scripts/build_tp_from_hourly.py` does.

---

## 3. What was downloaded

One request per year-month, via `scripts/build_tp_from_hourly.py --download`:

```
dataset          reanalysis-era5-land
variable         total_precipitation
year / month     2020-01 … 2024-12  (one request each)
day              01 … 31   (all)
time             00:00 … 23:00  (all 24)
area             [48.2, 5.0, 45.4, 11.0]   N, W, S, E  ->  29 × 61
data_format      netcdf
download_format  unarchived
```

Plus **one extra day**: `2025-01-01`, all 24 hours. Day D's total needs the
accumulation at 00:00 of D+1, so without this boundary request 2024-12-31 would
be silently missing. 61 requests in total.

The area must be exactly these four numbers. `reanalysis-era5-land` accepts 0.1°
bounds — all 13 successful jobs on 2026-07-23 used precisely this box. Do **not**
substitute the 0.25° box `[48.0, 5.5, 45.5, 11.0]`; that belongs to the
pressure-level and single-level datasets, and using it on ERA5-Land produced the
26 × 56 file now sitting in `datasets/ERA5_Land/_quarantine_wronggrid/`.

If a web form forces 0.25° steps, snap **outward** to
`[48.25, 5.0, 45.25, 11.0]`, which returns 30 × 61, then trim with
`ds.sel(latitude=slice(48.2, 45.4))` to get back to 29 rows.

---

## 4. How ERA5-Land `tp` behaves

`tp` is an **accumulation**, not a rate. It starts at zero at 00 UTC each day and
grows monotonically until it resets. So the value stamped 14:00 on day D is the
rain that fell between 00:00 and 14:00 on D — and the value stamped **00:00 on
day D+1 is the complete total for day D**.

This is documented behaviour, not an inference from our data. ECMWF's
[ERA5-Land: data documentation](https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation)
states that the convention "differs with that for ERA5", that accumulations are
"accumulated from the beginning of the forecast to the end of the forecast
step", and specifically:

> for the CDS time of 00 UTC, the accumulations are over the 24 hours ending at
> 00 UTC i.e. the accumulation is during the previous day

> The maximum accumulation is over 24 hours, i.e., from day=D, time=0 to
> day=D+1, time=0 (step=24)

That is precisely the `next00` rule below, arrived at independently here by
matching the legacy files (§6.1). Documentation and measurement agree.

Two consequences that cause most of the mistakes in this area:

- You cannot sum the 24 hourly values. That double-counts massively, because
  each one already contains all the preceding hours.
- Taking the **maximum** over a day's own 00:00-23:00 stamps is wrong in a
  subtler way. The 00:00 stamp holds the *previous* day's finished total, so the
  maximum returns `max(total(D-1), partial(D))` — inflated, and biased toward the
  previous day. This is exactly the defect in
  `datasets/ERA5_Land/precipitation/tp-2024.nc` (see §8).

---

## 5. The window rules

| rule | definition | formula |
|---|---|---|
| `next00` | 00 UTC D → 00 UTC D+1 | `acc(00:00 D+1)` |
| `w0606` | 06 UTC D → 06 UTC D+1 | `acc(00:00 D+1) − acc(06:00 D) + acc(06:00 D+1)` |
| `max24` | *(defective)* | `max over D 00:00…23:00` |
| `at23` | *(lossy)* | `acc(23:00 D)` |

`w0606` needs two partial cycles because the accumulation resets in the middle of
the window: the remainder of day D after 06:00, plus the first six hours of
D+1.

`max24` and `at23` exist in the script only so their failure can be demonstrated
rather than asserted. Do not use them.

---

## 6. Validation

### 6.1 `next00` reproduces the legacy files exactly

Against `datasets/ERA5_Land/precipitation/tp-2020.nc`, 274 days,
`--years 2020 --validate`:

```
rule      n_cmp  mean mm/d   legacy   ratio   maxdiff     corr
next00      274     3.6547   3.6547   1.000   0.00000  1.00000
w0606       274     3.6548   3.6547   1.000  62.45948  0.92057
max24       274     5.5420   3.6415   1.522  97.90096  0.79207
at23        274     3.5254   3.6415   0.968  10.59017  0.99881
```

`next00` matches to **0.00000 mm** with correlation **1.00000**. So the legacy
2020-2023 files were built on the 00-00 UTC window, and the rebuild is
bit-compatible with them. This is measured, not inferred.

The other rows quantify the alternatives: `max24` inflates by **1.52×** (1.67× on
the full 2024 file, since the inflation scales with how much rain falls in the
straddled day), and `at23` looks respectable at corr 0.999 while quietly losing
~3% of the annual total by dropping each day's final hour.

### 6.2 RhiresD uses the 06-06 window

Domain-mean daily series, 2020-01-01 to 2020-10-31, 304 days, against
`datasets/MeteoSwiss/RhiresD_v2.0_swiss.lv95/`:

```
window                        mean mm/d  corr vs RhiresD    n
00-00 UTC (next00)               3.9110          0.91204  304
06-06 UTC start D (w0606)        3.9110          0.96443  304     <- best
06-06 UTC end D                  3.9110          0.39435  304
07-07 UTC start D                3.9110          0.96269  304
05-05 UTC start D                3.9110          0.96328  304
RhiresD itself                   3.6518
```

**RhiresD's day D is the 06 UTC D → 06 UTC D+1 window, labelled by its start
date.** The 0.964 against 0.394 for the same window labelled by its *end* date
settles the labelling direction; the 05-05 and 07-07 rows bracket the boundary at
about 06 UTC.

The NetCDF metadata declares no window — only `long_name: daily precipitation
sum` — so this was established empirically first, then confirmed against the
official product documentation, *Documentation of MeteoSwiss Grid-Data Products,
Daily Precipitation (final analysis): RhiresD*
([PDF](https://www.meteoswiss.admin.ch/dam/jcr:4f51f0f1-0fe3-48b5-9de0-15666327e63c/ProdDoc_RhiresD.pdf)),
which defines the variable as:

> Daily precipitation on day D, corresponding to rainfall and snowfall water
> equivalent accumulated from 06:00 UTC of day D to 06:00 UTC of day D+1. In
> millimeters (equivalent to liters per square meter).

Start-labelled, 06-06 UTC — exactly what the correlation table selects.

Note that **all windows share the same mean**. Changing the window does not
create or destroy precipitation; it only reattributes it between adjacent days.
That is the expected signature and a useful sanity check on any future rule.

---

## 7. Which file to use

The window is baked into the **file**. Nothing downstream re-windows it —
`precip_baseline.py` interpolates spatially and reindexes by date, and
`datasets.load_era5_precip_aligned` does floor-to-day plus reindex with no shift.
So the choice is made purely by which path you point at.

**`tp-<year>.nc` (`next00`, 00-00 UTC)** — continuous with the legacy 2020-2023
series, so existing skill numbers stay comparable and only 2024 changes. Use for
the model input channel (`--use-surface-precip`) and wherever continuity with
prior results matters.

**`w0606/tp-<year>.nc` (06-06 UTC)** — aligned to how RhiresD defines a day. Use
as the evaluation baseline when the comparison against the truth should be fair:

```bash
python eval_precip.py --model-dir <RUN>/precip --eval-year 2024 \
  --baseline-glob "tp_hourly/w0606/tp-{year}.nc"

# CV spans four years -- pass a multi-year glob, or the default resolves to
# tp-<DATA_YEAR_START> alone and silently scores ~365 of 1461 days
python eval_precip.py --model-dir <RUN>/precip \
  --baseline-glob "tp_hourly/w0606/tp-*.nc"
```

`precip_baseline.py` does `(glob or DEFAULT_GLOB).format(year=year)`, so the
override takes either a literal glob or a `{year}` placeholder.

### Feature importance is not affected

FI perturbs input channels and scores the model against the truth; the ERA5
reference never enters it, so changing the baseline to `w0606` cannot move an FI
number. `feature_importance/baseline.json` holds the *unperturbed model* score,
not a skill reference. The existing precip FI runs used **2022**, whose inputs
come from the legacy files and are correct.

The one case that would break: running FI with `--year 2024` on the `sfctp`
model, the only run with `tp` as an input. That would read the defective
`datasets/ERA5_Land/precipitation/tp-2024.nc`. Point it at `tp_hourly/` first.

### Skill will fall

**Expect reported skill to fall when switching to `w0606`.** The 00-00 baseline is
handicapped by a six-hour offset against the truth, which flatters
`skill = 1 − MAE_model / MAE_baseline`. The lower number is the more defensible
one, and the change should be stated explicitly in any write-up, since it is not
self-evident from the numbers.

---

## 8. Problems in the existing files that this replaces

**`tp-2024.nc` is inflated ~1.67×.** It reads 6.889 mm/day against an expected
3.3-4.5. CDS job history shows why: the derived-daily request failed at
08:00:58Z on 2026-07-23, and three minutes after the last of twelve fallback
hourly downloads the file was written locally using the `max24` rule. The
aggregation code was ad hoc and is not in this repository.

That file is the reference behind the four precip models' 2024 `skill_mae`, which
are therefore overstated. No trained model is affected — `tp` has only ever been
a reference, never an input, until `--use-surface-precip` was added.

**The legacy files are packaged by accumulation window, not calendar year.**
`tp-2020.nc` spans 2019-12-31 to 2020-12-30, and the set chains seamlessly to
2023-12-30. Since `tp-2024.nc` starts on 2024-01-01, **2023-12-31 exists in no
file**. Individual records are correctly dated — verified against RhiresD at lag
0 — so the fix is to trim to covered days, never to fill.

The rebuilt files are cut on calendar years, so this gap does not arise.

---

## 9. Regenerating

```bash
# 1. download (resume-safe; existing months are skipped)
python scripts/build_tp_from_hourly.py --years 2020-2024 --download

# 2. confirm the rule still reproduces the legacy files
python scripts/build_tp_from_hourly.py --years 2020 --validate

# 3. write both window sets
python scripts/build_tp_from_hourly.py --years 2020-2024 --aggregate --rule next00
python scripts/build_tp_from_hourly.py --years 2020-2024 --aggregate --rule w0606
```

Step 2 is worth keeping. If CDS ever changes its accumulation convention, that
check fails loudly against known-good data instead of quietly shifting every
downstream number.

Roughly 2¼ hours for a full download — CDS runs one job at a time per account, at
about 2.5 minutes per month-request.

---

## 10. Caveats

- **The mean comparison against RhiresD in §6.2 is not a bias estimate.** The
  ERA5 box spans 45.4-48.2 N and 5-11 E, taking in French, Italian and German
  terrain, whereas RhiresD covers Switzerland only. The 3.911 against 3.652 gap
  conflates a genuine ERA5-Land wet bias with that domain difference. The
  correlations are unaffected, since every window shares the same domain.
- **§6.2 covers ten months of 2020**, not the full record. The ranking is
  unambiguous, but the correlation values themselves will move once all five
  years are in.
- **These files are not in `datasets/`** and nothing reads them by default.
  Swapping them in is a deliberate step, and it changes the precip baseline and
  potentially an input channel.
- **Files store metres.** Multiply by 1000 for mm/day.

---

## 11. Sources

- ECMWF, *ERA5-Land: data documentation*, Copernicus Knowledge Base.
  <https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation>
  — accumulation convention; the 00 UTC validity time; step-24 spanning
  day D 00:00 to day D+1 00:00.
- ECMWF, *Conversion table for accumulated variables (total precipitation /
  fluxes)*, Copernicus Knowledge Base.
  <https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790>
- MeteoSwiss, *Documentation of MeteoSwiss Grid-Data Products — Daily
  Precipitation (final analysis): RhiresD*.
  <https://www.meteoswiss.admin.ch/dam/jcr:4f51f0f1-0fe3-48b5-9de0-15666327e63c/ProdDoc_RhiresD.pdf>
  — the 06 UTC to 06 UTC day definition, start-labelled.
- Muñoz Sabater, J. (2019), *ERA5-Land hourly data from 1950 to present*,
  Copernicus Climate Change Service (C3S) Climate Data Store.
  doi:[10.24381/cds.e2161bac](https://doi.org/10.24381/cds.e2161bac)
  — already cited in `report-src/references.bib` as `era5land`.
