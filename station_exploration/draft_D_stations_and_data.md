# Draft D — Station Observations Used for Verification

*Candidate for the main report: the shortest possible data statement — which stations,
what they provide, how they are considered. No figures; one table. Can stand alone or
open whichever of drafts A–C is chosen.*

---

Three station sets verify the selected models against point measurements rather than
gridded analyses. All are MeteoSwiss open data, cover 2020–2024, and entered no training
or model-selection step of this work.

| Set | Stations | Observable | Series | Verifies |
|---|---|---|---|---|
| NBCN | $28$ | daily max 2 m temperature, `ths200dx` | homogenised | \texttt{atm+wind-clip60} |
| SMN temperature | $147$ | daily max 2 m temperature, `tre200dx` | operational | \texttt{atm+wind-clip60} |
| SMN rain gauges | $139$ | daily precipitation 06–06 UTC, `rre150d0` | operational | \texttt{precip-FT-CRPS}, \texttt{precip-SFC-TP} |

All sets are treated identically. The model is decoded directly at the station
coordinates through its final interpolation layer, so no gridded prediction is resampled
and the observations are never interpolated. A station enters its set if it reports on
at least $95\,\%$ of the days in 2020–2024, and is scored only where at least $10$ valid
days remain. The day-labelling of each observable against its gridded counterpart was
verified rather than assumed: every station matches its nearest analysis cell best at
lag $0$ (median correlation $0.98$ against RhiresD, $0.999$ against TmaxD). The
references the models are scored against are bilinear ERA5-Land — lapse-corrected to
station elevation for temperature, the corrected 06–06 accumulation rebuild for
precipitation.

Two limits are structural. The precipitation observable is $\geq 1\,\mathrm{mm}$ wetter
than drizzle, and the gauges feed RhiresD, so precipitation verification is a
point-versus-cell comparison, not an independent data source. And the SMN series are
operational rather than homogenised; at the $28$ sites present in both temperature
sets, per-station errors agree to a median of $0.00\,^{\circ}$C over this span, so the
distinction carries no weight here.

---

*Note for assembly: the second caveat sentence makes the operational/homogenised
distinction dismissible with evidence; the first ("$\geq 1$ mm wetter than drizzle")
refers to the wet-day definition of \sref{subsubsec:meth-skill-precip} and can be
dropped if the chapter follows the methodology chapter closely.*
