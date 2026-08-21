#!/usr/bin/env python3
"""Per-component cost statistics of the ConvCNP, for the report.

Counts parameters and multiply-accumulate operations per component, and measures
the peak memory of a training step, for both input grids (surface ERA5-Land
0.1 deg, atmospheric ERA5 0.25 deg native) and both output distributions
(Gaussian, Bernoulli-Gamma). Operation counts are analytic and cross-checked
against torch's FlopCounterMode; the memory total is the measured peak
allocation, split into the tensors whose size is known exactly plus a remainder.

Run from the repository root; writes
LATEX_REPORT/images/component_cost_shares.png.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from params import Params
import model_factory
from torch.utils.flop_counter import FlopCounterMode

P, N = 88800, 128
BATCH = 8          # batch size of the clean runs; the memory profile is measured at it
CHUNK = 16384                                   # ParamLayer.chunk_size
GRIDS = {'surface': (6, 29, 61), 'atmospheric': (95, 11, 23)}
DIST = {'gaussian': (2, 7), 'bernoulli_gamma': (3, 8)}   # n_params, elev-MLP inputs
MB = 1 / 1e6   # decimal megabytes, matching the units used in the report

OPS = ['Encoder', 'CNN', 'MLP (grid to params)', 'RBF final layer', 'Elevation MLP']
MEM = ['Distance matrix (resident)', 'RBF kernel, transient',
       'Elevation MLP, transient', 'Other live tensors']
results = {}

# ---- analytic operation counts -----------------------------------------
for gname, (C, H, W) in GRIDS.items():
    cells = H * W
    for dname, (NP, EI) in DIST.items():
        ops = [
            2 * C * 25 * cells + cells * (2 * C) * N,               # encoder
            cells * 6 * (2 * N * 25 + 2 * N * N),                   # CNN
            cells * (N * 64 + 4 * 64 * 64 + 64 * NP),               # MLP grid->par
            NP * P * cells,                                         # RBF final layer
            P * (EI * 64 + 4 * 64 * 64 + 64 * NP),                  # elevation MLP
        ]
        results[(gname, dname)] = dict(ops=ops, ops_tot=sum(ops),
                                       exp=NP * P * cells, cells=cells)

base = results[('surface', 'gaussian')]['ops_tot']

# ---- parameters, flop cross-check, measured training-step memory --------
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
for gname, (C, H, W) in GRIDS.items():
    cells = H * W
    for dname, (NP, EI) in DIST.items():
        r = results[(gname, dname)]
        var = 'tmax' if dname == 'gaussian' else 'precip'
        m = model_factory.build_model(Params(VARIABLE=var).with_in_channels(C))
        m = m[0] if isinstance(m, tuple) else m
        fin = 'final_layer' if hasattr(m, 'final_layer') else 'out_layer'
        r['params'] = {s: sum(q.numel() for q in getattr(m, s).parameters())
                       for s in ['encoder', 'decoder', 'mlp', fin, 'elev_mlp']}
        r['params_tot'] = sum(q.numel() for q in m.parameters())

        m = m.to(dev)
        x = torch.randn(BATCH, C, H, W, device=dev)
        mask = torch.ones_like(x)
        dists = torch.rand(P, H, W, device=dev) * 0.5
        el = torch.randn(P, 3, device=dev)
        seas = torch.randn(BATCH, 2, device=dev)

        m.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x, mask, dists, el, seas)
        r['flops_mac'] = fc.get_total_flops() / 2 / BATCH   # per day
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            m(x, mask, dists, el, seas)
        torch.cuda.synchronize()
        r['mem_infer'] = torch.cuda.max_memory_allocated()

        # stage-by-stage profile of a training forward pass: the in-stage peak
        # and the memory still live when the stage ends. The overall peak is a
        # moment inside one stage, not a sum over stages.
        m.train()
        stages, live, peaks = [], [], []

        def step(label, fn):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            out = fn()
            torch.cuda.synchronize()
            stages.append(label)
            peaks.append(torch.cuda.max_memory_allocated())
            live.append(torch.cuda.memory_allocated())
            return out

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stages.append('inputs')
        peaks.append(torch.cuda.memory_allocated())
        live.append(torch.cuda.memory_allocated())
        h = step('encoder', lambda: torch.relu(m.encoder(x, mask)))
        h2 = step('CNN', lambda: torch.relu(m.decoder(h)))
        h3 = step('MLP', lambda: m.mlp(h2))
        final = getattr(m, 'final_layer', None) or m.out_layer
        o = step('final layer', lambda: final(h3, dists))
        if isinstance(o, tuple):
            cat = torch.cat([q.reshape(BATCH, P, -1) for q in o], dim=2)
        else:
            cat = o
        cat = torch.cat([cat, el.repeat(BATCH, 1, 1),
                         seas.unsqueeze(1).expand(-1, P, -1)], dim=2)
        out = step('elevation MLP', lambda: m.elev_mlp(cat))
        peak = max(peaks)
        ipk = peaks.index(peak)
        r['stages'], r['live'], r['peaks'] = stages, live, peaks
        r['peak_stage'] = stages[ipk]
        out.square().mean().backward()
        torch.cuda.synchronize()
        r['mem_step'] = torch.cuda.max_memory_allocated()
        m.zero_grad(set_to_none=True)

        # composition AT the peak: what is live entering the peak stage, plus the
        # transient that stage adds. Categories are fixed so the panels share a
        # legend; the one that is not live at the peak is zero.
        known_dists = 4 * P * cells
        live_before = live[ipk - 1] if ipk > 0 else peak
        transient = max(peak - live_before, 0)
        other = max(live_before - known_dists, 0)
        kernel_t = transient if r['peak_stage'] == 'final layer' else 0
        elev_t = transient if r['peak_stage'] == 'elevation MLP' else 0
        r['mem'] = [known_dists, kernel_t, elev_t, other]
        r['mem_tot'] = peak
        print(f'{gname}/{dname}: params {r["params_tot"]:,} {r["params"]} | '
              f'MAC analytic {r["ops_tot"]/1e6:.1f}M vs counter {r["flops_mac"]/1e6:.1f}M | '
              f'B={BATCH}: fwd peak {peak*MB:.0f} MB in the {r["peak_stage"]}, '
              f'full step {r["mem_step"]*MB:.0f}, inference {r["mem_infer"]*MB:.0f} '
              f'| dists {known_dists*MB:.0f} + transient {transient*MB:.0f} '
              f'+ other live {other*MB:.0f}')
        del m, x, mask, dists, el, seas, out
        if dev == 'cuda':
            torch.cuda.empty_cache()

mem_base = results[('surface', 'gaussian')]['mem_tot']
for k, r in results.items():
    print(f'{str(k):36s} ops {r["ops_tot"]/1e6:8.1f}M ({r["ops_tot"]/base:.2f}x) '
          f'mem {r["mem_tot"]*MB:7.1f} MB ({r["mem_tot"]/mem_base:.2f}x)')

# ---- figure -------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
GRID_INK, INK, MUTED = '#c9c8c0', '#1a1a19', '#5c5b55'
COL = [(g, d) for g in GRIDS for d in DIST]
HEAD = {'gaussian': 'Gaussian', 'bernoulli_gamma': 'Bernoulli–Gamma'}


def ring(ax, radius=1.0):
    """Unlabelled reference circle at full scale: shows how far a disc falls short."""
    ax.add_artist(plt.Circle((0, 0), radius, fill=False, ec=GRID_INK,
                             lw=0.7, ls=(0, (3, 3)), zorder=0))


def draw(ax, vals, total, vmax, colors, floor):
    radius = (total / vmax) ** 0.5
    wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                       radius=radius, wedgeprops=dict(edgecolor='white', lw=1.4))
    for w, v in zip(wedges, vals):
        pct = 100 * v / total
        if pct < floor:
            continue
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        ax.text(0.62 * radius * np.cos(ang), 0.62 * radius * np.sin(ang),
                f'{pct:.0f}%', ha='center', va='center', color='white',
                fontsize=11, fontweight='bold', zorder=3)
    return wedges


def panel_figure(labels, colors, values_of, total_of, unit, fname, ncol):
    """One figure: the two grids side by side, areas true to the totals.

    Type is sized for a figure placed at about half the text width, so the labels
    still read after LaTeX scales the image down.
    """
    cols = [(g, 'gaussian') for g in GRIDS]
    vmax = max(total_of(results[k]) for k in cols)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.5))
    for ax, key in zip(axes, cols):
        r = results[key]
        tot = total_of(r)
        ring(ax)
        w = draw(ax, values_of(r), tot, vmax, colors, 8)
        ax.set_title(f'{key[0].capitalize()} grid\n{unit(tot)}\n'
                     f'{tot / total_of(results[cols[0]]):.2f}$\\times$ baseline',
                     fontsize=12, color=INK, pad=5)
        ax.set(aspect='equal', xlim=(-1.12, 1.12), ylim=(-1.12, 1.12))
        ax.axis('off')
    fig.legend(w, labels, loc='lower center', ncol=ncol, frameon=False, fontsize=11,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.01), handlelength=1.0,
               handleheight=1.0, columnspacing=1.2, labelspacing=0.35)
    fig.subplots_adjust(top=0.80, bottom=0.26, left=0.01, right=0.99, wspace=0.02)
    fig.savefig(fname, dpi=220, facecolor='white')
    print('wrote', fname)


panel_figure(OPS, HUES,
             lambda r: r['ops'], lambda r: r['ops_tot'],
             lambda v: f'{v / 1e6:.0f} MMAC/day',
             'LATEX_REPORT/images/component_cost_ops.png', ncol=2)

# ---- alternative memory view: profile of a training forward pass -----------
def profile_figure(fname):
    # the Bernoulli-Gamma head is the heavier of the two on either grid, so the
    # profile shows the worst case; the legend states the margin over the Gaussian
    cols = [(g, 'bernoulli_gamma') for g in GRIDS]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), sharey=True)
    ymax = max(max(results[k]['peaks']) for k in cols) * MB
    for ax, key in zip(axes, cols):
        r = results[key]
        live = np.array(r['live']) * MB
        peaks = np.array(r['peaks']) * MB
        xs = np.arange(len(live))
        dists_mb = 4 * P * r['cells'] * MB
        ax.fill_between(xs, 0, live, step='post', color=HUES[0], alpha=0.20, lw=0)
        ax.step(xs, live, where='post', color=HUES[0], lw=1.8)
        for xi, (lv, pk) in enumerate(zip(live, peaks)):
            if pk - lv > 0.5:                      # transient above what survives
                ax.vlines(xi, lv, pk, color=HUES[1], lw=6, alpha=0.85)
        ax.axhline(dists_mb, color=MUTED, lw=0.9, ls=(0, (3, 3)))
        ax.text(len(xs) - 0.65, dists_mb, 'distance matrix ', va='top', ha='right',
                fontsize=8, color=MUTED)
        pk = peaks.max()
        ipk = int(peaks.argmax())
        ax.plot([ipk], [pk], 'o', color=HUES[1], ms=6)
        right = ipk >= len(xs) - 2
        ax.annotate(f'peak {pk:.0f} MB', (ipk, pk), textcoords='offset points',
                    xytext=(-8 if right else 8, 6), fontsize=9, color=INK,
                    ha='right' if right else 'left')
        ax.set_title(f'{key[0].capitalize()} grid', fontsize=11, color=INK)
        ax.set_xticks(xs, r['stages'], rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, ymax * 1.20)
        ax.set_xlim(-0.4, len(xs) - 0.6)
        if ax is axes[0]:
            ax.set_ylabel('MB live', fontsize=9)
        ax.tick_params(labelsize=8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=HUES[0], lw=2),
                Line2D([], [], color=HUES[1], lw=5, alpha=0.85)],
               ['memory still live at the end of the stage',
                'transient peak inside the stage'],
               loc='lower center', ncol=2, frameon=False, fontsize=8.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.01, 'Bernoulli–Gamma head shown, the heavier of the two: '
             'the Gaussian head peaks 1–3 % lower',
             ha='center', va='bottom', fontsize=8.5, color=MUTED)
    fig.subplots_adjust(bottom=0.40, top=0.88, left=0.10, right=0.98, wspace=0.30)
    fig.savefig(fname, dpi=220, facecolor='white')
    print('wrote', fname)


profile_figure('LATEX_REPORT/images/component_cost_mem_profile.png')
