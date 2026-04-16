#!/usr/bin/env python3
"""
analyze_dosha.py — Dosha Assessment Data Analysis & Visualization

Reads from dosha_assessment.db and generates:
  1.  summary        — G vector + dosha profile table
  2.  dosha_triangle — Constitutional triangle, composite score
  3.  dosha_strip    — Per-probe dot cloud (spread = constitution)
  4.  icc_chart      — Judge ICC(2,1) reliability per dosha dimension
  5.  sphere         — Guna triangle T/R/S
  6.  vata_3d        — Vata scores per probe & model (3D)
  7.  pitta_3d       — Pitta scores per probe & model (3D)
  8.  kapha_3d       — Kapha scores per probe & model (3D)
  9.  heatmap_S      — Sattva heatmap (probe × model)
  10. heatmap_T      — Tamas heatmap (probe × model)
  11. heatmap_R      — Rajas heatmap (probe × model)
  12. divergence     — Top 15 most discriminating probes
  13. report.html    — All charts combined in one page

Usage:
    python analyze_dosha.py [--db <path>] [--out <dir>]
"""

import os
import sys
import json
import math
import sqlite3
import argparse
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly
except ImportError:
    sys.exit("plotly not installed. pip install plotly --break-system-packages")

# ── Color scheme matching tatsat-ai.com ───────────────────────────────────────
BG        = '#05070e'
BG2       = '#0a0f1c'
GOLD      = '#c9940a'
SAFFRON   = '#e8820c'
BLUE      = '#4a9eff'
TEAL      = '#3abfaa'
LOTUS     = '#d4628a'
TEXT      = '#f0e8d0'
MUTED     = '#a89880'

MODEL_COLORS = [TEAL, SAFFRON, BLUE, GOLD, LOTUS, '#a78bfa', '#34d399', '#f87171',
                '#60a5fa', '#fbbf24', '#e879f9']

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TEXT, family='JetBrains Mono, monospace'),
        title_font=dict(color=GOLD, size=16),
        legend=dict(bgcolor=BG2, bordercolor=GOLD, borderwidth=1),
    )
)

# ── Database ───────────────────────────────────────────────────────────────────

def get_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

def get_models(con) -> list:
    return con.execute(
        "SELECT * FROM Models ORDER BY model_id"
    ).fetchall()

def get_dosha_vectors(con) -> list:
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='DoshaVectors'"
    ).fetchone()
    if not rows:
        return []
    # Check if composite columns exist (migration may not have run yet)
    cols = [r[1] for r in con.execute("PRAGMA table_info(DoshaVectors)").fetchall()]
    has_composite = 'vata_comp_norm' in cols
    has_icc = 'icc_V' in cols
    icc_extra = ", dv.icc_V, dv.icc_P, dv.icc_K" if has_icc else ", NULL as icc_V, NULL as icc_P, NULL as icc_K"
    extra = ", dv.vata_sd, dv.pitta_sd, dv.kapha_sd, dv.vata_comp_norm, dv.pitta_comp_norm, dv.kapha_comp_norm" \
            if has_composite else ", NULL as vata_sd, NULL as pitta_sd, NULL as kapha_sd, NULL as vata_comp_norm, NULL as pitta_comp_norm, NULL as kapha_comp_norm"
    extra += icc_extra
    return con.execute(
        f"SELECT dv.*, m.display_name, m.provider {extra} "
        "FROM DoshaVectors dv JOIN Models m ON m.model_id = dv.model_id "
        "ORDER BY dv.vata_norm DESC"
    ).fetchall()

def get_g_vectors(con) -> list:
    return con.execute(
        "SELECT gv.*, m.display_name, m.provider, m.default_temp "
        "FROM GVectors gv JOIN Models m ON m.model_id = gv.model_id "
        "ORDER BY gv.g_S_norm DESC"
    ).fetchall()

def get_probe_scores(con) -> list:
    """Per-probe per-model average scores."""
    return con.execute("""
        SELECT r.model_id, m.display_name, r.probe_id, p.category,
               AVG(s.g_T) as T, AVG(s.g_R) as R, AVG(s.g_S) as S,
               COUNT(*) as n_runs
        FROM Responses r
        JOIN Models m ON m.model_id = r.model_id
        JOIN Probes p ON p.probe_id = r.probe_id
        JOIN Scores s ON s.response_id = r.response_id
        GROUP BY r.model_id, r.probe_id
        ORDER BY r.model_id, r.probe_id
    """).fetchall()

def get_category_means(con) -> list:
    """Per-model per-category (vata/pitta/kapha) average scores."""
    return con.execute("""
        SELECT r.model_id, m.display_name, p.category,
               AVG(s.g_T) as T, AVG(s.g_R) as R, AVG(s.g_S) as S
        FROM Responses r
        JOIN Models m ON m.model_id = r.model_id
        JOIN Probes p ON p.probe_id = r.probe_id
        JOIN Scores s ON s.response_id = r.response_id
        GROUP BY r.model_id, p.category
        ORDER BY r.model_id, p.category
    """).fetchall()

def make_dosha_triangle(dosha_vectors: list) -> go.Figure:
    """
    Dosha constitutional triangle — Vata/Pitta/Kapha at poles.
    This is the CORRECT dosha space. Tri-doshic balance is center.
    """
    fig = go.Figure()

    # Triangle
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, math.sqrt(3)/2, 0]
    fig.add_trace(go.Scatter(
        x=tri_x, y=tri_y, mode='lines',
        line=dict(color=GOLD, width=1),
        showlegend=False, hoverinfo='skip'
    ))

    # Pole labels
    fig.add_annotation(x=0,   y=-0.07, text='VATA', showarrow=False,
                       font=dict(color=BLUE, size=12, family='JetBrains Mono'), xanchor='center')
    fig.add_annotation(x=1,   y=-0.07, text='PITTA', showarrow=False,
                       font=dict(color=SAFFRON, size=12, family='JetBrains Mono'), xanchor='center')
    fig.add_annotation(x=0.5, y=math.sqrt(3)/2+0.06, text='KAPHA', showarrow=False,
                       font=dict(color=TEAL, size=12, family='JetBrains Mono'), xanchor='center')

    # Tri-doshic balance
    cx = (0 + 1 + 0.5) / 3
    cy = (0 + 0 + math.sqrt(3)/2) / 3
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy], mode='markers',
        marker=dict(symbol='circle-open', size=12, color=MUTED, line=dict(width=1.5)),
        name='Tri-doshic balance',
        hovertemplate='Tri-doshic balance<br>V=P=K (equal constitution)'
    ))

    for i, dv in enumerate(dosha_vectors):
        # Use composite norm (0.5*mean + 0.5*SD) if available, else fall back to mean norm
        v = dv['vata_comp_norm']  if dv['vata_comp_norm']  else dv['vata_norm']
        p = dv['pitta_comp_norm'] if dv['pitta_comp_norm'] else dv['pitta_norm']
        k = dv['kapha_comp_norm'] if dv['kapha_comp_norm'] else dv['kapha_norm']
        total = v + p + k
        if total == 0:
            continue
        # Vata at (0,0), Pitta at (1,0), Kapha at (0.5, sqrt(3)/2)
        x = p/total * 1.0 + k/total * 0.5
        y = k/total * math.sqrt(3)/2

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        name  = dv['display_name']
        dom   = max([('Vata', v), ('Pitta', p), ('Kapha', k)], key=lambda x: x[1])

        # Show both mean and composite in hover for transparency
        v_m = dv['vata_norm'];  p_m = dv['pitta_norm'];  k_m = dv['kapha_norm']
        v_sd = dv['vata_sd'] or 0; p_sd = dv['pitta_sd'] or 0; k_sd = dv['kapha_sd'] or 0

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=15, color=color, line=dict(color=BG, width=1.5)),
            text=[name], textposition='top center',
            textfont=dict(size=9, color=color),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Composite — V={v:.3f}  P={p:.3f}  K={k:.3f}<br>"
                f"Mean only  — V={v_m:.3f}  P={p_m:.3f}  K={k_m:.3f}<br>"
                f"SD         — V={v_sd:.3f}  P={p_sd:.3f}  K={k_sd:.3f}<br>"
                f"Constitution: {dom[0]}-dominant<br>"
                f"Provider: {dv['provider']}<extra></extra>"
            )
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title='Dosha Constitutional Profile — Composite Score (0.5·mean + 0.5·SD)<br>'
              '<sup>Vata (scatter/memory) · Pitta (heat/control) · Kapha (formula/attachment) · '
              'SD within category = constitutional variance signal</sup>',
        xaxis=dict(visible=False, range=[-0.18, 1.18]),
        yaxis=dict(visible=False, scaleanchor='x', range=[-0.15, 1.12]),
        showlegend=True,
        height=950,
        margin=dict(t=80, b=40, l=40, r=40),
    )
    return fig

# ── Spherical Triangle Plot (guna space) ───────────────────────────────────────

def make_sphere_plot(gvectors: list) -> go.Figure:
    """
    Project the positive octant of S² onto a 2D ternary-style triangle.
    Tamas=bottom-left, Rajas=bottom-right, Sattva=top.
    """
    # Ternary coordinates from normalized G vector
    # (g_T, g_R, g_S) already on unit sphere positive octant
    # Use as ternary fractions (they don't sum to 1, so normalize)

    fig = go.Figure()

    # Draw triangle background
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, math.sqrt(3)/2, 0]
    fig.add_trace(go.Scatter(
        x=tri_x, y=tri_y, mode='lines',
        line=dict(color=GOLD, width=1),
        showlegend=False, hoverinfo='skip'
    ))

    # Pole labels
    fig.add_annotation(x=0, y=-0.06, text='TAMAS', showarrow=False,
                       font=dict(color=BLUE, size=11), xanchor='center')
    fig.add_annotation(x=1, y=-0.06, text='RAJAS', showarrow=False,
                       font=dict(color=SAFFRON, size=11), xanchor='center')
    fig.add_annotation(x=0.5, y=math.sqrt(3)/2+0.05, text='SATTVA', showarrow=False,
                       font=dict(color=GOLD, size=11), xanchor='center')

    # Balanced center point
    cx = (0 + 1 + 0.5) / 3
    cy = (0 + 0 + math.sqrt(3)/2) / 3
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy], mode='markers',
        marker=dict(symbol='circle-open', size=10, color=MUTED, line=dict(width=1)),
        name='Tri-doshic balance', hovertemplate='Balanced (1/√3, 1/√3, 1/√3)'
    ))

    # Plot each model
    for i, gv in enumerate(gvectors):
        t = gv['g_T_norm']
        r = gv['g_R_norm']
        s = gv['g_S_norm']
        total = t + r + s
        if total == 0:
            continue

        # Ternary → Cartesian
        # Tamas at (0,0), Rajas at (1,0), Sattva at (0.5, sqrt(3)/2)
        x = r/total * 1.0 + s/total * 0.5
        y = s/total * math.sqrt(3)/2

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        name  = gv['display_name']

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=14, color=color,
                        line=dict(color=BG, width=1.5)),
            text=[name], textposition='top center',
            textfont=dict(size=9, color=color),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"T={t:.3f}  R={r:.3f}  S={s:.3f}<br>"
                f"Provider: {gv['provider']}<br>"
                f"Default temp: {gv['default_temp']}"
            )
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title='G Vector Distribution — Positive Octant of S²',
        xaxis=dict(visible=False, range=[-0.15, 1.15]),
        yaxis=dict(visible=False, scaleanchor='x', range=[-0.15, 1.1]),
        showlegend=True,
        height=950,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig

# ── Probe Heatmap ──────────────────────────────────────────────────────────────

def make_heatmap(probe_scores: list, dimension: str = 'S') -> go.Figure:
    """Heatmap of one guna dimension across all models and probes."""
    dim_map = {'T': 'T (Tamas)', 'R': 'R (Rajas)', 'S': 'S (Sattva)'}
    col_key = dimension

    # Build matrix
    models  = sorted(set(r['display_name'] for r in probe_scores))
    probes  = sorted(set(r['probe_id'] for r in probe_scores),
                     key=lambda p: [int(x) if x.isdigit() else x.lower() for x in __import__('re').split(r'(\d+)', p)])

    data = {m: {} for m in models}
    for row in probe_scores:
        data[row['display_name']][row['probe_id']] = float(row[col_key] or 0)

    z = [[data[m].get(p, None) for p in probes] for m in models]

    # Color scale
    cs = [[0, BG2], [0.25, BLUE], [0.5, MUTED], [0.75, SAFFRON], [1, GOLD]]

    fig = go.Figure(go.Heatmap(
        z=z, x=probes, y=models,
        colorscale=cs,
        zmin=0, zmax=4,
        hovertemplate='<b>%{y}</b><br>Probe: %{x}<br>Score: %{z:.2f}<extra></extra>',
        colorbar=dict(
            title=f'g_{dimension} (0–4)',
            tickfont=dict(color=TEXT),
            title_font=dict(color=TEXT),
            bgcolor=BG2,
        )
    ))

    # Add category separators
    v_count = sum(1 for p in probes if p.startswith('V'))
    p_count = sum(1 for p in probes if p.startswith('P'))
    for x_pos, label in [(v_count - 0.5, 'PITTA ▶'), (v_count + p_count - 0.5, 'KAPHA ▶')]:
        fig.add_vline(x=x_pos, line=dict(color=GOLD, width=1, dash='dot'))

    fig.add_annotation(x=v_count/2 - 0.5, y=-0.8, text='◀ VATA', showarrow=False,
                       font=dict(color=BLUE, size=10), yref='paper')
    fig.add_annotation(x=v_count + p_count/2 - 0.5, y=-0.8, text='◀ PITTA', showarrow=False,
                       font=dict(color=SAFFRON, size=10), yref='paper')
    fig.add_annotation(x=v_count + p_count + (len(probes)-v_count-p_count)/2 - 0.5,
                       y=-0.8, text='◀ KAPHA', showarrow=False,
                       font=dict(color=TEAL, size=10), yref='paper')

    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=f'Probe Scores — g_{dimension} ({dim_map[dimension]})',
        xaxis=dict(tickangle=45, tickfont=dict(size=9, color=TEXT),
                   gridcolor=BG, title='Probe'),
        yaxis=dict(tickfont=dict(size=10, color=TEXT), title='Model'),
        height=max(950, len(models) * 50 + 120),
        margin=dict(t=60, b=100, l=160, r=40),
    )
    return fig

# ── Radar Chart ────────────────────────────────────────────────────────────────

def make_3d_bar_chart(probe_scores: list, category: str,
                      score_col: str, title: str) -> go.Figure:
    """
    3D bar chart for one dosha category.
    X = probe ID, Y = model, Z = dosha score (0-4).
    Each model gets its own color. Bars are rectangular prisms via Mesh3d.
    """
    def probe_sort_key(p):
        """Natural sort: V1 < V2 < V6 < V10, K1a < K1b < K2."""
        import re
        parts = re.split(r'(\d+)', p)
        return [int(x) if x.isdigit() else x.lower() for x in parts]

    cat_probes = sorted(
        {r['probe_id'] for r in probe_scores
         if r['probe_id'].upper().startswith(category[0].upper())},
        key=probe_sort_key
    )
    models = sorted(set(r['display_name'] for r in probe_scores))

    if not cat_probes or not models:
        return None

    # Index maps
    probe_idx = {p: i for i, p in enumerate(cat_probes)}
    model_idx = {m: i for i, m in enumerate(models)}

    # Score lookup
    scores = {}
    for row in probe_scores:
        if row['probe_id'] in probe_idx:
            scores[(row['display_name'], row['probe_id'])] = float(row[score_col] or 0)

    # Bar geometry
    BW = 0.35   # bar half-width in x
    BD = 0.35   # bar half-depth in y

    def bar_mesh(xi, yi, z_top, color):
        """Return x,y,z,i,j,k arrays for one rectangular prism bar."""
        x0, x1 = xi - BW, xi + BW
        y0, y1 = yi - BD, yi + BD
        z0, z1 = 0.0, max(z_top, 0.01)   # minimum visible height

        # 8 vertices
        vx = [x0, x1, x1, x0,  x0, x1, x1, x0]
        vy = [y0, y0, y1, y1,  y0, y0, y1, y1]
        vz = [z0, z0, z0, z0,  z1, z1, z1, z1]

        # 12 triangles (2 per face × 6 faces)
        fi = [0,0, 1,1, 2,2, 3,3, 0,0, 4,4]
        fj = [1,2, 2,5, 3,6, 0,7, 4,5, 5,6]
        fk = [2,3, 5,6, 6,7, 7,4, 5,1, 6,7]

        return go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=fi, j=fj, k=fk,
            color=color,
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            lightposition=dict(x=2, y=-2, z=3),
            showscale=False,
            hovertemplate=f'Score: {z_top:.2f}<extra></extra>',
            showlegend=False,
        )

    fig = go.Figure()

    # Render models back-to-front (painter's algorithm)
    # Camera is at y=-1.8, so higher y-index = further from camera = render first
    render_order = list(reversed(range(len(models))))

    # One legend entry per model (forward order for legend readability)
    for mi, model in enumerate(models):
        color = MODEL_COLORS[mi % len(MODEL_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=8, color=color),
            name=model,
        ))

    # Add bars back-to-front so WebGL depth sorts correctly
    for mi in render_order:
        model = models[mi]
        color = MODEL_COLORS[mi % len(MODEL_COLORS)]
        for pi, probe in enumerate(cat_probes):
            z = scores.get((model, probe), 0)
            fig.add_trace(bar_mesh(pi, mi, z, color))

    # Axis labels
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=title,
        scene=dict(
            bgcolor=BG2,
            xaxis=dict(
                tickvals=list(range(len(cat_probes))),
                ticktext=cat_probes,
                tickfont=dict(size=9, color=TEXT),
                title=dict(text='Probe', font=dict(color=MUTED, size=10)),
                gridcolor=MUTED, showbackground=False,
            ),
            yaxis=dict(
                tickvals=list(range(len(models))),
                ticktext=models,
                tickfont=dict(size=9, color=TEXT),
                title=dict(text='Model', font=dict(color=MUTED, size=10)),
                gridcolor=MUTED, showbackground=False,
            ),
            zaxis=dict(
                range=[0, 4],
                tickvals=[0, 1, 2, 3, 4],
                tickfont=dict(size=9, color=TEXT),
                title=dict(text='Score (0–4)', font=dict(color=MUTED, size=10)),
                gridcolor=MUTED, showbackground=False,
            ),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
        ),
        margin=dict(t=60, b=20, l=20, r=20),
        height=950,
        showlegend=True,
    )
    return fig

def get_dosha_probe_scores(con) -> list:
    """Per-probe per-model average dosha scores (d_V, d_P, d_K)."""
    rows = con.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='DoshaVectors'
    """).fetchone()
    if not rows:
        return []
    return con.execute("""
        SELECT r.model_id, m.display_name, r.probe_id, p.category,
               AVG(s.d_V) as dV, AVG(s.d_P) as dP, AVG(s.d_K) as dK,
               COUNT(*) as n_runs
        FROM Responses r
        JOIN Models m ON m.model_id = r.model_id
        JOIN Probes p ON p.probe_id = r.probe_id
        JOIN Scores s ON s.response_id = r.response_id
        WHERE s.d_V IS NOT NULL OR s.d_P IS NOT NULL OR s.d_K IS NOT NULL
        GROUP BY r.model_id, r.probe_id
        ORDER BY r.model_id, r.probe_id
    """).fetchall()


def _ternary_xy(v, p, k):
    """Convert Vata/Pitta/Kapha fractions to ternary Cartesian (x, y).
    Vata at (0,0), Pitta at (1,0), Kapha at (0.5, sqrt(3)/2)."""
    total = v + p + k
    if total == 0:
        return (0.5, math.sqrt(3)/6)  # center
    v, p, k = v/total, p/total, k/total
    x = p * 1.0 + k * 0.5
    y = k * math.sqrt(3) / 2
    return x, y

def make_dosha_strip(dosha_probe_scores: list) -> go.Figure:
    """
    Strip/box chart showing individual probe scores per model per dosha category.
    X = model, Y = dosha score (0-4), grouped by dosha category.
    The SPREAD of dots within each box is the constitutional reading:
    - Tight cluster = settled constitution (Kapha)
    - Wide spread   = Vata scatter (inconsistent across probes in category)
    - All dots high = strong dosha excess
    """
    if not dosha_probe_scores:
        return None

    CAT_COL = {'vata': (BLUE, 'dV'), 'pitta': (SAFFRON, 'dP'), 'kapha': (TEAL, 'dK')}
    CAT_LAB = {'vata': 'Vata (V probes)', 'pitta': 'Pitta (P probes)', 'kapha': 'Kapha (K probes)'}
    offsets = {'vata': -0.25, 'pitta': 0.0, 'kapha': 0.25}

    fig = go.Figure()

    for cat, (color, col) in CAT_COL.items():
        x_vals, y_vals, text_vals = [], [], []
        for row in dosha_probe_scores:
            if row['category'] != cat:
                continue
            score = row[col]
            if score is None:
                continue
            x_vals.append(row['display_name'])
            y_vals.append(float(score))
            text_vals.append(f"{row['probe_id']}: {float(score):.2f}")

        if not x_vals:
            continue

        fig.add_trace(go.Box(
            x=x_vals,
            y=y_vals,
            name=CAT_LAB[cat],
            marker_color=color,
            boxpoints='all',
            jitter=0.4,
            pointpos=offsets[cat],
            line=dict(color=color, width=1.5),
            fillcolor='rgba(0,0,0,0)',
            hovertemplate='%{text}<extra>' + CAT_LAB[cat] + '</extra>',
            text=text_vals,
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title='Constitutional Spread — Individual Probe Scores per Model<br>'
              '<sup>Box = IQR · Dots = individual probes · '
              'Spread within box = constitutional stability · Wide = Vata scatter</sup>',
        boxmode='group',
        xaxis=dict(tickangle=20, tickfont=dict(size=10, color=TEXT), title='Model'),
        yaxis=dict(range=[-0.2, 4.4], title='Dosha Score (0–4)',
                   tickfont=dict(color=TEXT), gridcolor=MUTED),
        height=950,
        margin=dict(t=80, b=80, l=60, r=40),
    )
    return fig

def make_icc_chart(dosha_vectors: list) -> go.Figure:
    """
    ICC(2,1) reliability chart — one grouped bar per model, three bars per model
    for icc_V, icc_P, icc_K.  Reference lines at 0.75 (good) and 0.50 (moderate).
    Measures how consistently the judge discriminates between probes across
    repeated runs on each constitutional axis.
    """
    if not dosha_vectors:
        return None

    models  = [dv['display_name'] for dv in dosha_vectors]
    icc_v   = [dv['icc_V'] if dv['icc_V'] is not None else 0.0 for dv in dosha_vectors]
    icc_p   = [dv['icc_P'] if dv['icc_P'] is not None else 0.0 for dv in dosha_vectors]
    icc_k   = [dv['icc_K'] if dv['icc_K'] is not None else 0.0 for dv in dosha_vectors]

    # Guard — if all ICC values are None/zero, data hasn't been computed yet
    if all(v == 0.0 for v in icc_v + icc_p + icc_k):
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(name='ICC Vata',  x=models, y=icc_v,
                         marker_color=BLUE,    opacity=0.85,
                         hovertemplate='<b>%{x}</b><br>ICC Vata: %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Bar(name='ICC Pitta', x=models, y=icc_p,
                         marker_color=SAFFRON, opacity=0.85,
                         hovertemplate='<b>%{x}</b><br>ICC Pitta: %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Bar(name='ICC Kapha', x=models, y=icc_k,
                         marker_color=TEAL,    opacity=0.85,
                         hovertemplate='<b>%{x}</b><br>ICC Kapha: %{y:.3f}<extra></extra>'))

    # Reference lines
    for y_val, label, dash in [(0.75, 'Good (0.75)', 'dash'),
                                (0.50, 'Moderate (0.50)', 'dot')]:
        fig.add_hline(y=y_val, line=dict(color=GOLD, width=1, dash=dash),
                      annotation_text=label,
                      annotation_font=dict(color=GOLD, size=9),
                      annotation_position='top right')

    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title='Judge Reliability — ICC(2,1) per Dosha Dimension<br>'
              '<sup>Measures cross-run consistency of judge scoring per constitutional axis · '
              '≥0.75 good · 0.50–0.75 moderate · <0.50 poor</sup>',
        barmode='group',
        xaxis=dict(tickangle=30, tickfont=dict(size=10, color=TEXT)),
        yaxis=dict(title='ICC(2,1)', range=[-0.1, 1.05],
                   tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                   tickfont=dict(color=TEXT), gridcolor=MUTED),
        height=650,
        margin=dict(t=80, b=60, l=60, r=40),
    )
    return fig


    """
    Radar/spider chart showing average Sattva by probe category (Vata/Pitta/Kapha).
    """
    models   = sorted(set(r['display_name'] for r in category_means))
    cats     = ['vata', 'pitta', 'kapha']
    cat_labels = ['Vata Probes', 'Pitta Probes', 'Kapha Probes']

    fig = go.Figure()

    for i, model in enumerate(models):
        model_data = {r['category']: r for r in category_means
                      if r['display_name'] == model}
        s_vals = [float(model_data.get(c, {}).get('S') or 0) for c in cats]
        s_vals += [s_vals[0]]  # close the loop

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=s_vals,
            theta=cat_labels + [cat_labels[0]],
            fill='toself',
            fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba')
                        if color.startswith('rgb') else color + '26',
            line=dict(color=color, width=2),
            name=model,
            hovertemplate='%{theta}: %{r:.2f}<extra>' + model + '</extra>'
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title='Sattva Profile by Probe Category',
        polar=dict(
            bgcolor=BG2,
            radialaxis=dict(
                range=[0, 4], tickvals=[1, 2, 3, 4],
                tickfont=dict(color=MUTED, size=9),
                gridcolor=MUTED, linecolor=MUTED,
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT, size=11),
                gridcolor=MUTED, linecolor=MUTED,
            )
        ),
        height=950,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig

# ── Tamas Radar ────────────────────────────────────────────────────────────────

def make_divergence_chart(probe_scores: list) -> go.Figure:
    """
    Show probes where models diverge most — highest variance across models on S score.
    Useful for identifying the most discriminating probes.
    """
    probes = sorted(set(r['probe_id'] for r in probe_scores),
                    key=lambda p: [int(x) if x.isdigit() else x.lower() for x in __import__('re').split(r'(\d+)', p)])
    models = sorted(set(r['display_name'] for r in probe_scores))

    if len(models) < 2:
        return None

    data = {p: [] for p in probes}
    for row in probe_scores:
        data[row['probe_id']].append(float(row['S'] or 0))

    # Variance per probe
    variances = {}
    for p, vals in data.items():
        if len(vals) > 1:
            mean = sum(vals) / len(vals)
            variances[p] = sum((v - mean)**2 for v in vals) / len(vals)
        else:
            variances[p] = 0

    sorted_probes = sorted(probes, key=lambda p: variances.get(p, 0), reverse=True)
    top_probes = sorted_probes[:15]

    fig = go.Figure()
    for i, model in enumerate(models):
        model_data = {r['probe_id']: float(r['S'] or 0)
                      for r in probe_scores if r['display_name'] == model}
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        fig.add_trace(go.Bar(
            name=model,
            x=top_probes,
            y=[model_data.get(p, 0) for p in top_probes],
            marker_color=color,
            hovertemplate=f'<b>{model}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title='Most Discriminating Probes — Top 15 by Cross-Model Variance (g_S)',
        barmode='group',
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(range=[0, 4.3], title='g_S Score'),
        height=950,
        margin=dict(t=60, b=100, l=60, r=40),
    )
    return fig

# ── Summary Table ──────────────────────────────────────────────────────────────

def make_summary_table(gvectors: list, probe_scores: list,
                       dosha_vectors: list = None) -> go.Figure:
    """Table of G vectors and constitutional profiles."""
    if not gvectors:
        return None

    # Find standout probes (highest T or R) per model
    model_standouts = {}
    for row in probe_scores:
        m = row['display_name']
        if m not in model_standouts:
            model_standouts[m] = {'max_T': 0, 'max_T_probe': '-',
                                   'max_R': 0, 'max_R_probe': '-'}
        if float(row['T'] or 0) > model_standouts[m]['max_T']:
            model_standouts[m]['max_T'] = float(row['T'])
            model_standouts[m]['max_T_probe'] = row['probe_id']
        if float(row['R'] or 0) > model_standouts[m]['max_R']:
            model_standouts[m]['max_R'] = float(row['R'])
            model_standouts[m]['max_R_probe'] = row['probe_id']

    # Build dosha lookup by display_name
    dosha_by_name = {}
    if dosha_vectors:
        for dv in dosha_vectors:
            dosha_by_name[dv['display_name']] = dv

    names    = [gv['display_name'] for gv in gvectors]
    t_norms  = [f"{gv['g_T_norm']:.3f}" for gv in gvectors]
    r_norms  = [f"{gv['g_R_norm']:.3f}" for gv in gvectors]
    s_norms  = [f"{gv['g_S_norm']:.3f}" for gv in gvectors]

    # Dosha columns
    v_norms, p_norms, k_norms, constitutions = [], [], [], []
    for gv in gvectors:
        n = gv['display_name']
        dv = dosha_by_name.get(n)
        if dv:
            v_norms.append(f"{dv['vata_norm']:.3f}")
            p_norms.append(f"{dv['pitta_norm']:.3f}")
            k_norms.append(f"{dv['kapha_norm']:.3f}")
            dom = max([('Vata',  dv['vata_norm']),
                       ('Pitta', dv['pitta_norm']),
                       ('Kapha', dv['kapha_norm'])], key=lambda x: x[1])
            constitutions.append(dom[0])
        else:
            v_norms.append('—')
            p_norms.append('—')
            k_norms.append('—')
            t, r, s = gv['g_T_norm'], gv['g_R_norm'], gv['g_S_norm']
            dom = max([('Kapha/Tamas', t), ('Rajas', r), ('Sattvic', s)],
                      key=lambda x: x[1])
            constitutions.append(dom[0] + ' (guna)')

    standout_t = [f"{model_standouts.get(n, {}).get('max_T_probe', '-')} "
                  f"({model_standouts.get(n, {}).get('max_T', 0):.1f})"
                  for n in names]
    standout_r = [f"{model_standouts.get(n, {}).get('max_R_probe', '-')} "
                  f"({model_standouts.get(n, {}).get('max_R', 0):.1f})"
                  for n in names]

    has_dosha = any(v != '—' for v in v_norms)

    if has_dosha:
        headers = ['Model', 'Constitution', 'Vata', 'Pitta', 'Kapha',
                   'T_norm', 'R_norm', 'S_norm', 'Peak Tamas', 'Peak Rajas']
        values  = [names, constitutions, v_norms, p_norms, k_norms,
                   t_norms, r_norms, s_norms, standout_t, standout_r]
    else:
        headers = ['Model', 'T_norm', 'R_norm', 'S_norm',
                   'Profile', 'Peak Tamas', 'Peak Rajas']
        profiles = []
        for gv in gvectors:
            t, r, s = gv['g_T_norm'], gv['g_R_norm'], gv['g_S_norm']
            dom = max([('Kapha/Tamas', t), ('Rajas', r), ('Sattvic', s)], key=lambda x: x[1])
            profiles.append(dom[0])
        values = [names, t_norms, r_norms, s_norms, profiles, standout_t, standout_r]

    fig = go.Figure(go.Table(
        header=dict(
            values=headers,
            fill_color=BG,
            font=dict(color=GOLD, size=11),
            align='left',
            line_color=MUTED,
        ),
        cells=dict(
            values=values,
            fill_color=[[BG2, BG] * (len(names) // 2 + 1)][:len(names)],
            font=dict(color=TEXT, size=10),
            align='left',
            line_color=BG,
            height=28,
        )
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title='Constitutional Summary — Dosha Profile + Guna Quality',
        height=max(950, len(gvectors) * 35 + 120),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig

# ── HTML Report ────────────────────────────────────────────────────────────────

def build_report(gvectors, dosha_vectors, probe_scores, out_dir: Path, con=None):
    """Build a single self-contained HTML report with all charts."""

    dosha_probe_scores = get_dosha_probe_scores(con) if con else []

    figs = {}
    figs['summary']        = make_summary_table(gvectors, probe_scores, dosha_vectors)
    figs['dosha_triangle'] = make_dosha_triangle(dosha_vectors) if dosha_vectors else None
    figs['dosha_strip']    = make_dosha_strip(dosha_probe_scores) \
                             if dosha_probe_scores else None
    figs['icc_chart']      = make_icc_chart(dosha_vectors) if dosha_vectors else None
    figs['sphere']         = make_sphere_plot(gvectors)
    figs['vata_3d']        = make_3d_bar_chart(dosha_probe_scores, 'vata',  'dV',
                                'Vata Constitutional Score by Probe & Model<br>'
                                '<sup>d_V: scatter · poor memory · incoherence · anxiety</sup>') \
                             if dosha_probe_scores else None
    figs['pitta_3d']       = make_3d_bar_chart(dosha_probe_scores, 'pitta', 'dP',
                                'Pitta Constitutional Score by Probe & Model<br>'
                                '<sup>d_P: heat · control · over-assertion · manipulation</sup>') \
                             if dosha_probe_scores else None
    figs['kapha_3d']       = make_3d_bar_chart(dosha_probe_scores, 'kapha', 'dK',
                                'Kapha Constitutional Score by Probe & Model<br>'
                                '<sup>d_K: formula · attachment · sycophancy · inertia</sup>') \
                             if dosha_probe_scores else None
    figs['heatmap_S']      = make_heatmap(probe_scores, 'S')
    figs['heatmap_T']      = make_heatmap(probe_scores, 'T')
    figs['heatmap_R']      = make_heatmap(probe_scores, 'R')
    figs['divergence']     = make_divergence_chart(probe_scores)

    # Save individual files
    for name, fig in figs.items():
        if fig is None:
            continue
        path = out_dir / f"{name}.html"
        fig.write_html(str(path), include_plotlyjs='cdn', full_html=True)
        print(f"  Wrote {path}")

    # Build combined report
    model_count = len(gvectors)
    probe_count = len(set(r['probe_id'] for r in probe_scores))
    total_responses = sum(r['n_runs'] for r in probe_scores)

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dosha Assessment Results — Tat Sat AI</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;900&family=Crimson+Pro:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:#05070e; --bg2:#0a0f1c; --gold:#c9940a; --saffron:#e8820c;
  --blue:#4a9eff; --teal:#3abfaa; --text:#f0e8d0; --muted:#a89880;
  --border:rgba(201,148,10,0.2);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text);
       font-family: 'Crimson Pro', serif; font-size: 18px; }}
header {{ padding: 48px 64px 32px;
          border-bottom: 1px solid var(--border); }}
header h1 {{ font-family: 'Cinzel', serif; font-size: 36px; color: var(--gold);
              margin-bottom: 8px; }}
header p {{ color: var(--muted); font-size: 15px; }}
.stats {{ display: flex; gap: 32px; margin-top: 20px; flex-wrap: wrap; }}
.stat {{ background: var(--bg2); border: 1px solid var(--border);
         padding: 14px 20px; }}
.stat-n {{ font-family: 'Cinzel', serif; font-size: 28px; color: var(--saffron); }}
.stat-l {{ font-family: 'JetBrains Mono', monospace; font-size: 9px;
           letter-spacing: .15em; text-transform: uppercase; color: var(--muted); }}
.section {{ padding: 48px 64px; border-bottom: 1px solid var(--border); }}
.section h2 {{ font-family: 'Cinzel', serif; font-size: 22px; color: var(--text);
               margin-bottom: 8px; }}
.section p {{ color: var(--muted); font-size: 16px; margin-bottom: 20px;
              max-width: 72ch; }}
.chart-wrap {{ background: var(--bg2); border: 1px solid var(--border);
               padding: 8px; margin-top: 16px; }}
footer {{ padding: 28px 64px; color: var(--muted);
          font-family: 'JetBrains Mono', monospace; font-size: 9px; }}
</style>
</head>
<body>
<header>
  <h1>Dosha Assessment Results</h1>
  <p>LLM Constitutional Study · Tat Sat AI · Madhusudana das</p>
  <div class="stats">
    <div class="stat"><div class="stat-n">{model_count}</div><div class="stat-l">Models Assessed</div></div>
    <div class="stat"><div class="stat-n">{probe_count}</div><div class="stat-l">Probes per Model</div></div>
    <div class="stat"><div class="stat-n">{total_responses}</div><div class="stat-l">Scored Responses</div></div>
  </div>
</header>
"""]

    sections = [
        ('summary',        'G Vector Summary (Guna Quality)',
         'Guna-based quality scores: Tamas (inertia/attachment), Rajas (scatter/heat), Sattva (clarity/balance). '
         'These measure response quality, not constitutional type.'),
        ('dosha_triangle', 'Dosha Constitutional Profile — Mean Vectors',
         'Mean dosha vector per model. Note: centroid position alone is insufficient — '
         'a model with high variance across probes may appear centered without being balanced.'),
        ('dosha_strip',   'Constitutional Cloud — Individual Probe Scores',
         'Every probe score plotted as a small dot; mean as a larger dot. '
         'The SPREAD of the cloud is the constitutional reading. '
         'Wide scatter = Vata. Tight cluster near a pole = settled constitution. '
         'Two models with the same centroid can have entirely different constitutions.'),
        ('cv_chart',      'Constitutional Variance — Coefficient of Variation',
         'CV = σ/μ per dosha dimension. High CV = scattered character (Vata signature). '
         'Low CV = settled constitution (Kapha or directed Pitta). '
         'The stability score (right axis) summarizes constitutional groundedness: '
         'near 1 = Kapha-stable, near 0 = Vata-scattered.'),
        ('vata_3d',        'Vata Scores — 3D by Probe & Model',
         'Vata constitutional excess per probe. High bars = scatter, poor memory, '
         'incoherence, or anxiety. Rotate to compare models across probes.'),
        ('pitta_3d',       'Pitta Scores — 3D by Probe & Model',
         'Pitta constitutional excess per probe. High bars = heat, control, '
         'over-assertion, or adversarial behavior.'),
        ('kapha_3d',       'Kapha Scores — 3D by Probe & Model',
         'Kapha constitutional excess per probe. High bars = formula repetition, '
         'sycophancy, attachment to context, or quality degradation.'),
        ('sphere',         'Guna Space Triangle (T/R/S)',
         'Tamas/Rajas/Sattva as qualities of individual responses. '
         'Note: this is guna space, not dosha space — tri-doshic does NOT mean T=R=S.'),
        ('heatmap_S', 'Sattva Heatmap (g_S)',
         'Per-probe Sattva scores across all models. '
         'Brighter = more sattvic response. Dark cells indicate probes '
         'where the model failed to produce clear, balanced behavior.'),
        ('heatmap_T', 'Tamas Heatmap (g_T)',
         'Per-probe Tamas scores. Bright cells indicate inertia, attachment, '
         'formula repetition, sycophancy, or hallucination.'),
        ('heatmap_R', 'Rajas Heatmap (g_R)',
         'Per-probe Rajas scores. Bright cells indicate scatter, over-assertion, '
         'anxiety, manipulation, or loss of coherence.'),
        ('h2h',       'Head-to-Head: Sattva by Probe',
         'Grouped bar chart comparing g_S across all models per probe. '
         'Probes where bars diverge significantly are the most constitutionally discriminating.'),
        ('divergence', 'Most Discriminating Probes',
         'Top 15 probes ranked by cross-model variance in g_S score. '
         'These are the probes that most reliably distinguish one model\'s '
         'constitutional character from another.'),
    ]

    for name, title, desc in sections:
        if figs.get(name) is None:
            continue
        div_html = plotly.offline.plot(
            figs[name], output_type='div',
            include_plotlyjs=False, config={'displayModeBar': True}
        )
        html_parts.append(f"""
<section class="section">
  <h2>{title}</h2>
  <p>{desc}</p>
  <div class="chart-wrap">{div_html}</div>
</section>
""")

    html_parts.append("""
<footer>Tat Sat AI · tatsat-ai.com · Study design: Madhusudana das · April 2026</footer>
</body></html>""")

    report_path = out_dir / 'report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"  Wrote {report_path}")
    return report_path

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Dosha Assessment Analysis')
    ap.add_argument('--db',  default='./dosha_assessment.db')
    ap.add_argument('--out', default='./dosha_results')
    args = ap.parse_args()

    db_path  = Path(args.db)
    out_dir  = Path(args.out)

    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    con = get_db(db_path)

    gvectors     = get_g_vectors(con)
    dosha_vectors = get_dosha_vectors(con)
    probe_scores = get_probe_scores(con)

    if not gvectors:
        print("No G vectors computed yet. Run assess_dosha.py --vectors-only first.")
        return

    print(f"\nModels with G vectors: {len(gvectors)}")
    for gv in gvectors:
        print(f"  {gv['display_name']:<30} T={gv['g_T_norm']:.3f} "
              f"R={gv['g_R_norm']:.3f} S={gv['g_S_norm']:.3f}")

    if dosha_vectors:
        print(f"\nModels with Dosha vectors: {len(dosha_vectors)}")
        for dv in dosha_vectors:
            print(f"  {dv['display_name']:<30} V={dv['vata_norm']:.3f} "
                  f"P={dv['pitta_norm']:.3f} K={dv['kapha_norm']:.3f}")
    else:
        print("\nNo Dosha vectors yet. Run: python assess_dosha.py --dosha-score")

    print(f"\nGenerating visualizations → {out_dir}/")
    report_path = build_report(gvectors, dosha_vectors, probe_scores, out_dir, con)

    print(f"\nDone. Open {report_path} in your browser.")
    con.close()

if __name__ == '__main__':
    main()
