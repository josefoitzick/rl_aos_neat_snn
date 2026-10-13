#!/usr/bin/env python3
"""Comparación estadística de configuraciones según la propuesta, §3.4.

Uso (entorno conda `tesis-analisis`, ver analysis/environment.yml):
    python analysis/compare.py ~/tesis/runs/lb_pendulum --adaptive ap --baselines static \\
        --threshold-from ~/tesis/runs/calib_replace --threshold-frac 0.95

Por par (adaptativa vs línea base), con las réplicas de un mismo entorno:
  - Eficacia:     fitness de test final del mejor genoma → Mann-Whitney U + Â12
  - Complejidad:  neuronas ocultas y conexiones habilitadas del mejor genoma → Mann-Whitney U + Â12
  - Estabilidad:  dispersión del fitness de test final → Brown-Forsythe (Levene con mediana)
  - Eficiencia:   generaciones hasta que el test del campeón alcanza el umbral → log-rank
                  (las corridas que no lo alcanzan quedan censuradas, no se descartan)
Todos los p-valores del entorno se corrigen con Holm (α = 0,05).

Umbral (fijado ANTES de las réplicas finales): --threshold T absoluto, o normalizado desde corridas
preliminares (--threshold-from DIR --threshold-frac q):
    umbral = f_ref + q·(f_best − f_ref)
con f_ref = mediana del test del campeón en la generación 0 y f_best = mejor test de campeón observado.
Esta forma evita el problema del "95 % del mejor" con recompensas negativas (docs/diseno_neat_ap.md §9.3).

Salida en <experimento>/analysis/: report.md, runs.csv y figuras PNG.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats

OPS = ["mutate_weights", "add_connection", "add_node"]
ALPHA = 0.05


# ───────────────────────────── carga ─────────────────────────────

def read_summary(path):
    d = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def load_experiment(root):
    runs, gens = [], []
    for vdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "analysis"):
        for sdir in sorted(vdir.glob("seed_*")):
            if not (sdir / "summary.txt").exists():
                continue
            s = read_summary(sdir / "summary.txt")
            seed = int(sdir.name.split("_")[1])
            runs.append({"variant": vdir.name, "seed": seed, "final_test": float(s["final_test"]),
                         "best_train": float(s["best_train"]), "hidden": int(s["best_hidden"]),
                         "conns": int(s["best_conns"])})
            g = pd.read_csv(sdir / "generations.csv")
            g["variant"], g["seed"] = vdir.name, seed
            gens.append(g)
    return pd.DataFrame(runs), pd.concat(gens, ignore_index=True)


# ───────────────────────────── estadística ─────────────────────────────

def a12(x, y):
    """Vargha-Delaney Â12: P(X > Y) + 0,5·P(X = Y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m, n = len(x), len(y)
    ranks = stats.rankdata(np.concatenate([x, y]))
    return (ranks[:m].sum() / m - (m + 1) / 2) / n


def a12_magnitude(a):
    d = abs(a - 0.5)
    return "despreciable" if d < 0.06 else "pequeño" if d < 0.14 else "mediano" if d < 0.21 else "grande"


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def time_to_threshold(gdf, thr):
    """Generaciones hasta que champion_test >= thr (evento) o última generación (censura)."""
    out = {}
    for seed, g in gdf.groupby("seed"):
        g = g.sort_values("gen")
        hit = g[g["champion_test"] >= thr]
        out[seed] = (int(hit["gen"].iloc[0]), 1) if len(hit) else (int(g["gen"].iloc[-1]), 0)
    return out


def threshold_from(pilot_dir, frac):
    _, g = load_experiment(Path(pilot_dir).expanduser())
    f_ref = g[g["gen"] == 0]["champion_test"].median()
    f_best = g["champion_test"].max()
    return f_ref + frac * (f_best - f_ref), f_ref, f_best


# ───────────────────────────── figuras ─────────────────────────────

def plot_curves(gens, variants, thr, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variants:
        q = gens[gens["variant"] == v].groupby("gen")["champion_test"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.plot(q.index, q[0.5], label=f"{v} (mediana)")
        ax.fill_between(q.index, q[0.25], q[0.75], alpha=0.2)
    if thr is not None:
        ax.axhline(thr, ls="--", c="gray", lw=1, label=f"umbral {thr:.1f}")
    ax.set_xlabel("generación"); ax.set_ylabel("fitness de test del campeón"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_box(runs, variants, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(axes, ["final_test", "hidden", "conns"],
                              ["test final (mejor genoma)", "neuronas ocultas", "conexiones habilitadas"]):
        data = [runs[runs["variant"] == v][col] for v in variants]
        ax.boxplot(data, tick_labels=variants)
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i) + np.random.uniform(-0.08, 0.08, len(d)), d, s=12, alpha=0.6)
        ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_survival(tte, variants, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        d = tte[v]
        KaplanMeierFitter().fit([t for t, _ in d.values()], [e for _, e in d.values()], label=v) \
            .plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlabel("generación"); ax.set_ylabel("P(no alcanzó el umbral)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_operators(gens, variants, path):
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), squeeze=False)
    for ax, v in zip(axes[0], variants):
        m = gens[gens["variant"] == v].groupby("gen")[[f"p_{o}" for o in OPS]].mean()
        for o in OPS:
            ax.plot(m.index, m[f"p_{o}"], label=o)
        ax.set_title(f"P medio por generación — {v}"); ax.set_xlabel("generación"); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ───────────────────────────── principal ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--adaptive", nargs="+", required=True, help="variante(s) a evaluar")
    ap.add_argument("--baselines", nargs="+", required=True, help="línea(s) base")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--threshold-from", help="corridas preliminares para el umbral normalizado")
    ap.add_argument("--threshold-frac", type=float, default=0.95)
    args = ap.parse_args()

    root = Path(args.experiment).expanduser()
    runs, gens = load_experiment(root)
    variants = list(dict.fromkeys(args.adaptive + args.baselines))
    runs, gens = runs[runs["variant"].isin(variants)], gens[gens["variant"].isin(variants)]
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)

    thr, thr_note = args.threshold, "absoluto"
    if thr is None and args.threshold_from:
        thr, f_ref, f_best = threshold_from(args.threshold_from, args.threshold_frac)
        thr_note = f"f_ref {f_ref:.1f} + {args.threshold_frac}·(f_best {f_best:.1f} − f_ref)"

    tte = {v: time_to_threshold(gens[gens["variant"] == v], thr) for v in variants} if thr is not None else None

    # Todas las comparaciones del entorno, luego Holm
    rows = []
    for a in args.adaptive:
        for b in args.baselines:
            if a == b:
                continue
            A, B = runs[runs["variant"] == a], runs[runs["variant"] == b]
            for metric, col in [("eficacia: test final", "final_test"), ("complejidad: ocultas", "hidden"),
                                ("complejidad: conexiones", "conns")]:
                p = stats.mannwhitneyu(A[col], B[col], alternative="two-sided").pvalue
                rows.append([a, b, metric, "Mann-Whitney U", A[col].median(), B[col].median(), p, a12(A[col], B[col])])
            p = stats.levene(A["final_test"], B["final_test"], center="median").pvalue
            rows.append([a, b, "estabilidad: dispersión test final", "Brown-Forsythe",
                         A["final_test"].std(), B["final_test"].std(), p, np.nan])
            if tte is not None:
                ta, tb = tte[a], tte[b]
                r = logrank_test([t for t, _ in ta.values()], [t for t, _ in tb.values()],
                                 event_observed_A=[e for _, e in ta.values()],
                                 event_observed_B=[e for _, e in tb.values()])
                rows.append([a, b, "eficiencia: generaciones a umbral", "log-rank",
                             sum(e for _, e in ta.values()), sum(e for _, e in tb.values()), r.p_value, np.nan])
    res = pd.DataFrame(rows, columns=["adaptativa", "base", "métrica", "test", "valor A", "valor B", "p", "A12"])
    res["p_holm"] = holm(res["p"]) if len(res) else []
    res["signif"] = res["p_holm"] < ALPHA

    plot_curves(gens, variants, thr, out / "convergencia.png")
    plot_box(runs, variants, out / "cajas.png")
    plot_operators(gens, variants, out / "operadores.png")
    if tte is not None:
        plot_survival(tte, variants, out / "supervivencia.png")

    # Reporte
    lines = [f"# Comparación — {root.name}", ""]
    lines.append("| variante | n | test final (mediana) | media ± de | ocultas (mediana) | conexiones (mediana) |")
    lines.append("|---|---|---|---|---|---|")
    for v in variants:
        r = runs[runs["variant"] == v]
        lines.append(f"| {v} | {len(r)} | {r['final_test'].median():.1f} | {r['final_test'].mean():.1f} ± "
                     f"{r['final_test'].std():.1f} | {r['hidden'].median():.0f} | {r['conns'].median():.0f} |")
    lines += ["", f"Umbral de eficiencia: {thr if thr is None else f'{thr:.2f}'} ({thr_note})."]
    if tte is not None:
        for v in variants:
            hits = [t for t, e in tte[v].values() if e]
            lines.append(f"- {v}: {len(hits)}/{len(tte[v])} alcanzan el umbral"
                         + (f"; mediana {np.median(hits):.0f} generaciones" if hits else ""))
    lines += ["", "| adaptativa | base | métrica | test | valor A | valor B | p | p Holm | Â12 | ¿signif.? |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        a12s = "—" if np.isnan(r["A12"]) else f"{r['A12']:.2f} ({a12_magnitude(r['A12'])})"
        lines.append(f"| {r['adaptativa']} | {r['base']} | {r['métrica']} | {r['test']} | {r['valor A']:.2f} | "
                     f"{r['valor B']:.2f} | {r['p']:.4f} | {r['p_holm']:.4f} | {a12s} | {'sí' if r['signif'] else 'no'} |")
    lines += ["", "Valores A/B: mediana (Mann-Whitney), desviación estándar (Brown-Forsythe), "
              "n.º de réplicas que alcanzan el umbral (log-rank). Â12 = P(adaptativa > base); en complejidad, "
              "Â12 < 0,5 significa redes más pequeñas.", "",
              "Figuras: convergencia.png, cajas.png, operadores.png" + (", supervivencia.png" if tte else "")]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
