#!/usr/bin/env python3
"""Figuras de curvas de aprendizaje para presentar (una fila por entorno).

Uso (entorno conda `tesis-analisis`):
    python analysis/plot_curvas.py ~/tesis/runs/reunion2_pendulum ~/tesis/runs/reunion2_acrobot \\
        --variants estatico ap ap_snap --out ~/tesis/runs/reunion2_figuras

Por entorno:
  (a) fitness de test del campeón por generación: mediana e intervalo intercuartil entre réplicas
  (b) tamaño de la red del campeón (conexiones habilitadas): mediana
  (c) probabilidad media de cada operador por generación (una línea por variante y operador)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OPS = [("mutate_weights", "mutar pesos"), ("add_connection", "agregar conexión"), ("add_node", "agregar nodo")]
NOMBRES = {"estatico": "NEAT estático (Stanley 2002)", "static": "NEAT estático (Stanley 2002)",
           "ap": "NEAT + AP (crédito extremo)", "ap_snap": "NEAT + AP (crédito SNAP-NEAT)"}
COLORES = {"estatico": "#1f77b4", "static": "#1f77b4", "ap": "#d62728", "ap_snap": "#2ca02c"}
ESTILOS = {"mutate_weights": "-", "add_connection": "--", "add_node": ":"}


def load(root, variants):
    frames = []
    for v in variants:
        for f in sorted((root / v).glob("seed_*/generations.csv")):
            g = pd.read_csv(f)
            g["variant"], g["seed"] = v, f.parent.name
            frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    exps = [Path(e).expanduser() for e in args.experiments]
    fig, axes = plt.subplots(len(exps), 3, figsize=(17, 4.6 * len(exps)), squeeze=False)
    for row, root in zip(axes, exps):
        df = load(root, args.variants)
        env = root.name.split("_")[-1].capitalize()
        if df.empty:
            row[0].set_title(f"{env}: sin datos todavía")
            continue
        for v in args.variants:
            d = df[df["variant"] == v]
            if d.empty:
                continue
            n = d["seed"].nunique()
            q = d.groupby("gen")["champion_test"].quantile([0.25, 0.5, 0.75]).unstack()
            row[0].plot(q.index, q[0.5], color=COLORES.get(v), label=f"{NOMBRES.get(v, v)} (n={n})")
            row[0].fill_between(q.index, q[0.25], q[0.75], color=COLORES.get(v), alpha=0.18)
            c = d.groupby("gen")["champion_conns"].median()
            row[1].plot(c.index, c.values, color=COLORES.get(v), label=NOMBRES.get(v, v))
            m = d.groupby("gen")[[f"p_{o}" for o, _ in OPS]].mean()
            for o, nombre in OPS:
                row[2].plot(m.index, m[f"p_{o}"], color=COLORES.get(v), ls=ESTILOS[o],
                            label=f"{NOMBRES.get(v, v).split(' (')[0]} · {nombre}" if v != "estatico" or o == OPS[0][0] else None)
        row[0].set_title(f"{env}: fitness de test del campeón (mediana e IQR)")
        row[0].set_xlabel("generación"); row[0].set_ylabel("retorno medio (10 episodios no vistos)")
        row[1].set_title(f"{env}: conexiones habilitadas del campeón (mediana)")
        row[1].set_xlabel("generación")
        row[2].set_title(f"{env}: probabilidad de cada operador (media)")
        row[2].set_xlabel("generación"); row[2].set_ylim(0, 1)
        for ax in row:
            ax.grid(alpha=0.3)
        row[0].legend(fontsize=9, loc="lower right")
        row[2].legend(fontsize=7, loc="center right")
    fig.suptitle("NEAT + SNN (Izhikevich IZ-A) en control RLtools — selección de operadores estática vs adaptativa",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "curvas_aprendizaje.png", dpi=160)
    fig.savefig(out / "curvas_aprendizaje.pdf")
    print(f"→ {out / 'curvas_aprendizaje.png'}")


if __name__ == "__main__":
    main()
