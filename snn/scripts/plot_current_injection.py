#!/usr/bin/env python3
"""Grafica el CSV generado por snn_current_injection: potencial de membrana
de cada tipo de neurona (2 filas x 3 columnas) y la corriente inyectada."""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Ruta al CSV generado por snn_current_injection")
    parser.add_argument("-o", "--output", help="Guardar la figura en este archivo en vez de mostrarla")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    neuron_columns = [c for c in df.columns if c not in ("step", "current")]

    plt.rcParams.update({"font.size": 16})

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)

    for ax, column in zip(axes.flat, neuron_columns):
        ax.plot(df["step"], df[column], color="tab:blue", linewidth=1)
        ax.set_title(column, fontsize=18)
        ax.set_xlabel("Step", fontsize=16)
        ax.set_ylabel("Potencial de membrana (mV)", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)

        current_ax = ax.twinx()
        current_ax.plot(df["step"], df["current"], color="tab:red", linewidth=1, alpha=0.5)
        current_ax.set_ylabel("Corriente (mA)", color="tab:red", fontsize=16)
        current_ax.tick_params(axis="y", labelcolor="tab:red", labelsize=14)

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Figura guardada en {args.output}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
