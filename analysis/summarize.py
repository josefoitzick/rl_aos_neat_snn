#!/usr/bin/env python3
"""Resumen rápido de un experimento (solo biblioteca estándar).

Uso:
    python3 analysis/summarize.py ~/tesis/runs/calib_episodes

Por variante: n réplicas, test final (media ± desv. est. y mediana), fitness de entrenamiento,
brecha entrenamiento − test, complejidad del mejor genoma, especies promedio y tiempo.
"""
import csv
import statistics as st
import sys
from pathlib import Path


def read_summary(path):
    d = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def fmt(xs, prec=1):
    if not xs:
        return "—"
    m = st.mean(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.{prec}f} ± {s:.{prec}f}"


def main():
    root = Path(sys.argv[1]).expanduser()
    rows = []
    for vdir in sorted(p for p in root.iterdir() if p.is_dir()):
        test, train, hidden, conns, secs, species = [], [], [], [], [], []
        for sdir in sorted(vdir.glob("seed_*")):
            f = sdir / "summary.txt"
            if not f.exists():
                continue
            s = read_summary(f)
            test.append(float(s["final_test"]))
            train.append(float(s["best_train"]))
            hidden.append(int(s["best_hidden"]))
            conns.append(int(s["best_conns"]))
            secs.append(float(s["seconds"]))
            with open(sdir / "generations.csv") as fh:
                sp = [float(r["species"]) for r in csv.DictReader(fh)]
            species.append(st.mean(sp))
        if test:
            gap = [a - b for a, b in zip(train, test)]
            rows.append((vdir.name, len(test), fmt(test), f"{st.median(test):.1f}", fmt(train), fmt(gap),
                         fmt(hidden), fmt(conns), fmt(species), f"{st.mean(secs):.0f}"))

    header = ("variante", "n", "test final", "mediana", "entrenamiento", "brecha", "ocultas", "conexiones",
              "especies", "s/corrida")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


if __name__ == "__main__":
    main()
