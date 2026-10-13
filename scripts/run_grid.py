#!/usr/bin/env python3
"""Lanza un experimento: variantes × semillas de `neat_control`, con paralelismo acotado.

Uso:
    python3 scripts/run_grid.py configs/calib_episodes.json --seeds 1-5 --jobs 6 --threads 8

El JSON define:
    {
      "name": "calib_episodes",                     # carpeta en ~/tesis/runs/
      "base": ["--env", "pendulum", ...],           # argumentos comunes
      "variants": {"E5": ["--episodes", "5"], ...}  # argumentos por variante
    }

Salida: ~/tesis/runs/<name>/<variante>/seed_<NN>/{generations.csv, summary.txt, log.txt}.
Las corridas con summary.txt se saltan (el experimento se puede reanudar).
mullo es un nodo compartido: jobs × threads debería quedar muy por debajo de 576.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = Path.home() / "tesis" / "runs"


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_one(binary, args, out_dir, threads, nice=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Marca atómica de "en curso": si otro proceso ya tomó esta corrida, no se ejecuta dos veces
    # (dos procesos escribiendo el mismo generations.csv lo corromperían).
    marker = out_dir / ".running"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return -2, 0.0
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    prefix = ["nice", "-n", str(nice)] if nice else []
    t0 = time.time()
    with open(out_dir / "log.txt", "w") as log:
        log.write(" ".join([str(binary)] + args) + "\n")
        log.flush()
        proc = subprocess.run(prefix + [str(binary)] + args, stdout=log, stderr=subprocess.STDOUT, env=env)
    marker.unlink(missing_ok=True)
    return proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="JSON del experimento (configs/*.json)")
    ap.add_argument("--seeds", default="1-11", help="p. ej. 1-11 o 1,3,5 (por defecto 1-11: 11 réplicas)")
    ap.add_argument("--jobs", type=int, default=6, help="procesos simultáneos")
    ap.add_argument("--threads", type=int, default=8, help="hilos OpenMP por proceso")
    ap.add_argument("--binary", default=str(REPO / "build" / "bin" / "neat_control"))
    ap.add_argument("--only", help="variantes a correr, separadas por comas")
    ap.add_argument("--exclude", help="variantes a omitir, separadas por comas")
    ap.add_argument("--nice", type=int, default=0, help="prioridad baja para los procesos (nice), p. ej. 10")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    variants = cfg["variants"]
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only.split(",")}
    if args.exclude:
        variants = {k: v for k, v in variants.items() if k not in args.exclude.split(",")}
    seeds = parse_seeds(args.seeds)
    root = RUNS / cfg["name"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    tasks = []
    for vname, vargs in variants.items():
        for s in seeds:
            out = root / vname / f"seed_{s:02d}"
            if (out / "summary.txt").exists():
                continue
            full = list(cfg["base"]) + list(vargs) + ["--seed", str(s), "--out", str(out), "--print-every", "1000"]
            tasks.append((vname, s, full, out))

    total = len(variants) * len(seeds)
    print(f"{cfg['name']}: {len(tasks)} corridas pendientes de {total} "
          f"({args.jobs} procesos × {args.threads} hilos) → {root}", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, args.binary, full, out, args.threads, args.nice): (v, s) for v, s, full, out in tasks}
        for k, fut in enumerate(as_completed(futures), 1):
            v, s = futures[fut]
            code, secs = fut.result()
            failures += code != 0
            if code == -2:
                print(f"[{k}/{len(tasks)}] {v} seed {s}: OMITIDA (otro proceso la tiene en curso: .running)", flush=True)
                continue
            print(f"[{k}/{len(tasks)}] {v} seed {s}: {'ok' if code == 0 else f'ERROR {code}'} ({secs:.0f} s)", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
