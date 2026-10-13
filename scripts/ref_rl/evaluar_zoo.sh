#!/usr/bin/env bash
# Paso 3 de docs/diseno_neat_ap.md §15.4: evalúa cada política RL final (zoo de RLtools y configuraciones
# propias) en NUESTRO entorno, con los mismos 100 episodios finales de cada semilla de NEAT (1..5).
#
# Salida en ~/tesis/runs/ref_rl:
#   evaluacion.csv  una fila por política × semilla NEAT, más los controles 'zero' y 'random' y el oráculo de
#                   memory (src/ref_rl/memory_oracle.cpp, óptimo exacto);
#   zoo_propio.csv  último retorno medio reportado por el entrenador en su propia evaluación (horizonte y
#                   aritmética float del zoo: sirve solo de referencia cruzada, no para el score).
# Controles: rl_ref_eval exige el autotest del checkpoint y dimensiones iguales; 'zero' y 'random' deben
# reproducir el trivial del probe.
#
# Uso: [PARALELO=12] bash scripts/ref_rl/evaluar_zoo.sh [v2]   (reanudable: solo recompila las políticas nuevas;
#      PARALELO acota los procesos simultáneos de compilación y evaluación, por el tope de hilos del nodo)
# La versión elige las carpetas: v2 (OpenBLAS, 10 semillas; por omisión) = zoo_v2/ y zoo_propias_v2/;
# v1 (sin BLAS, interrumpida el 2026-10-04 15:00) = zoo/ y zoo_propias/.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=~/tesis/runs/ref_rl
EVAL=$PWD/build/bin/rl_ref_eval
V=${1:-v2}
if [[ $V == v1 ]]; then DIRS=("$R"/zoo "$R"/zoo_propias); P=$R/politicas; OUTV=""; else DIRS=("$R"/zoo_$V "$R"/zoo_propias_$V); P=$R/politicas_$V; OUTV=_$V; fi
mkdir -p "$P"

# Entrenador → "entorno nuestro  torque_scale"
declare -A ENV=(
  [pendulum-v1]="pendulum 1" [pendulum-position-v1]="pendulum-position 1" [pendulum-multitask-v1]="pendulum-meta 1"
  [acrobot-swingup-v0]="acrobot-swingup 1" [reacher-v0]="reacher 1" [l2f]="l2f-tiny 1" [bottleneck-v0]="bottleneck 1"
  [ant-v4]="ant 1"
  [tesis-acrobot-t5]="acrobot 1" [tesis-acrobot-t1]="acrobot 0.2"
  [tesis-pendulum-multitask]="pendulum-multitask 1" [tesis-pendulum-velocity]="pendulum-velocity 1"
  [tesis-pendulum-position-gru]="pendulum-position 1" [tesis-memory]="memory 1" [tesis-l2f-full]="l2f 1"
  [tesis-car]="car 1" [tesis-car-track]="car-track 1"
)

# 1. Corridas terminadas sin error: checkpoint final, biblioteca, entorno, torque
echo "zoo,seed,zoo_mean_return,zoo_mean_len" > "$R/zoo_propio$OUTV.csv"
: > "$P/lista.txt"
for d in "${DIRS[0]}"/*/seed_* "${DIRS[1]}"/*/seed_*; do
  [[ -f $d/FIN && $(cut -d' ' -f1 "$d/FIN") == 0 ]] || continue
  nombre=$(basename "$(dirname "$d")"); seed=${d##*seed_}; ent=${nombre%_*}
  env=${ENV[$ent]:-}
  [[ -n $env ]] || { echo "sin mapeo: $ent" >&2; continue; }
  ck=$(ls -d "$d"/experiments/*/*/*/*/steps/*/checkpoint.h | sort | tail -1)
  echo "$ck $P/${nombre}_seed$seed.so $env" >> "$P/lista.txt"
  ultimo=$(grep 'Mean return' "$d/log.txt" | tail -1)
  echo "$nombre,$seed,$(sed -E 's/.*Mean return: ([^ ]+).*length: ([^ ]+).*/\1,\2/' <<< "$ultimo")" >> "$R/zoo_propio$OUTV.csv"
done

# 2. Compilación en paralelo (solo si la biblioteca no existe o es más vieja que el checkpoint)
compilar() {
  [[ $1 -ot $2 ]] && return 0
  bash scripts/ref_rl/compilar_politica.sh "$1" "$2" > "$2.log" 2>&1 || echo "ERROR compilando $2 (ver $2.log)"
}
export -f compilar
xargs -P "${PARALELO:-12}" -L 1 bash -c 'compilar $0 $1' < "$P/lista.txt"
g++ -std=c++17 -O2 -fPIC -shared -fvisibility=hidden src/ref_rl/memory_oracle.cpp -o "$P/memory_oracle.so"

# 3. Evaluación en paralelo: políticas, oráculo de memory y controles por (entorno, torque)
rm -f "$R"/eval_*.csv  # temporales de esta corrida
{
  while read -r ck so env ts; do [[ -f $so ]] && echo "$so $env $ts"; done < "$P/lista.txt"
  echo "$P/memory_oracle.so memory 1"
  { cut -d' ' -f3,4 "$P/lista.txt"; echo "memory 1"; } | sort -u | while read -r env ts; do
    echo "zero $env $ts"; echo "random $env $ts"
  done
} | xargs -P "${PARALELO:-12}" -L 1 bash -c \
  "$EVAL --env \$1 --torque-scale \$2 --policy \$0 --out $R/eval_\$(basename \$0 .so)_\$1_\$2.csv > /dev/null || echo \"ERROR evaluando \$0 en \$1\""

head -1 "$(ls "$R"/eval_*.csv | head -1)" > "$R/evaluacion$OUTV.csv"
for f in "$R"/eval_*.csv; do tail -n +2 "$f" >> "$R/evaluacion$OUTV.csv"; rm "$f"; done
echo "filas: $(($(wc -l < "$R/evaluacion$OUTV.csv") - 1))  → $R/evaluacion$OUTV.csv"
