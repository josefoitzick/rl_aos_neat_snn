#!/usr/bin/env bash
# Verificación V4 del entrenador de PPO sobre NEAT (docs/diseno_ppo.md §5): pendulum con un presupuesto chico.
#   A: entrenamiento completo (64 corridas = 8 iteraciones) y validación pareada contra P₀ al final.
#   B: el mismo entrenamiento detenido en la iteración 4 (--max-iter) y reanudado (--reanudar).
# A y B deben coincidir bit a bit: ppo.csv (sin la columna de segundos), corridas.csv, decisiones.csv y el actor
# exportado (actor_final/checkpoint.h).
#
# Uso: scripts/verificar_ppo_pendulum.sh [directorio de salida]
set -u
OUT=${1:-$HOME/tesis/runs/ppo_pendulum_v4}
B=$(cd "$(dirname "$0")/.." && pwd)/build/bin/aos_ppo_train
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
rm -rf "$OUT/A" "$OUT/B"
mkdir -p "$OUT"
echo "aos_ppo_train $(sha256sum "$B" | cut -c1-16)"
COM="--env pendulum --resample-train --pop 30 --generations 51 --T 50 --episodes 3 --test-episodes 3 --final-episodes 20
     --episodios-recompensa 10 --corridas 64 --semilla 1"
fallas=0
nice -n 10 "$B" $COM --validar 16 --out "$OUT/A" > "$OUT/A.log" 2>&1 || { echo "FALLA: A"; fallas=$((fallas + 1)); }
nice -n 10 "$B" $COM --max-iter 4 --out "$OUT/B" > "$OUT/B1.log" 2>&1 || { echo "FALLA: B (primera parte)"; fallas=$((fallas + 1)); }
nice -n 10 "$B" $COM --reanudar --out "$OUT/B" > "$OUT/B2.log" 2>&1 || { echo "FALLA: B (reanudación)"; fallas=$((fallas + 1)); }

igual() {   # $1 = descripción, $2 y $3 = archivos (o sustituciones de proceso)
    if diff -q "$2" "$3" > /dev/null; then echo "  OK    $1"; else echo "  FALLA $1"; fallas=$((fallas + 1)); fi
}
echo "A (de corrido) frente a B (detenido en la iteración 4 y reanudado):"
igual "ppo.csv" <(cut -d, -f1,2,4- "$OUT/A/ppo.csv") <(cut -d, -f1,2,4- "$OUT/B/ppo.csv")
igual "corridas.csv" "$OUT/A/corridas.csv" "$OUT/B/corridas.csv"
igual "decisiones.csv" "$OUT/A/decisiones.csv" "$OUT/B/decisiones.csv"
igual "actor exportado" "$OUT/A/actor_final/checkpoint.h" "$OUT/B/actor_final/checkpoint.h"
grep -q "reanudando: iteración 4" "$OUT/B2.log" && echo "  OK    B se reanudó desde la iteración 4" || { echo "  FALLA B no se reanudó desde la iteración 4"; fallas=$((fallas + 1)); }
echo "Iteraciones de A:"
grep "^iter" "$OUT/A.log"
grep "^validación" "$OUT/A.log"
echo "fallas: $fallas"
[ $fallas == 0 ]
