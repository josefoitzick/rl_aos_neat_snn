#!/usr/bin/env bash
# B5: equivalencia entre el binario congelado del barrido A y el binario actual (con l2f-visual y hyperdrone).
# Corre ambos con la misma configuración corta y compara generations.csv sin la columna de tiempo (seconds)
# y summary.txt sin la duración total. Idénticos ⇒ los entornos de A no cambiaron y A + B son un solo experimento.
#
# Uso: bash scripts/equivalencia_binarios.sh [binario_A] [binario_nuevo]
set -u
cd "$(dirname "$0")/.."
A=${1:-$HOME/tesis/runs/barrido_etapa_a_bin/neat_control}
B=${2:-build/bin/neat_control}
OUT=$HOME/tesis/runs/equivalencia_binarios
mkdir -p "$OUT"
echo "A: $A ($(sha256sum "$A" | cut -c1-16))"
echo "B: $B ($(sha256sum "$B" | cut -c1-16))"

# --generations y --seed van por caso (CLI11 rechaza opciones repetidas)
BASE=(--selector static --episodes 2 --test-episodes 2 --final-episodes 2
      --weight-replace-prob 0.02 --target-species 10 --print-every 1000)
# Cubre: genérico, torque_scale, memoria (keep-state), multiagente, imagen no negativa, car (InitialState
# propio), car-track, flag (parámetros por episodio), l2f del zoo, ant (MuJoCo)
CASOS=(
  "pendulum|--env pendulum --generations 3 --seed 7"
  "acrobot_t1|--env acrobot --torque-scale 0.2 --generations 3 --seed 7"
  "memory_memoria|--env memory --keep-state --generations 3 --seed 7"
  "flag_memoria|--env flag --keep-state --generations 3 --seed 7"
  "bottleneck|--env bottleneck --generations 3 --seed 7"
  "reacher-visual|--env reacher-visual --generations 3 --seed 7"
  "car|--env car --generations 3 --seed 7"
  "car-track|--env car-track --generations 3 --seed 7"
  "l2f-tiny|--env l2f-tiny --generations 3 --seed 7"
  "ant|--env ant --generations 1 --seed 7"
)
fallas=0
for c in "${CASOS[@]}"; do
    n=${c%%|*}; args=${c#*|}
    mkdir -p "$OUT/$n"
    for k in A B; do
        bin=$([ $k = A ] && echo "$A" || echo "$B")
        OMP_NUM_THREADS=4 nice -n 10 "$bin" "${BASE[@]}" $args --out "$OUT/$n/$k" > "$OUT/$n/$k.log" 2>&1 || echo "  $n/$k: salida $?"
    done
    # Falla explícita si falta algún resultado (comparar dos archivos ausentes no prueba nada)
    ok=1
    for k in A B; do
        for f in generations.csv summary.txt; do
            [ -s "$OUT/$n/$k/$f" ] || { echo "FALTA     $n/$k/$f"; ok=0; }
        done
    done
    if [ $ok = 0 ]; then fallas=$((fallas+1)); continue; fi
    gens=$(($(wc -l < "$OUT/$n/A/generations.csv") - 1))
    col=$(head -1 "$OUT/$n/A/generations.csv" | tr ',' '\n' | grep -n '^seconds$' | cut -d: -f1)
    if [ "$gens" -ge 1 ] && cmp -s <(cut -d, -f"$col" --complement "$OUT/$n/A/generations.csv") <(cut -d, -f"$col" --complement "$OUT/$n/B/generations.csv") \
       && cmp -s <(grep -v '^seconds=' "$OUT/$n/A/summary.txt") <(grep -v '^seconds=' "$OUT/$n/B/summary.txt"); then
        echo "IDENTICO  $n ($gens generaciones, $(grep '^final_test=' "$OUT/$n/A/summary.txt"))"
    else
        echo "DIFIERE   $n"; fallas=$((fallas+1))
    fi
done
# Control negativo: la misma comparación entre semillas distintas DEBE detectar diferencia
mkdir -p "$OUT/control_negativo"
OMP_NUM_THREADS=4 nice -n 10 "$B" "${BASE[@]}" --env pendulum --generations 3 --seed 8 --out "$OUT/control_negativo/B" > "$OUT/control_negativo/B.log" 2>&1
col=$(head -1 "$OUT/pendulum/A/generations.csv" | tr ',' '\n' | grep -n '^seconds$' | cut -d: -f1)
if [ -s "$OUT/control_negativo/B/generations.csv" ] && ! cmp -s <(cut -d, -f"$col" --complement "$OUT/pendulum/A/generations.csv") <(cut -d, -f"$col" --complement "$OUT/control_negativo/B/generations.csv"); then
    echo "CONTROL_NEGATIVO ok (la comparación detecta diferencias)"
else
    echo "CONTROL_NEGATIVO FALLA (la comparación no detecta diferencias)"; fallas=$((fallas+1))
fi
echo "EQUIVALENCIA_FIN fallas=$fallas"
