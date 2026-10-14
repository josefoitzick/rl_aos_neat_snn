#!/usr/bin/env bash
# Verificación V1 de NEAT como entorno del agente PPO (docs/diseno_ppo.md §5): aos_neat_check (NeatRun / NeatPool)
# debe dar exactamente lo mismo que neat_control con las mismas opciones y la misma secuencia de P.
# Se comparan generations.csv (sin la columna de segundos) y final_test / best_validation_test de summary.txt.
# Además, el neat_control recién compilado se compara con un binario congelado de referencia (control de que
# los cambios en src/neat no alteraron nada).
#
# Uso: scripts/verificar_neat_run.sh <neat_control de referencia congelado> [directorio de salida]
set -u
REF=${1:?falta el neat_control de referencia}
OUT=${2:-$HOME/tesis/runs/verif_neat_run}
B=$(cd "$(dirname "$0")/.." && pwd)/build/bin
NC=$B/neat_control
CHK=$B/aos_neat_check
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$OUT"
echo "neat_control $(sha256sum "$NC" | cut -c1-16) · aos_neat_check $(sha256sum "$CHK" | cut -c1-16) · referencia $(sha256sum "$REF" | cut -c1-16)"

fallas=0
comparar() {   # $1 = nombre, $2 = dir A, $3 = dir B
    local ok=1
    if ! diff -q <(cut -d, -f1-12,14- "$2/generations.csv") <(cut -d, -f1-12,14- "$3/generations.csv") > /dev/null; then ok=0; fi
    for k in final_test best_validation_test; do
        a=$(grep "^$k=" "$2/summary.txt" | cut -d= -f2); b=$(grep "^$k=" "$3/summary.txt" | cut -d= -f2)
        [ "$a" == "$b" ] || ok=0
    done
    local n=$(($(wc -l < "$2/generations.csv") - 1))
    if [ $ok == 1 ]; then echo "  OK    $1 ($n generaciones, test final $(grep '^final_test=' "$2/summary.txt" | cut -d= -f2))"
    else echo "  FALLA $1"; fallas=$((fallas + 1)); fi
}

caso() {   # $1 = nombre, $2 = T, resto = opciones comunes
    local nom=$1 T=$2; shift 2
    nice -n 10 "$NC" "$@" --out "$OUT/$nom/nc" > "$OUT/$nom.nc.log" 2>&1 || { echo "  FALLA $nom: neat_control"; fallas=$((fallas + 1)); return; }
    nice -n 10 "$CHK" "$@" --T "$T" --out "$OUT/$nom/chk" > "$OUT/$nom.chk.log" 2>&1 || { echo "  FALLA $nom: aos_neat_check"; fallas=$((fallas + 1)); return; }
    comparar "$nom" "$OUT/$nom/nc" "$OUT/$nom/chk"
}

COM="--resample-train --episodes 10 --test-episodes 10 --final-episodes 100 --print-every 1000"
P49="--pop 130 --target-species 23 --compat-weight-scale 19.8409 --survival 0.110521 --weight-power 0.694949 --weight-replace-prob 0.0480648 --add-node-weight 47.3157 --weight-init-range 22.8526 --weight-max 45.3731 --mutation-only 0.416365 --species-stagnation 21 --input-gain 61.951 --spikes-full 2.18394 --step-ms 30 --p 0.629289,0.492025,0.0889994"
P65="--pop 210 --target-species 21 --compat-weight-scale 23.2562 --survival 0.222545 --weight-power 0.624492 --weight-replace-prob 0.00239789 --add-node-weight 44.3952 --weight-init-range 47.2032 --weight-max 123.354 --mutation-only 0.315704 --species-stagnation 33 --input-gain 37.7812 --spikes-full 6.18327 --step-ms 80 --p 0.671579,0.00572059,0.00608217"
P96="--pop 70 --target-species 17 --compat-weight-scale 26.4399 --survival 0.206831 --weight-power 3.2612 --weight-replace-prob 0.00650202 --add-node-weight 35.5621 --weight-init-range 35.1012 --weight-max 68.4052 --mutation-only 0.229568 --species-stagnation 16 --input-gain 86.6369 --spikes-full 7.0623 --step-ms 50 --p 0.536266,0.024667,0.0490802"
CAT=mutate_weights,add_connection,add_node,weights_fine,weights_reset,weights_directed,neuron_bias,neuron_dynamics,neuron_family,input_gain,add_node_family,add_node_multi,duplicate_node,delete_connection,delete_simple_node
PCAT=0.4,0.05,0.04,0.1,0.03,0.05,0.04,0.04,0.03,0.04,0.03,0.02,0.03,0.05,0.05

echo "NeatRun / NeatPool frente a neat_control:"
caso pendulum_stanley 10 --env pendulum --seed 1 --generations 41 $COM
caso pendulum_switch 10 --env pendulum --seed 2 --generations 41 $COM --p 0.8,0.1,0.1 --p2 0.5,0.3,0.2 --p-switch 13
caso pendulum_catalogo 10 --env pendulum --seed 3 --generations 31 $COM --catalog $CAT --p $PCAT --credito-rango
caso acrobot_49 5 --env acrobot-swingup --seed 1 --generations 21 $COM $P49
caso l2f_65 5 --env l2f --seed 1 --generations 11 $COM $P65
caso bottleneck_96 5 --env bottleneck --seed 1 --generations 11 $COM $P96
# Camino paralelo (NeatPool): 4 semillas a la vez frente a 4 corridas de neat_control
nom=pendulum_paralelo
nice -n 10 "$CHK" --env pendulum --seed 1 --generations 41 $COM --T 10 --paralelo 4 --out "$OUT/$nom/chk" > "$OUT/$nom.chk.log" 2>&1 || { echo "  FALLA $nom: aos_neat_check"; fallas=$((fallas + 1)); }
for s in 1 2 3 4; do
    nice -n 10 "$NC" --env pendulum --seed $s --generations 41 $COM --out "$OUT/$nom/nc_$s" > "$OUT/$nom.nc_$s.log" 2>&1
    comparar "$nom (semilla $s)" "$OUT/$nom/nc_$s" "$OUT/$nom/chk/seed_0$s"
done

echo "neat_control recompilado frente a la referencia congelada:"
refcaso() {
    local nom=$1; shift
    nice -n 10 "$REF" "$@" --out "$OUT/ref_$nom" > "$OUT/ref_$nom.log" 2>&1
    if diff -q <(cut -d, -f1-12,14- "$OUT/$nom/nc/generations.csv") <(cut -d, -f1-12,14- "$OUT/ref_$nom/generations.csv") > /dev/null && \
       [ "$(grep '^final_test=' "$OUT/$nom/nc/summary.txt")" == "$(grep '^final_test=' "$OUT/ref_$nom/summary.txt")" ]; then
        echo "  OK    $nom"; else echo "  FALLA $nom"; fallas=$((fallas + 1)); fi
}
refcaso pendulum_catalogo --env pendulum --seed 3 --generations 31 $COM --catalog $CAT --p $PCAT --credito-rango
refcaso acrobot_49 --env acrobot-swingup --seed 1 --generations 21 $COM $P49

# Control negativo: una semilla distinta debe dar otro CSV (la comparación sí detecta diferencias)
nice -n 10 "$CHK" --env pendulum --seed 9 --generations 41 $COM --T 10 --out "$OUT/negativo" > "$OUT/negativo.log" 2>&1
if diff -q <(cut -d, -f1-12,14- "$OUT/pendulum_stanley/nc/generations.csv") <(cut -d, -f1-12,14- "$OUT/negativo/generations.csv") > /dev/null; then
    echo "  FALLA control negativo: semillas distintas dieron el mismo CSV"; fallas=$((fallas + 1))
else echo "  OK    control negativo (otra semilla da otro CSV)"; fi

echo "fallas: $fallas"
[ $fallas == 0 ]
