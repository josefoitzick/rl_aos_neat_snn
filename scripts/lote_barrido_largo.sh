#!/usr/bin/env bash
# Barrido largo de la fase de comparación (docs/diseno_neat_ap.md §15.4.1, configs/barrido_largo.json):
# 22 entornos × 2 protocolos (_fijo: 10 episodios fijos; _rot: --resample-train) × 2000 generaciones.
# ant va aparte con 1000 generaciones (scripts/lote_barrido_largo_ant.sh).
# Responde dos preguntas: ¿NEAT cierra la brecha con más presupuesto (cortes en 500, 1000 y 2000)? y ¿cuánto
# del mal desempeño es sobreajuste a los episodios fijos?
#
# Reparto: desde el 2026-10-04 ~17:30 el tope es de 256 hilos en total (entra otro usuario).
#   - semillas 1–5 primero (para la reunión del martes 2026-10-06): 36 procesos × 4 hilos = 144, nice 10;
#     junto con ant (10 × 10 = 100) y la última referencia RL (10) suman ~254;
#   - semillas 6–10 después, con el mismo lanzador.
# La eficiencia por corrida cae sobre 8 hilos (~/tesis/runs/bench_hilos).
#
# Protecciones contra resultados corruptos (las de los barridos A y B):
#   - candado: un segundo lanzamiento no arranca mientras éste corre;
#   - run_grid.py marca cada corrida con .running (atómico) y no la ejecuta dos veces;
#   - binario congelado en ~/tesis/runs/barrido_largo_bin (sha256 en el log), verificado equivalente al del
#     barrido A sin --resample-train (scripts/equivalencia_binarios.sh);
#   - reanudable: run_grid.py omite las corridas con summary.txt.
#
# Uso: tmux new -d -s barrido_largo 'bash scripts/lote_barrido_largo.sh 1-5 36'
set -u
cd "$(dirname "$0")/.."
SEMILLAS=${1:-1-5}
JOBS=${2:-36}
LOG=~/tesis/runs/barrido_largo_logs
mkdir -p "$LOG"
CFG=configs/barrido_largo.json
BIN=~/tesis/runs/barrido_largo_bin/neat_control

exec 9> ~/tesis/runs/barrido_largo.lock
if ! flock -n 9; then
    echo "Ya hay un barrido_largo en curso (candado tomado). No se lanza otro." | tee -a "$LOG/lanzador.log"
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "Falta el binario congelado $BIN (copiarlo tras verificar la equivalencia)." | tee -a "$LOG/lanzador.log"
    exit 1
fi
echo "inicio: $(date)  semillas: $SEMILLAS  jobs: $JOBS × 4 hilos  commit: $(git rev-parse --short HEAD 2>/dev/null)" \
     "rl-tools: $(git -C external/rl-tools rev-parse --short HEAD)  binario sha256: $(sha256sum "$BIN" | cut -c1-16)" \
     | tee -a "$LOG/lanzador.log"
python3 scripts/run_grid.py "$CFG" --seeds "$SEMILLAS" --jobs "$JOBS" --threads 4 --nice 10 --binary "$BIN" \
    > "$LOG/resto_s$SEMILLAS.log" 2>&1
echo "barrido_largo (semillas $SEMILLAS) terminado: $(date)" | tee -a "$LOG/lanzador.log" > "$LOG/FIN_s$SEMILLAS"
