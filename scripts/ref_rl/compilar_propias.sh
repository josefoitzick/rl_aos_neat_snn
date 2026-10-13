#!/usr/bin/env bash
# Compila los entrenadores PROPIOS de la referencia RL (configuraciones en src/ref_rl, docs/diseno_neat_ap.md
# §15.4, paso 2) a partir de una copia generada del zoo con un gancho (scripts/ref_rl/generar_zoo_tesis.py).
# Binarios en ~/tesis/runs/ref_rl/bin_propias_v2, congelados (SHA256SUMS), con el prefijo zoo_ para reutilizar el
# lanzador (scripts/ref_rl/lanzar_propias.sh).
#
# Uso: bash scripts/ref_rl/compilar_propias.sh
# Backend OpenBLAS (~/opt/conda-blas, conda-forge openblas + libcblas, sin root): ~15× más rápido que el backend
# genérico en SAC (medido en acrobot-swingup). Se corre con OPENBLAS_NUM_THREADS=1: más hilos por proceso casi no
# ayudan; conviene más procesos (semillas). v1 (sin BLAS, carpetas bin/ y bin_propias/) quedó interrumpida.
set -euo pipefail
cd "$(dirname "$0")/../.."
RLT=external/rl-tools
BIN=~/tesis/runs/ref_rl/bin_propias_v2
BLAS=~/opt/conda-blas
GEN=~/tesis/runs/ref_rl/gen
mkdir -p "$BIN" "$GEN"
python3 scripts/ref_rl/generar_zoo_tesis.py "$RLT/src/rl/zoo/zoo.cpp" "$GEN/zoo_tesis.cpp"
# zoo_cli.cpp va aparte y sin -include: RLtools define funciones no inline en sus headers (una sola unidad)
g++ -std=c++17 -O3 -march=native -I "$RLT/include" -DRL_TOOLS_ENABLE_CLI11 -c "$RLT/src/rl/zoo/zoo_cli.cpp" -o "$GEN/zoo_cli.o"

# Formato: <nombre> <header de src/ref_rl> <ALGORITMO> [definiciones extra]
OBJETIVOS=(
  "acrobot-t5_sac           acrobot.h   SAC  -DTESIS_ACROBOT_TORQUE=5"
  "acrobot-t1_sac           acrobot.h   SAC  -DTESIS_ACROBOT_TORQUE=1"
  "pendulum-multitask_sac   pendulos.h  SAC  -DTESIS_PENDULO=1"
  "pendulum-multitask_td3   pendulos.h  TD3  -DTESIS_PENDULO=1"
  "pendulum-multitask_ppo   pendulos.h  PPO  -DTESIS_PENDULO=1"
  "pendulum-velocity_sac    pendulos.h  SAC  -DTESIS_PENDULO=2"
  "pendulum-velocity_td3    pendulos.h  TD3  -DTESIS_PENDULO=2"
  "pendulum-velocity_ppo    pendulos.h  PPO  -DTESIS_PENDULO=2"
  "pendulum-position-gru_sac pendulos.h SAC  -DTESIS_PENDULO=3"
  "memory_sac               memory.h    SAC"
  "l2f-full_sac             l2f.h       SAC  -DRL_TOOLS_RL_ZOO_ENVIRONMENT_L2F"
  "car_ppo                  car.h       PPO  -DTESIS_CAR_TRACK=0"
  "car-track_ppo            car.h       PPO  -DTESIS_CAR_TRACK=1"
)

compilar() {
  local nombre=$1 header=$2 alg=$3; shift 3
  local h; h=$(realpath "src/ref_rl/$header")
  if g++ -std=c++17 -O3 -march=native -I "$RLT/include" -iquote "$RLT/src/rl/zoo" -DRL_TOOLS_ENABLE_CLI11 \
       -I "$BLAS/include" -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS \
       -DRL_TOOLS_RL_ZOO_ALGORITHM_"$alg" "$@" -include "$h" -DTESIS_REF_CONFIG="\"$h\"" \
       -DTESIS_REF_ENV_NAME="\"tesis-${nombre%_*}\"" \
       "$GEN/zoo_tesis.cpp" "$GEN/zoo_cli.o" -o "$BIN/zoo_tesis-$nombre" -L"$BLAS/lib" -Wl,-rpath,"$BLAS/lib" -lopenblas \
       > "$BIN/zoo_tesis-$nombre.compilacion.log" 2>&1; then
    echo "ok     $nombre"
  else
    echo "ERROR  $nombre (ver $BIN/zoo_tesis-$nombre.compilacion.log)"
  fi
}
export -f compilar
export RLT BIN GEN BLAS
printf '%s\n' "${OBJETIVOS[@]}" | xargs -P 16 -L 1 bash -c 'compilar "$@"' _
( cd "$BIN" && sha256sum $(ls zoo_* | grep -v '\.log$') > SHA256SUMS )
echo "rl-tools: $(git -C "$RLT" rev-parse --short HEAD)" > "$BIN/VERSION"
