#!/usr/bin/env bash
# Compila los entrenadores del zoo de RLtools usados como referencia RL (docs/diseno_neat_ap.md §15.4, paso 2).
# Un binario por (entorno del zoo, algoritmo), en ~/tesis/runs/ref_rl/bin_v2. Quedan congelados: el lanzador
# registra su sha256. Sin -ffast-math (a diferencia del README del zoo) para no alterar la aritmética.
#
# Uso: bash scripts/ref_rl/compilar_zoo.sh          (compila en paralelo, ~10–60 s cada uno)
# Backend OpenBLAS (~/opt/conda-blas, conda-forge openblas + libcblas, sin root): ~15× más rápido que el backend
# genérico en SAC (medido en acrobot-swingup). Se corre con OPENBLAS_NUM_THREADS=1: más hilos por proceso casi no
# ayudan; conviene más procesos (semillas). v1 (sin BLAS, carpetas bin/ y bin_propias/) quedó interrumpida.
set -euo pipefail
cd "$(dirname "$0")/../.."
RLT=external/rl-tools
BIN=~/tesis/runs/ref_rl/bin_v2
BLAS=~/opt/conda-blas
MUJOCO=~/opt/mujoco-2.3.5
mkdir -p "$BIN"

# Formato: <nombre> <MACRO_ENTORNO> <MACRO_ALGORITMO>
OBJETIVOS=(
  "pendulum-v1_sac            PENDULUM_V1            SAC"
  "pendulum-v1_td3            PENDULUM_V1            TD3"
  "pendulum-v1_ppo            PENDULUM_V1            PPO"
  "pendulum-position-v1_sac   PENDULUM_POSITION_V1   SAC"
  "pendulum-position-v1_td3   PENDULUM_POSITION_V1   TD3"
  "pendulum-position-v1_ppo   PENDULUM_POSITION_V1   PPO"
  "pendulum-multitask-v1_sac  PENDULUM_MULTITASK_V1  SAC"
  "pendulum-multitask-v1_td3  PENDULUM_MULTITASK_V1  TD3"
  "pendulum-multitask-v1_ppo  PENDULUM_MULTITASK_V1  PPO"
  "acrobot-swingup-v0_sac     ACROBOT_SWINGUP_V0     SAC"
  "reacher-v0_ppo             REACHER_V0             PPO"
  "l2f_sac                    L2F                    SAC"
  "l2f_td3                    L2F                    TD3"
  "bottleneck-v0_ppo          BOTTLENECK_V0          PPO"
  "ant-v4_ppo                 ANT_V4                 PPO"
  "ant-v4_td3                 ANT_V4                 TD3"
)

compilar() {
  local nombre=$1 env=$2 alg=$3 extra=()
  if [[ $env == ANT_V4 ]]; then
    extra=(-isystem "$MUJOCO/include" -L"$MUJOCO/lib" -Wl,-rpath,"$MUJOCO/lib" -lmujoco)
  fi
  if g++ -std=c++17 -O3 -march=native -I "$RLT/include" -DRL_TOOLS_ENABLE_CLI11 \
       -I "$BLAS/include" -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS \
       -DRL_TOOLS_RL_ZOO_ALGORITHM_"$alg" -DRL_TOOLS_RL_ZOO_ENVIRONMENT_"$env" \
       "$RLT/src/rl/zoo/zoo_cli.cpp" "$RLT/src/rl/zoo/zoo.cpp" -o "$BIN/zoo_$nombre" "${extra[@]}" -L"$BLAS/lib" -Wl,-rpath,"$BLAS/lib" -lopenblas \
       > "$BIN/zoo_$nombre.compilacion.log" 2>&1; then
    echo "ok     $nombre"
  else
    echo "ERROR  $nombre (ver $BIN/zoo_$nombre.compilacion.log)"
  fi
}
export -f compilar
export RLT BIN MUJOCO BLAS
printf '%s\n' "${OBJETIVOS[@]}" | xargs -P 16 -L 1 bash -c 'compilar $0 $1 $2'
( cd "$BIN" && sha256sum zoo_* 2>/dev/null | grep -v '\.log' > SHA256SUMS )
echo "rl-tools: $(git -C "$RLT" rev-parse --short HEAD)" > "$BIN/VERSION"
