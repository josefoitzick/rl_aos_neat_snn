#!/usr/bin/env bash
# Compila un checkpoint.h del zoo de RLtools como biblioteca de política para rl_ref_eval
# (src/experiment/rl_policy_shim.cpp; símbolos ocultos salvo rlp_*).
#
# Parche (bug de RLtools render@6bb2fda): la exportación de la capa td3_sampling (actores TD3) escribe
# Configuration<float, ...> donde el resto de las capas usa la política de tipos numéricos. Se corrige en una
# COPIA del checkpoint (<salida>.checkpoint.h); el original no se toca. En modo evaluación la capa no muestrea:
# el parche no cambia la acción.
#
# Uso: bash scripts/ref_rl/compilar_politica.sh <checkpoint.h> <salida.so>
set -euo pipefail
CK=$(realpath "$1")            # rutas absolutas ANTES del cd
OUT=$(realpath -m "$2")
cd "$(dirname "$0")/../.."
COPIA=$OUT.checkpoint.h
POL='RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::UseCase<RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::categories::Parameter, float>>'
sed "s#td3_sampling::Configuration<float, #td3_sampling::Configuration<$POL, #" "$CK" > "$COPIA"
g++ -std=c++17 -O2 -fPIC -shared -fvisibility=hidden -fvisibility-inlines-hidden -Wl,-Bsymbolic \
    -I external/rl-tools/include -DRL_POLICY_CHECKPOINT="\"$(realpath "$COPIA")\"" \
    src/experiment/rl_policy_shim.cpp -o "$OUT"
