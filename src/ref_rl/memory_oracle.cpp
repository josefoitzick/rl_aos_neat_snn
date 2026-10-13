// Política óptima de memory (óptimo analítico, docs/diseno_neat_ap.md §15.4, paso 2, nivel 3).
// La recompensa es −10·|u − c/10|, con c = unos en la ventana de las últimas HORIZON = 10 observaciones
// (incluida la actual). Con memoria perfecta de la ventana, u = c/10 da recompensa 0 en cada paso (c ≤ 10,
// así que u ∈ [0, 1] no se recorta): el retorno óptimo es exactamente 0.
// Misma interfaz que rl_policy_shim.cpp (rlp_*), para evaluarla con rl_ref_eval igual que a las políticas RL:
// sirve también de control de que la recompensa de nuestro entorno es la esperada (debe dar 0 exacto).
#include <deque>

namespace {
constexpr int HORIZON = 10;
std::deque<double> ventana;
}  // namespace

#define RLP_EXPORT extern "C" __attribute__((visibility("default")))

RLP_EXPORT int rlp_obs_dim() { return 1; }
RLP_EXPORT int rlp_act_dim() { return 1; }
RLP_EXPORT const char* rlp_name() { return "memory_oracle"; }
RLP_EXPORT double rlp_selftest() { return 0.0; }

// Estado inicial del entorno: historia en cero (memory/operations_generic.h, initial_state)
RLP_EXPORT void rlp_reset() { ventana.assign(HORIZON, 0.0); }

RLP_EXPORT void rlp_act(const double* obs, double* act) {
    ventana.pop_front();
    ventana.push_back(obs[0] > 0.5 ? 1.0 : 0.0);
    double c = 0;
    for (double b : ventana) c += b;
    act[0] = c / 10.0;
}
