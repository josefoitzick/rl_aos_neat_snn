#ifndef AOS_ACTION_MAP_HPP
#define AOS_ACTION_MAP_HPP

// Acción del agente PPO → P sobre el catálogo (docs/diseno_ppo.md §2.2):
//   P = softmax(log P₀ + c·tanh(a/c))
// Con a = 0 se obtiene P₀ (la P estática ajustada). Cada logit se aleja a lo más ±c de log P₀, así que el
// cociente P_i/P_j se mueve a lo más e^{±2c} respecto de P₀. Lo comparten el entrenador y el despliegue.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace aos {

// log P₀ con piso: ningún operador queda inalcanzable. P₀ se renormaliza después de aplicar el piso.
inline std::vector<double> logP0(const std::vector<double>& P0, double floor = 0.002) {
    if (P0.empty()) throw std::invalid_argument("logP0: P0 vacía");
    std::vector<double> p(P0.size());
    double s = 0.0;
    for (size_t i = 0; i < P0.size(); i++) {
        if (!(P0[i] >= 0.0) || !std::isfinite(P0[i])) throw std::invalid_argument("logP0: valor negativo o no finito");
        p[i] = std::max(P0[i], floor);
        s += p[i];
    }
    for (auto& x : p) x = std::log(x / s);
    return p;
}

template <typename A>
std::vector<double> actionToP(const A* a, const std::vector<double>& logp0, double c) {
    const size_t K = logp0.size();
    std::vector<double> z(K);
    double zmax = -INFINITY;
    for (size_t i = 0; i < K; i++) {
        const double ai = static_cast<double>(a[i]);
        if (!std::isfinite(ai)) throw std::runtime_error("actionToP: acción no finita");
        z[i] = logp0[i] + c * std::tanh(ai / c);
        zmax = std::max(zmax, z[i]);
    }
    double s = 0.0;
    for (auto& v : z) {
        v = std::exp(v - zmax);
        s += v;
    }
    for (auto& v : z) v /= s;
    return z;
}

// Fronteras de decisión (docs/diseno_ppo.md §2.1): la decisión t se toma tras evaluar la generación
// g_t = ⌊t·(G−1)/T⌋ y su P rige las generaciones g_t+1 … g_{t+1}; g_T = G−1. Con G ≥ T + 1, cada
// ventana tiene al menos una generación.
inline std::vector<int> decisionBoundaries(int G, int T) {
    if (T < 1 || G < T + 1) throw std::invalid_argument("decisionBoundaries: se necesita G ≥ T + 1");
    std::vector<int> g(T + 1);
    for (int t = 0; t <= T; t++) g[t] = static_cast<int>((static_cast<long long>(t) * (G - 1)) / T);
    return g;
}

}  // namespace aos

#endif
