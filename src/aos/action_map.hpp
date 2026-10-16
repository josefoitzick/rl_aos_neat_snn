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
#include "neat/variacion.hpp"

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

// Plan B (docs/experimentos/15_margen_intensidades.md): además de P, o en vez de P, la acción mueve las intensidades
// de variación. Cada una se mueve alrededor de su valor base con la misma forma acotada que P (c·tanh(a/c)):
//   potencia de pesos   base · exp(c_p · tanh(a/c_p)),  c_p = ln 10 (×/÷ 10)
//   reasignar un peso   odds = odds(base) · exp(c_r · tanh(a/c_r)), c_r = 3
//   hijos sin cruce     odds = odds(base) · exp(c_m · tanh(a/c_m)), c_m = 2
// y se recorta a los rangos de Optuna ([0,5; 15], [0; 0,2] y [0,1; 0,6]), ampliados si la base cae fuera. Con a = 0 se
// obtiene exactamente la base (las intensidades de la configuración), y P₀ tal cual si la acción no mueve P.
enum class ModoAccion { P, Intensidades, Ambas };

struct MapaAccion {
    ModoAccion modo = ModoAccion::P;
    std::vector<double> P0;      // P₀ normalizada, tal cual (se usa cuando la acción no mueve P)
    std::vector<double> logp0;   // log P₀ con piso (logP0)
    double c = 3.0;
    neat::Variacion base;        // intensidades de la configuración
    double c_potencia = 2.302585092994046, c_reasignar = 3.0, c_sin_cruce = 2.0;
    double potencia_min = 0.5, potencia_max = 15.0, reasignar_max = 0.2, sin_cruce_min = 0.1, sin_cruce_max = 0.6;

    static MapaAccion crear(ModoAccion modo, const std::vector<double>& P0, double c, const neat::Variacion& base) {
        MapaAccion m;
        m.modo = modo;
        m.P0 = P0;
        m.logp0 = logP0(P0);
        m.c = c;
        m.base = base;
        return m;
    }
    int K() const { return static_cast<int>(P0.size()); }
    bool mueveP() const { return modo != ModoAccion::Intensidades; }
    bool mueveIntensidades() const { return modo != ModoAccion::P; }
    int dim() const { return (mueveP() ? K() : 0) + (mueveIntensidades() ? 3 : 0); }

    template <typename A>
    void decodificar(const A* a, std::vector<double>& P, neat::Variacion& v) const {
        for (int j = 0; j < dim(); j++)
            if (!std::isfinite(static_cast<double>(a[j]))) throw std::runtime_error("MapaAccion: acción no finita");
        int j = 0;
        if (mueveP()) {
            P = actionToP(a, logp0, c);
            j = K();
        } else {
            P = P0;
        }
        v = base;
        if (!mueveIntensidades()) return;
        const double dp = c_potencia * std::tanh(static_cast<double>(a[j]) / c_potencia);
        const double dr = c_reasignar * std::tanh(static_cast<double>(a[j + 1]) / c_reasignar);
        const double dm = c_sin_cruce * std::tanh(static_cast<double>(a[j + 2]) / c_sin_cruce);
        if (dp != 0.0)
            v.weight_perturb_power = std::clamp(base.weight_perturb_power * std::exp(dp), std::min(potencia_min, base.weight_perturb_power),
                                                std::max(potencia_max, base.weight_perturb_power));
        if (dr != 0.0) {
            const double r0 = 1.0 - base.weight_perturb_prob;
            const double r = moverProb(r0, dr, 1e-3);
            v.weight_perturb_prob = 1.0 - std::clamp(r, std::min(0.0, r0), std::max(reasignar_max, r0));
        }
        if (dm != 0.0) {
            const double m0 = base.mutation_only_fraction;
            v.mutation_only_fraction = std::clamp(moverProb(m0, dm, 1e-3), std::min(sin_cruce_min, m0), std::max(sin_cruce_max, m0));
        }
    }

private:
    // p con sus odds multiplicadas por e^d; con p = 0 parte de un piso para poder subir
    static double moverProb(double p, double d, double piso) {
        const double q = std::clamp(p, piso, 1.0 - piso);
        const double odds = q / (1.0 - q) * std::exp(d);
        return odds / (1.0 + odds);
    }
};

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
