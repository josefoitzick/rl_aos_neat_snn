#ifndef EXPERIMENT_AOS_TOY_HPP
#define EXPERIMENT_AOS_TOY_HPP

// AOS sintético ("NEAT de juguete") para validar y calibrar el entrenador PPO sin NEAT
// (docs/diseno_ppo.md §5, paso V3). Corre en microsegundos y se conoce la respuesta correcta.
//
// Una corrida = T decisiones de k generaciones. Estado oculto de cada corrida:
//   q  calidad de la población, en unidades de score;
//   c  complejidad media ("neuronas ocultas").
// Parámetros por corrida, sacados de la semilla: calidad máxima, complejidad óptima c* (no observable) y
// tasas de avance. Operadores (K = 8), con un efecto inmediato sobre los hijos (lo que mide el crédito
// por rango) y un efecto sobre la dinámica:
//   0 pesos        acerca q a la capacidad; rinde más lejos de ella (temprano)
//   1 pesos finos  acerca q a la capacidad; rinde más cerca de ella (tarde)
//   2 + conexión   sube c un poco; el hijo apenas empeora
//   3 + neurona    sube c bastante; el hijo empeora (disrupción)
//   4 − conexión   baja c un poco; el hijo mejora si c > c* y empeora si c < c*
//   5 − neurona    baja c bastante, igual que 4; falla sin ocultas (c < 1)
//   6 neutro       nada
//   7 dañino       empeora q
// La capacidad crece con c y se satura; sobre c* el exceso (hinchazón) frena el progreso y baja la
// capacidad. Así, la mejor política depende del estado (crecer hasta c*, después afinar; podar si se pasa)
// y del tiempo (pesos finos al final), y ninguna P fija la iguala. Con `con_estado = false` la capacidad no
// depende de c ni del tiempo y el óptimo es una P fija: sirve de control (PPO no debe empeorar a P₀).
//
// Campeón y recompensa como en NEAT con rotación: cada generación se valida la población con ruido; el
// campeón es el de mejor validación (con sesgo del ganador) y su calidad verdadera es el score final. Φ es
// esa calidad medida con poco ruido (el conjunto de recompensa, stream 3) y se recalcula cuando cambia el
// campeón. La observación tiene los 14 + 2K rasgos de §2.3, algunos informativos y otros puro ruido.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace aos::toy {

constexpr int K = 8;
constexpr int OBS = 14 + 2 * K;

enum Op { W = 0, WF, GC, GN, DC, DN, NEU, DAN };

struct Config {
    int T = 50;      // decisiones por corrida
    int k = 10;      // generaciones por decisión
    int pop = 100;   // hijos por generación (fija el ruido del crédito)
    bool con_estado = true;
    double sd_q = 0.002, sd_c = 0.02, sd_val = 0.015, sd_phi = 0.005;
    double cap_mean = 0.80, cap_sd = 0.03;
    double cstar_lo = 3.0, cstar_hi = 7.0;
    double rate_sd = 0.2;     // desviación del logaritmo de las tasas por corrida
    double sd_credit = 0.15;  // ruido del crédito con un solo hijo
};

class Run {
public:
    void reset(uint64_t seed, const Config& cfg) {
        cfg_ = cfg;
        rng_.seed(seed);
        std::normal_distribution<double> n01(0.0, 1.0);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        cap_max_ = cfg_.cap_mean + cfg_.cap_sd * n01(rng_);
        cstar_ = cfg_.cstar_lo + (cfg_.cstar_hi - cfg_.cstar_lo) * u01(rng_);
        rw_ = std::exp(cfg_.rate_sd * n01(rng_));
        rg_ = std::exp(cfg_.rate_sd * n01(rng_));
        q_ = 0.05 + 0.03 * u01(rng_);
        c_ = 0.0;
        G_ = cfg_.T * cfg_.k + 1;
        gen_ = 0;
        t_ = 0;
        credit_.fill(0.0);
        fail_.fill(0.0);
        // generación 0: población inicial evaluada y validada
        const double v = q_ + cfg_.sd_val * n01(rng_);
        vbest_ = v;
        qchamp_ = q_;
        cchamp_ = c_;
        tau_imp_ = 0.0;
        phi_ = qchamp_ + cfg_.sd_phi * n01(rng_);
        vbest_window_start_ = vbest_;
        vsum_ = v;
        vcount_ = 1;
        windowObservation();
    }

    // Una decisión: k generaciones con la misma P
    void advance(const std::vector<double>& P) {
        if (static_cast<int>(P.size()) != K) throw std::invalid_argument("toy::Run::advance: P de tamaño distinto de K");
        if (t_ >= cfg_.T) throw std::logic_error("toy::Run::advance: la corrida ya terminó");
        vbest_window_start_ = vbest_;
        vsum_ = 0.0;
        vcount_ = 0;
        champ_changed_ = false;
        std::array<double, K> acc_e{}, acc_n{}, acc_fail{};
        for (int g = 0; g < cfg_.k; g++) generation(P, acc_e, acc_n, acc_fail);
        // crédito por rango observado: media móvil (η = 0,5 por ventana) de una medida ruidosa del efecto
        // inmediato; solo se actualiza si el operador tuvo al menos un hijo en la ventana
        std::normal_distribution<double> n01(0.0, 1.0);
        for (int i = 0; i < K; i++) {
            if (acc_n[i] < 1.0) continue;
            const double m = 0.5 * std::tanh((acc_e[i] / acc_n[i]) / 0.002) + cfg_.sd_credit / std::sqrt(acc_n[i]) * n01(rng_);
            credit_[i] = std::clamp(0.5 * credit_[i] + 0.5 * m, -0.5, 0.5);
            fail_[i] = 0.5 * fail_[i] + 0.5 * (acc_fail[i] / acc_n[i]);
        }
        if (champ_changed_) phi_ = qchamp_ + cfg_.sd_phi * n01(rng_);
        t_++;
        windowObservation();
    }

    const std::array<float, OBS>& obs() const { return obs_; }
    double phi() const { return phi_; }
    double finalScore() const { return qchamp_; }   // calidad verdadera del campeón
    int t() const { return t_; }
    int T() const { return cfg_.T; }
    bool done() const { return t_ >= cfg_.T; }
    // estado oculto, solo para la política oráculo de la calibración
    double q() const { return q_; }
    double c() const { return c_; }
    double cstar() const { return cstar_; }
    double capacity() const { return cap(c_); }

private:
    Config cfg_;
    std::mt19937_64 rng_;
    double cap_max_ = 0, cstar_ = 0, rw_ = 1, rg_ = 1;
    double q_ = 0, c_ = 0, qchamp_ = 0, cchamp_ = 0, vbest_ = 0, phi_ = 0, tau_imp_ = 0;
    double vbest_window_start_ = 0, vsum_ = 0;
    int vcount_ = 0, G_ = 1, gen_ = 0, t_ = 0;
    bool champ_changed_ = false;
    std::array<double, K> credit_{}, fail_{};
    std::array<float, OBS> obs_{};

    double tau(int g) const { return static_cast<double>(g) / (G_ - 1); }
    double cap(double c) const {
        if (!cfg_.con_estado) return cap_max_;
        return cap_max_ * (0.3 + 0.7 * (1.0 - std::exp(-c / 2.0))) - 0.02 * std::max(0.0, c - cstar_);
    }
    double bloat(double c) const { return cfg_.con_estado ? std::exp(-std::max(0.0, c - cstar_) / 2.0) : 1.0; }

    // Efecto inmediato de un uso de cada operador sobre el hijo (lo que ve el crédito por rango)
    std::array<double, K> immediate() const {
        const double cp = cap(c_);
        const double gap = std::max(0.0, cp - q_);
        const double qq = std::clamp(q_ / std::max(cp, 1e-6), 0.0, 1.0);
        const double h = bloat(c_);
        const double fine = cfg_.con_estado ? 0.5 + 2.0 * qq * qq : 0.5;
        const double prune = cfg_.con_estado ? 0.002 * std::tanh(c_ - cstar_) : 0.0;
        std::array<double, K> e{};
        e[W] = 0.012 * rw_ * h * gap;
        e[WF] = 0.012 * rw_ * h * gap * fine * 0.5;
        e[GC] = -0.0005;
        e[GN] = -0.003 * q_;
        e[DC] = prune;
        e[DN] = 2.0 * prune * std::min(1.0, c_);
        e[NEU] = 0.0;
        e[DAN] = -0.004 * q_;
        return e;
    }

    void generation(const std::vector<double>& P, std::array<double, K>& acc_e, std::array<double, K>& acc_n,
                    std::array<double, K>& acc_fail) {
        std::normal_distribution<double> n01(0.0, 1.0);
        const auto e = immediate();
        const double dn_ok = std::min(1.0, c_);
        for (int i = 0; i < K; i++) {
            const double n = P[i] * cfg_.pop;
            acc_e[i] += e[i] * n;
            acc_n[i] += n;
        }
        acc_fail[DN] += P[DN] * cfg_.pop * (1.0 - dn_ok);
        const double dq = P[W] * e[W] + P[WF] * e[WF] + P[GC] * e[GC] + P[GN] * e[GN] + P[DAN] * e[DAN];
        const double dc = rg_ * (0.04 * P[GC] + 0.10 * P[GN]) - 0.04 * P[DC] - 0.10 * P[DN] * dn_ok;
        c_ = std::max(0.0, c_ + dc + cfg_.sd_c * n01(rng_));
        q_ += dq + cfg_.sd_q * n01(rng_);
        const double cp = cap(c_);
        if (q_ > cp) q_ -= 0.05 * (q_ - cp);
        q_ = std::max(0.0, q_);
        gen_++;
        const double v = q_ + cfg_.sd_val * n01(rng_);
        vsum_ += v;
        vcount_++;
        if (v > vbest_) {
            vbest_ = v;
            qchamp_ = q_;
            cchamp_ = c_;
            tau_imp_ = tau(gen_);
            champ_changed_ = true;
        }
    }

    // Los 14 + 2K rasgos de docs/diseno_ppo.md §2.3, en su versión de juguete
    void windowObservation() {
        std::normal_distribution<double> n01(0.0, 1.0);
        const double tnow = tau(gen_);
        const double cp = cap(c_);
        const double gap = std::max(0.0, cp - q_);
        const double qq = std::clamp(q_ / std::max(cp, 1e-6), 0.0, 1.0);
        const double vgen = vsum_ / std::max(1, vcount_);
        int i = 0;
        obs_[i++] = static_cast<float>(tnow);                                   // τ
        obs_[i++] = static_cast<float>(vbest_);                                 // V*
        obs_[i++] = static_cast<float>(vgen);                                   // V_gen
        obs_[i++] = static_cast<float>(vbest_ - vgen);                          // brecha
        obs_[i++] = static_cast<float>(vbest_ - vbest_window_start_);           // ΔV*
        obs_[i++] = static_cast<float>(tnow - tau_imp_);                        // estancamiento
        obs_[i++] = static_cast<float>(0.05 + 0.3 * gap / cfg_.cap_mean * (1.0 + 0.2 * n01(rng_)));  // dispersión
        obs_[i++] = static_cast<float>(1.0 + 0.1 * n01(rng_));                  // especies / objetivo (ruido)
        obs_[i++] = static_cast<float>(0.3 + 0.3 * qq + 0.05 * n01(rng_));      // especie mayor
        obs_[i++] = static_cast<float>(0.1 * c_ + 0.2 * n01(rng_));             // log δt
        obs_[i++] = static_cast<float>(std::log1p(c_));                         // conexiones
        obs_[i++] = static_cast<float>(std::log1p(0.5 * c_) + 0.05 * n01(rng_));      // ocultas
        obs_[i++] = static_cast<float>(std::log1p(cchamp_));                    // conexiones del campeón
        obs_[i++] = static_cast<float>(std::log1p(0.5 * cchamp_) + 0.05 * n01(rng_)); // ocultas del campeón
        for (int k = 0; k < K; k++) obs_[i++] = static_cast<float>(credit_[k]);
        for (int k = 0; k < K; k++) obs_[i++] = static_cast<float>(fail_[k]);
    }
};

// Política oráculo (usa el estado oculto): crecer hasta c*, podar si se pasa y afinar al final.
// Solo sirve para medir cuánto se puede ganar sobre la mejor P fija.
inline std::vector<double> oraclePolicy(const Run& r, const Config& cfg) {
    std::vector<double> P(K, 0.0);
    if (!cfg.con_estado) {
        P[W] = 1.0;
        return P;
    }
    if (r.c() < r.cstar() - 0.7) {
        P[GN] = 0.45; P[GC] = 0.15; P[W] = 0.40;
    } else if (r.c() > r.cstar() + 0.7) {
        P[DN] = 0.25; P[DC] = 0.15; P[W] = 0.60;
    } else if (r.q() / std::max(r.capacity(), 1e-6) < 0.85) {
        P[W] = 0.9; P[WF] = 0.1;
    } else {
        P[W] = 0.3; P[WF] = 0.7;
    }
    return P;
}

// Varias corridas en ranuras, con la interfaz que espera el lote de RLtools (src/experiment/ppo_lote.hpp):
// start, advance, obs, phi, t, T y reward. Lleva la contabilidad para la verificación V2: cada corrida
// recibe exactamente T decisiones y ninguna semilla se repite.
class Pool {
public:
    struct Finished {
        uint64_t seed;
        double final_score;
        double ret;   // suma de recompensas
    };

    Pool(const Config& cfg, int slots, uint64_t seed_base, double w_area, double reward_scale)
        : cfg_(cfg), runs_(slots), seeds_(slots, 0), ret_(slots, 0.0), started_(slots, false),
          next_seed_(seed_base), w_area_(w_area), scale_(reward_scale) {}

    int T() const { return cfg_.T; }
    int slots() const { return static_cast<int>(runs_.size()); }

    void start(const std::vector<int>& slots) {
        for (int s : slots) {
            if (started_[s] && !runs_[s].done())
                throw std::logic_error("toy::Pool::start: se reinicia una corrida que no terminó");
            seeds_[s] = next_seed_++;
            runs_[s].reset(seeds_[s], cfg_);
            ret_[s] = 0.0;
            started_[s] = true;
            n_started_++;
        }
    }

    void advance(const std::vector<std::vector<double>>& P) {
        if (static_cast<int>(P.size()) != slots()) throw std::invalid_argument("toy::Pool::advance: falta P por ranura");
        for (int s = 0; s < slots(); s++) {
            if (!started_[s]) throw std::logic_error("toy::Pool::advance: ranura sin corrida");
            const double phi0 = runs_[s].phi();
            runs_[s].advance(P[s]);
            ret_[s] += reward(phi0, runs_[s].phi());
            for (int k = 0; k < K; k++) p_sum_[k] += P[s][k];
            p_count_++;
            n_decisions_++;
            if (runs_[s].done()) {
                finished_.push_back({seeds_[s], runs_[s].finalScore(), ret_[s]});
                n_finished_++;
            }
        }
    }

    // r_t = escala · [(Φ_{t+1} − Φ_t) + (w/T)·Φ_{t+1}]  (docs/diseno_ppo.md §2.4)
    double reward(double phi_prev, double phi_next) const {
        return scale_ * ((phi_next - phi_prev) + w_area_ / cfg_.T * phi_next);
    }

    const std::array<float, OBS>& obs(int s) const { return runs_[s].obs(); }
    double phi(int s) const { return runs_[s].phi(); }
    int t(int s) const { return runs_[s].t(); }
    const Run& run(int s) const { return runs_[s]; }

    // Registro: corridas terminadas y P media desde el último vaciado
    std::vector<Finished> takeFinished() {
        auto f = std::move(finished_);
        finished_.clear();
        return f;
    }
    std::array<double, K> takeMeanP() {
        std::array<double, K> m{};
        for (int k = 0; k < K; k++) m[k] = p_count_ ? p_sum_[k] / p_count_ : 0.0;
        p_sum_.fill(0.0);
        p_count_ = 0;
        return m;
    }
    long started() const { return n_started_; }
    long finishedCount() const { return n_finished_; }
    long decisions() const { return n_decisions_; }
    uint64_t nextSeed() const { return next_seed_; }

private:
    Config cfg_;
    std::vector<Run> runs_;
    std::vector<uint64_t> seeds_;
    std::vector<double> ret_;
    std::vector<bool> started_;
    std::vector<Finished> finished_;
    std::array<double, K> p_sum_{};
    long p_count_ = 0, n_started_ = 0, n_finished_ = 0, n_decisions_ = 0;
    uint64_t next_seed_;
    double w_area_, scale_;
};

// Score final medio de una P fija en `n` corridas con semillas seed_base, seed_base + 1, …
inline std::vector<double> evalStatic(const std::vector<double>& P, const Config& cfg, uint64_t seed_base, int n) {
    std::vector<double> out(n);
    Run r;
    for (int i = 0; i < n; i++) {
        r.reset(seed_base + i, cfg);
        while (!r.done()) r.advance(P);
        out[i] = r.finalScore();
    }
    return out;
}

inline std::vector<double> evalOracle(const Config& cfg, uint64_t seed_base, int n) {
    std::vector<double> out(n);
    Run r;
    for (int i = 0; i < n; i++) {
        r.reset(seed_base + i, cfg);
        while (!r.done()) r.advance(oraclePolicy(r, cfg));
        out[i] = r.finalScore();
    }
    return out;
}

}  // namespace aos::toy

#endif
