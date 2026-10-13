#ifndef AOS_ADAPTIVE_PURSUIT_HPP
#define AOS_ADAPTIVE_PURSUIT_HPP

#include <vector>
#include "neat/selector.hpp"

namespace aos {

// Línea base 2: AOS sin estado con Adaptive Pursuit (Thierens 2005).
//
// Por generación, con R_i el crédito del operador i (solo si se usó):
//   Q_i ← Q_i + α (R_i − Q_i)
//   i* = argmax Q
//   P_i* ← P_i* + β (P_max − P_i*),   P_j ← P_j + β (P_min − P_j)  (j ≠ i*)
//   con P_max = 1 − (K−1)·P_min
//
// Crédito por hijo (docs/diseno_neat_ap.md §7):
//   r = max(0, f_hijo − f_padre) / (f_mejor − f_peor + ε)   (mejora normalizada por el rango
//   de la generación, para no depender de la escala del entorno)
// y R_i = media (o máximo) de los r de los hijos que produjo el operador i.
struct AdaptivePursuitConfig {
    double alpha = 0.8;        // tasa de adaptación de Q  [verificar valores de Thierens 2005]
    double beta = 0.8;         // tasa de persecución de P [verificar valores de Thierens 2005]
    double p_min = 0.02;       // cota inferior; debe cumplir K·p_min < 1
    double q_init = 0.0;       // estimación inicial de calidad
    bool extreme_credit = false;   // true: R_i = máximo de r (crédito extremo); false: media
    // P inicial. Vacío → uniforme 1/K (Thierens). Los experimentos lo inician desde el P
    // estático de Stanley, para que la única diferencia con la línea base 1 sea la adaptación.
    std::vector<double> p_init;
    // K: operadores del catálogo (si p_init no está vacío, K = su tamaño)
    int num_operators = neat::kNumOperators;

    // ── Modo SNAP-NEAT (Kohl & Miikkulainen 2012, §III-B/C y Algoritmo 1) ──
    // Recompensa = fitness normalizado del individuo, (f − f_peor)/(f_mejor − f_peor), atribuido al
    // ÚLTIMO operador de su linaje (también en las élites reevaluadas: "actualizaciones continuas").
    // Q y P se actualizan después de CADA evaluación (secuencial), no una vez por generación.
    // Período inicial: las primeras `init_generations` generaciones con P uniforme; al terminar,
    // Q_i = media + 1 desviación estándar de las recompensas acumuladas del operador i.
    // Valores del paper: α = β = 0,05; P_min = 0,10; período inicial = 1/5 de las evaluaciones.
    bool snap = false;
    int init_generations = 0;
};

class AdaptivePursuitSelector : public neat::OperatorSelector {
public:
    explicit AdaptivePursuitSelector(const AdaptivePursuitConfig& cfg = AdaptivePursuitConfig());

    std::vector<double> probabilities() const override { return p_; }
    void update(const neat::GenerationReport& report) override;
    std::string name() const override { return "adaptive_pursuit"; }

    const std::vector<double>& quality() const { return q_; }
    const std::vector<double>& lastCredit() const { return last_credit_; }   // R_i (NaN si no se usó)

    // Crédito de un hijo (expuesto para tests)
    static double childReward(const neat::OperatorOutcome& o, double f_best, double f_worst);

private:
    AdaptivePursuitConfig cfg_;
    int K_;
    std::vector<double> p_, q_, last_credit_;
    int generations_seen_ = 0;
    std::vector<std::vector<double>> init_rewards_;   // modo SNAP: recompensas del período inicial
    void pursue(int star);
    void updateSnap(const neat::GenerationReport& report);
};

}  // namespace aos

#endif
