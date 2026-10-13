#ifndef NEAT_SELECTOR_HPP
#define NEAT_SELECTOR_HPP

#include <string>
#include <vector>
#include "genome.hpp"

namespace neat {

// Resultado de aplicar un operador: el hijo ya evaluado y la referencia de su(s) padre(s).
// `slot` es la posición del operador en el catálogo (−1: se usa el valor del enum, que coincide con la
// posición en el catálogo por defecto).
struct OperatorOutcome {
    Operator op;
    double child_fitness;
    double parent_fitness;
    int slot = -1;
    bool applied = true;   // false: el operador no pudo aplicarse
    int slotIndex() const { return slot >= 0 ? slot : static_cast<int>(op); }
};

// (último operador del linaje, fitness): crédito "continuo" de SNAP-NEAT
struct LastOpScore {
    Operator op;
    double fitness;
    int slot = -1;
    int slotIndex() const { return slot >= 0 ? slot : static_cast<int>(op); }
};

struct PopulationStats {
    int generation = 0;
    double best = 0.0, mean = 0.0, stdev = 0.0, worst = 0.0;
    int num_species = 0;
    double compat_threshold = 0.0;   // δt usado en esta especiación
    int champion_hidden = 0, champion_conns = 0;
    double mean_hidden = 0.0, mean_conns = 0.0;
};

// Lo que ve el selector al cerrar una generación (tras evaluar).
// Para el selector basado en estado (PPO), `stats` es la materia prima del estado.
struct GenerationReport {
    PopulationStats stats;
    std::vector<OperatorOutcome> outcomes;
    // (último operador del linaje, fitness) para TODOS los genomas con historia, incluidas las
    // élites reevaluadas: crédito "continuo" de SNAP-NEAT (Kohl & Miikkulainen 2012 §III-B)
    std::vector<LastOpScore> last_op_scores;
};

// Punto de enganche del AOS: NEAT solo pide el vector P y reporta resultados.
// Las tres configuraciones de la tesis (estática, Adaptive Pursuit, PPO) son
// implementaciones de esta interfaz sobre el mismo NEAT.
class OperatorSelector {
public:
    virtual ~OperatorSelector() = default;
    virtual std::vector<double> probabilities() const = 0;   // P sobre el catálogo (tamaño K), suma 1
    virtual void update(const GenerationReport& report) = 0;
    virtual std::string name() const = 0;
};

// Línea base 1: probabilidades fijas.
class StaticSelector : public OperatorSelector {
public:
    explicit StaticSelector(std::vector<double> weights);   // uno por operador del catálogo; se normalizan a suma 1

    // Stanley & Miikkulainen (2002) §4.1: pesos 0,8 · agregar conexión 0,05 · agregar nodo 0,03,
    // normalizados como distribución categórica (un operador por hijo).
    static StaticSelector stanley2002();

    std::vector<double> probabilities() const override { return p_; }
    void update(const GenerationReport&) override {}
    std::string name() const override { return "static"; }

private:
    std::vector<double> p_;
};

}  // namespace neat

#endif
