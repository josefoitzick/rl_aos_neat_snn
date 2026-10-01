#ifndef NEAT_POPULATION_HPP
#define NEAT_POPULATION_HPP

#include <cstdint>
#include <vector>
#include "config.hpp"
#include "genome.hpp"
#include "innovation.hpp"
#include "selector.hpp"

namespace neat {

struct Species {
    int id;
    Genome representative;         // miembro al azar de la generación anterior
    std::vector<int> members;      // índices en la población actual
    double best_fitness;           // mejor fitness histórico de la especie
    int last_improved;             // generación de la última mejora
};

// Ciclo por generación:
//   1. el llamador evalúa `genomes()` y asigna `fitness`
//   2. report = endGeneration()          (especiación, estancamiento, estadísticas, resultados)
//   3. selector.update(report)
//   4. reproduce(selector.probabilities())
class Population {
public:
    Population(const NeatConfig& cfg, int num_inputs, int num_outputs, uint64_t seed);

    std::vector<Genome>& genomes() { return genomes_; }
    const std::vector<Genome>& genomes() const { return genomes_; }

    GenerationReport endGeneration();
    void reproduce(const std::vector<double>& op_probs);

    int generation() const { return generation_; }
    const Genome& bestEver() const { return best_ever_; }
    const std::vector<Species>& species() const { return species_; }
    double compatThreshold() const { return compat_threshold_; }

private:
    NeatConfig cfg_;
    Rng rng_;
    InnovationTracker innov_;
    std::vector<Genome> genomes_;
    std::vector<Species> species_;
    int generation_ = 0;
    int next_genome_id_ = 0;
    int next_species_id_ = 0;
    bool ended_ = false;
    double compat_threshold_;   // δt actual (dinámico si cfg.target_species > 0)

    Genome best_ever_;
    double best_ever_fitness_;
    int last_global_improvement_ = 0;

    void speciate();
    void adaptThreshold();
    std::vector<int> allocateOffspring();
};

}  // namespace neat

#endif
