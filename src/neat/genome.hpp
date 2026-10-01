#ifndef NEAT_GENOME_HPP
#define NEAT_GENOME_HPP

#include <memory>
#include <random>
#include <vector>
#include "config.hpp"
#include "core/network.hpp"

namespace neat {

using Rng = std::mt19937_64;

class InnovationTracker;

enum class NodeRole { Input, Output, Hidden };

struct NodeGene {
    int id;
    NodeRole role;
    NeuronType type;
};

// Peso con signo: w >= 0 → sinapsis excitatoria, w < 0 → inhibitoria; magnitud |w|.
struct ConnGene {
    int innov;
    int in;
    int out;
    double w;
    bool enabled;
};

// Operadores de mutación del catálogo (NEAT clásico, Stanley 2002 §3.1).
enum class Operator { MutateWeights = 0, AddConnection = 1, AddNode = 2 };
constexpr int kNumOperators = 3;
const char* operatorName(Operator op);

// Cómo nació un genoma: permite dar crédito al operador cuando se conoce su fitness.
struct Origin {
    bool has_operator = false;     // false: población inicial o élite copiada sin cambios
    Operator op = Operator::MutateWeights;
    double parent_fitness = 0.0;   // clon: fitness del padre; cruce: el mejor de los dos padres
    int last_op = -1;              // último operador que modificó este linaje (se conserva en las
                                   // élites); lo usa el crédito "continuo" de SNAP-NEAT
};

class Genome {
public:
    int id = -1;
    double fitness = 0.0;
    Origin origin;

    // Topología mínima: cada entrada conectada a cada salida, pesos ~ U(-init, init).
    static Genome minimal(int id, int num_inputs, int num_outputs, const NeatConfig& cfg,
                          InnovationTracker& innov, Rng& rng);

    // Cruce (Stanley 2002 §3.2): homólogos al azar; disjuntos y exceso del padre más apto.
    static Genome crossover(const Genome& fitter, const Genome& other, const NeatConfig& cfg, Rng& rng);

    // Distancia de compatibilidad δ (ecuación 1 del paper).
    static double compatibility(const Genome& a, const Genome& b, const NeatConfig& cfg);

    // Operadores. Devuelven false si no pudieron aplicarse (el genoma queda igual).
    bool apply(Operator op, const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    void mutateWeights(const NeatConfig& cfg, Rng& rng);
    bool mutateAddConnection(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    bool mutateAddNode(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);

    // Fenotipo: red de espigas lista para simular (entradas, salidas y ocultas en ese orden).
    std::unique_ptr<Network> buildNetwork(const NeatConfig& cfg) const;

    const std::vector<NodeGene>& nodes() const { return nodes_; }
    const std::vector<ConnGene>& conns() const { return conns_; }
    int numInputs() const { return num_inputs_; }
    int numOutputs() const { return num_outputs_; }
    int numHidden() const;
    int numEnabledConns() const;
    bool hasConnection(int in, int out) const;

    // Invariantes: genes ordenados, sin duplicados, extremos existentes. Para tests.
    bool isValid() const;

private:
    int num_inputs_ = 0;
    int num_outputs_ = 0;
    std::vector<NodeGene> nodes_;   // ordenados por id
    std::vector<ConnGene> conns_;   // ordenados por innov

    const NodeGene* findNode(int id) const;
    void addNodeSorted(const NodeGene& n);
    void addConnSorted(const ConnGene& c);
};

}  // namespace neat

#endif
