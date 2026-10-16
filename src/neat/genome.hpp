#ifndef NEAT_GENOME_HPP
#define NEAT_GENOME_HPP

#include <memory>
#include <random>
#include <vector>
#include "config.hpp"
#include "core/network.hpp"
#include "operators.hpp"

namespace neat {

using Rng = std::mt19937_64;

class InnovationTracker;

enum class NodeRole { Input, Output, Hidden };

// Los parámetros de nodo valen por defecto lo que valía la red antes del catálogo ampliado (sin sesgo, ganancia 1,
// parámetros de la familia): con ellos el fenotipo y la distancia de compatibilidad no cambian.
struct NodeGene {
    int id;
    NodeRole role;
    NeuronType type;
    double bias = 0.0;     // B1: corriente de reposo (ocultas y salidas)
    double gain = 1.0;     // E1: multiplica la corriente de entrada (solo entradas)
    bool custom = false;   // B2: (a, b, c, d) propios en vez de los de la familia
    double abcd[4] = {0.0, 0.0, 0.0, 0.0};
    // Tasa de disparo (espigas por paso de simulación) en la última evaluación del genoma; < 0: desconocida.
    // Solo la escriben el evaluador (con NeatConfig::medir_actividad) y la usan F1–F4; los hijos la heredan del padre.
    double tasa = -1.0;
};

// Peso con signo: w >= 0 → sinapsis excitatoria, w < 0 → inhibitoria; magnitud |w|.
struct ConnGene {
    int innov;
    int in;
    int out;
    double w;
    bool enabled;
};

// Cómo nació un genoma: permite dar crédito al operador cuando se conoce su fitness.
struct Origin {
    bool has_operator = false;     // false: población inicial o élite copiada sin cambios
    Operator op = Operator::MutateWeights;
    double parent_fitness = 0.0;   // clon: fitness del padre; cruce: el mejor de los dos padres
    int last_op = -1;              // posición en el catálogo del último operador que modificó este linaje
                                   // (se conserva en las élites); lo usa el crédito "continuo" de SNAP-NEAT
    int slot = -1;                 // posición de `op` en el catálogo
    bool applied = true;           // false: el operador no pudo aplicarse (el hijo es copia del padre/cruce)
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
    // Los nodos comunes heredan todos sus parámetros de uno de los dos padres, al azar.
    static Genome crossover(const Genome& fitter, const Genome& other, const NeatConfig& cfg, Rng& rng);

    // Distancia de compatibilidad δ (ecuación 1 del paper) + c4 · diferencia media de parámetros de nodo.
    static double compatibility(const Genome& a, const Genome& b, const NeatConfig& cfg);

    // Operadores. Devuelven false si no pudieron aplicarse (el genoma queda igual).
    // `apply` cubre todos menos WeightsDirected, que necesita otro genoma (mutateDirected).
    bool apply(Operator op, const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    void mutateWeights(const NeatConfig& cfg, Rng& rng);
    bool mutateAddConnection(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    bool mutateAddNode(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    bool mutateWeightsFine(const NeatConfig& cfg, Rng& rng);
    bool mutateWeightsReset(const NeatConfig& cfg, Rng& rng);
    bool mutateDirected(const Genome& partner, const NeatConfig& cfg, Rng& rng);
    bool mutateNeuronBias(const NeatConfig& cfg, Rng& rng);
    bool mutateNeuronDynamics(const NeatConfig& cfg, Rng& rng);
    bool mutateNeuronFamily(const NeatConfig& cfg, Rng& rng);
    bool mutateInputGain(const NeatConfig& cfg, Rng& rng);
    bool mutateAddNodeFamily(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    bool mutateAddNodeMulti(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    bool mutateDuplicateNode(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    bool mutateDeleteConnection(const NeatConfig& cfg, Rng& rng);
    bool mutateDeleteSimpleNode(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    bool mutateWakeNeuron(NodeRole role, const NeatConfig& cfg, Rng& rng);   // F1 (salidas) y F2 (ocultas)
    bool mutateCalmSaturated(const NeatConfig& cfg, Rng& rng);
    bool mutateGainMuteInputs(const NeatConfig& cfg, Rng& rng);
    void escalarEntradasExcitatorias(int id, double factor, const NeatConfig& cfg);

    // Escribe la tasa de cada neurona desde las espigas contadas en la red (índices de buildNetwork) y los pasos
    void asignarTasas(const std::vector<long>& espigas, long pasos);

    // Fenotipo: red de espigas lista para simular (entradas, salidas y ocultas en ese orden).
    std::unique_ptr<Network> buildNetwork(const NeatConfig& cfg) const;

    // Ganancias de las entradas en el orden de las neuronas de entrada; vacío si todas valen 1
    // (el evaluador entonces no multiplica, y el resultado es idéntico al de antes del catálogo ampliado).
    std::vector<double> inputGains() const;

    const std::vector<NodeGene>& nodes() const { return nodes_; }
    const std::vector<ConnGene>& conns() const { return conns_; }
    int numInputs() const { return num_inputs_; }
    int numOutputs() const { return num_outputs_; }
    int numHidden() const;
    int numEnabledConns() const;
    bool hasConnection(int in, int out) const;
    const NodeGene* findNode(int id) const;

    // Invariantes: genes ordenados, sin duplicados, extremos existentes. Para tests.
    bool isValid() const;

private:
    int num_inputs_ = 0;
    int num_outputs_ = 0;
    std::vector<NodeGene> nodes_;   // ordenados por id
    std::vector<ConnGene> conns_;   // ordenados por innov

    void addNodeSorted(const NodeGene& n);
    void addConnSorted(const ConnGene& c);
    bool addNodeOnRandomConnection(NeuronType type, const NeatConfig& cfg, InnovationTracker& innov, Rng& rng);
    void removeNode(int id);   // borra el nodo y todos los genes de conexión que lo tocan
};

}  // namespace neat

#endif
