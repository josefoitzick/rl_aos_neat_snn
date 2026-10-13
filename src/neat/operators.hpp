#ifndef NEAT_OPERATORS_HPP
#define NEAT_OPERATORS_HPP

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace neat {

// Operadores de mutación. Los tres primeros son los de NEAT clásico (Stanley 2002 §3.1) y forman el
// catálogo por defecto; el resto son los candidatos de docs/operadores_candidatos.md (lista corta + variantes
// a tamizar). Un experimento elige su catálogo (subconjunto ordenado); P y los selectores trabajan sobre las
// posiciones del catálogo ("slots"), no sobre estos valores.
enum class Operator {
    MutateWeights = 0,   // A1: perturba todas las conexiones (y reasigna con p = 1 − weight_perturb_prob)
    AddConnection = 1,   // C1
    AddNode = 2,         // C3
    WeightsFine,         // A2: perturbación gaussiana chica de 1–3 conexiones (SharpNEAT)
    WeightsReset,        // A3: valor nuevo para 1–3 conexiones
    WeightsDirected,     // A7: Iso+LineDD sobre genes coincidentes con otro miembro de la especie
    NeuronBias,          // B1: corriente de reposo de una neurona oculta o de salida
    NeuronDynamics,      // B2: perturbación acotada de (a, b, c, d) de Izhikevich
    NeuronFamily,        // B3: cambio de familia de Izhikevich (RS, FS, IB, CH)
    InputGain,           // E1: ganancia de una neurona de entrada
    AddNodeFamily,       // C4: agregar nodo de una familia al azar
    AddNodeMulti,        // C9: agregar k nodos de una vez
    DuplicateNode,       // C5: duplicar una neurona oculta repartiendo sus salidas (Net2WiderNet)
    DeleteConnection,    // D1: borrar una conexión (y las ocultas que queden sin genes)
    DeleteSimpleNode,    // D2: borrar una oculta de 1 entrada y 1 salida (con puente) o sin salida/entrada
};

constexpr int kNumOperators = 3;        // tamaño del catálogo por defecto (NEAT clásico)
constexpr int kNumOperatorTypes = 15;   // todos los operadores implementados

inline const char* operatorName(Operator op) {
    switch (op) {
        case Operator::MutateWeights:    return "mutate_weights";
        case Operator::AddConnection:    return "add_connection";
        case Operator::AddNode:          return "add_node";
        case Operator::WeightsFine:      return "weights_fine";
        case Operator::WeightsReset:     return "weights_reset";
        case Operator::WeightsDirected:  return "weights_directed";
        case Operator::NeuronBias:       return "neuron_bias";
        case Operator::NeuronDynamics:   return "neuron_dynamics";
        case Operator::NeuronFamily:     return "neuron_family";
        case Operator::InputGain:        return "input_gain";
        case Operator::AddNodeFamily:    return "add_node_family";
        case Operator::AddNodeMulti:     return "add_node_multi";
        case Operator::DuplicateNode:    return "duplicate_node";
        case Operator::DeleteConnection: return "delete_connection";
        case Operator::DeleteSimpleNode: return "delete_simple_node";
    }
    return "?";
}

inline std::vector<Operator> defaultCatalog() {
    return {Operator::MutateWeights, Operator::AddConnection, Operator::AddNode};
}

// "mutate_weights,add_node,..." → catálogo (sin repetidos)
inline std::vector<Operator> parseCatalog(const std::string& txt) {
    std::vector<Operator> cat;
    std::stringstream ss(txt);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        bool found = false;
        for (int k = 0; k < kNumOperatorTypes; k++) {
            Operator op = static_cast<Operator>(k);
            if (tok == operatorName(op)) {
                for (Operator o : cat)
                    if (o == op) throw std::invalid_argument("catálogo: operador repetido '" + tok + "'");
                cat.push_back(op);
                found = true;
                break;
            }
        }
        if (!found) throw std::invalid_argument("catálogo: operador desconocido '" + tok + "'");
    }
    if (cat.empty()) throw std::invalid_argument("catálogo vacío");
    return cat;
}

}  // namespace neat

#endif
