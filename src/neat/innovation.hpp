#ifndef NEAT_INNOVATION_HPP
#define NEAT_INNOVATION_HPP

#include <map>
#include <utility>

namespace neat {

// Marcas históricas (Stanley 2002 §3.2). Una misma innovación estructural ocurrida
// más de una vez en la misma generación recibe el mismo número.
class InnovationTracker {
public:
    explicit InnovationTracker(int first_hidden_node_id = 0) : next_node_id_(first_hidden_node_id) {}

    // Innovación de la conexión in→out
    int connection(int in, int out);

    // Nodo que divide la conexión `split_innov`, y las innovaciones de in→nodo y nodo→out
    struct Split { int node_id; int innov_in; int innov_out; };
    Split split(int split_innov, int in, int out);

    // Olvida las innovaciones de la generación actual (los contadores siguen)
    void newGeneration() { conn_this_gen_.clear(); split_this_gen_.clear(); }

    int nextInnovation() const { return next_innov_; }
    int nextNodeId() const { return next_node_id_; }

private:
    int next_innov_ = 0;
    int next_node_id_;
    std::map<std::pair<int, int>, int> conn_this_gen_;
    std::map<int, Split> split_this_gen_;
};

}  // namespace neat

#endif
