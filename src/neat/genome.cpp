#include "genome.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include "innovation.hpp"

namespace neat {

const char* operatorName(Operator op) {
    switch (op) {
        case Operator::MutateWeights: return "mutate_weights";
        case Operator::AddConnection: return "add_connection";
        case Operator::AddNode:       return "add_node";
    }
    return "?";
}

static double uniform(Rng& rng, double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
}

static double clampWeight(double w, const NeatConfig& cfg) {
    return std::clamp(w, -cfg.weight_max, cfg.weight_max);
}

// ───────────────────────────── construcción ─────────────────────────────

Genome Genome::minimal(int id, int num_inputs, int num_outputs, const NeatConfig& cfg,
                       InnovationTracker& innov, Rng& rng) {
    Genome g;
    g.id = id;
    g.num_inputs_ = num_inputs;
    g.num_outputs_ = num_outputs;
    for (int i = 0; i < num_inputs; i++)
        g.nodes_.push_back({i, NodeRole::Input, cfg.neuron_type});
    for (int o = 0; o < num_outputs; o++)
        g.nodes_.push_back({num_inputs + o, NodeRole::Output, cfg.neuron_type});
    for (int i = 0; i < num_inputs; i++)
        for (int o = 0; o < num_outputs; o++) {
            int out = num_inputs + o;
            g.addConnSorted({innov.connection(i, out), i, out,
                             uniform(rng, -cfg.weight_init_range, cfg.weight_init_range), true});
        }
    return g;
}

Genome Genome::crossover(const Genome& fitter, const Genome& other, const NeatConfig& cfg, Rng& rng) {
    Genome child;
    child.num_inputs_ = fitter.num_inputs_;
    child.num_outputs_ = fitter.num_outputs_;

    // Todos los genes de conexión del hijo provienen del conjunto de innovaciones del padre
    // más apto (homólogos, disjuntos y exceso), así que sus nodos existen en ese padre.
    for (const NodeGene& n : fitter.nodes_) {
        NodeGene nn = n;
        if (const NodeGene* o = other.findNode(n.id); o && uniform(rng, 0.0, 1.0) < 0.5)
            nn.type = o->type;
        child.nodes_.push_back(nn);
    }

    size_t i = 0, j = 0;
    const auto& A = fitter.conns_;
    const auto& B = other.conns_;
    while (i < A.size()) {
        if (j < B.size() && A[i].innov == B[j].innov) {          // homólogo
            ConnGene c = (uniform(rng, 0.0, 1.0) < 0.5) ? A[i] : B[j];
            if (!A[i].enabled || !B[j].enabled)
                c.enabled = uniform(rng, 0.0, 1.0) >= cfg.disable_inherit_prob;
            else
                c.enabled = true;
            child.conns_.push_back(c);
            i++; j++;
        } else if (j < B.size() && B[j].innov < A[i].innov) {    // disjunto del menos apto
            j++;
        } else {                                                  // disjunto/exceso del más apto
            child.conns_.push_back(A[i]);
            i++;
        }
    }
    return child;
}

double Genome::compatibility(const Genome& a, const Genome& b, const NeatConfig& cfg) {
    const auto& A = a.conns_;
    const auto& B = b.conns_;
    if (A.empty() && B.empty()) return 0.0;
    int max_a = A.empty() ? -1 : A.back().innov;
    int max_b = B.empty() ? -1 : B.back().innov;

    int excess = 0, disjoint = 0, matching = 0;
    double weight_diff = 0.0;
    size_t i = 0, j = 0;
    while (i < A.size() || j < B.size()) {
        if (i < A.size() && j < B.size() && A[i].innov == B[j].innov) {
            weight_diff += std::abs(A[i].w - B[j].w);
            matching++; i++; j++;
        } else if (j >= B.size() || (i < A.size() && A[i].innov < B[j].innov)) {
            (A[i].innov > max_b) ? excess++ : disjoint++;
            i++;
        } else {
            (B[j].innov > max_a) ? excess++ : disjoint++;
            j++;
        }
    }
    int n_genes = static_cast<int>(std::max(A.size(), B.size()));
    double N = (static_cast<int>(A.size()) < cfg.small_genome_genes &&
                static_cast<int>(B.size()) < cfg.small_genome_genes) ? 1.0 : n_genes;
    double avg_w = matching > 0 ? weight_diff / matching / cfg.compat_weight_scale : 0.0;
    return cfg.c1 * excess / N + cfg.c2 * disjoint / N + cfg.c3 * avg_w;
}

// ───────────────────────────── operadores ─────────────────────────────

bool Genome::apply(Operator op, const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
    switch (op) {
        case Operator::MutateWeights: mutateWeights(cfg, rng); return true;
        case Operator::AddConnection: return mutateAddConnection(cfg, innov, rng);
        case Operator::AddNode:       return mutateAddNode(cfg, innov, rng);
    }
    return false;
}

void Genome::mutateWeights(const NeatConfig& cfg, Rng& rng) {
    for (ConnGene& c : conns_) {
        if (uniform(rng, 0.0, 1.0) < cfg.weight_perturb_prob)
            c.w = clampWeight(c.w + uniform(rng, -cfg.weight_perturb_power, cfg.weight_perturb_power), cfg);
        else
            c.w = uniform(rng, -cfg.weight_init_range, cfg.weight_init_range);
    }
}

bool Genome::mutateAddConnection(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
    // Pares (fuente, destino) no conectados. Destino: oculta o salida; se permiten
    // conexiones recurrentes, no autoconexiones.
    std::vector<std::pair<int, int>> candidates;
    for (const NodeGene& src : nodes_)
        for (const NodeGene& dst : nodes_) {
            if (dst.role == NodeRole::Input || src.id == dst.id) continue;
            if (!hasConnection(src.id, dst.id)) candidates.emplace_back(src.id, dst.id);
        }
    if (candidates.empty()) return false;
    auto [in, out] = candidates[std::uniform_int_distribution<size_t>(0, candidates.size() - 1)(rng)];
    addConnSorted({innov.connection(in, out), in, out,
                   uniform(rng, -cfg.weight_init_range, cfg.weight_init_range), true});
    return true;
}

bool Genome::mutateAddNode(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
    std::vector<size_t> enabled;
    for (size_t k = 0; k < conns_.size(); k++)
        if (conns_[k].enabled) enabled.push_back(k);
    if (enabled.empty()) return false;

    ConnGene old = conns_[enabled[std::uniform_int_distribution<size_t>(0, enabled.size() - 1)(rng)]];
    auto s = innov.split(old.innov, old.in, old.out);
    if (findNode(s.node_id) || hasConnection(old.in, s.node_id) || hasConnection(s.node_id, old.out))
        return false;

    for (ConnGene& c : conns_)
        if (c.innov == old.innov) c.enabled = false;
    addNodeSorted({s.node_id, NodeRole::Hidden, cfg.neuron_type});
    addConnSorted({s.innov_in,  old.in,     s.node_id, cfg.add_node_in_weight, true});
    addConnSorted({s.innov_out, s.node_id,  old.out,   old.w,                  true});
    return true;
}

// ───────────────────────────── fenotipo ─────────────────────────────

std::unique_ptr<Network> Genome::buildNetwork(const NeatConfig& cfg) const {
    auto net = std::make_unique<Network>(cfg.dt, /*allow_recurrent=*/true);
    std::unordered_map<int, int> idx;
    for (NodeRole role : {NodeRole::Input, NodeRole::Output, NodeRole::Hidden})
        for (const NodeGene& n : nodes_) {
            if (n.role != role) continue;
            int k = (role == NodeRole::Input)  ? net->addInputNeuron(n.type)
                  : (role == NodeRole::Output) ? net->addOutputNeuron(n.type)
                                               : net->addHiddenNeuron(n.type);
            idx[n.id] = k;
        }
    for (const ConnGene& c : conns_) {
        if (!c.enabled || c.in == c.out) continue;
        net->addSynapse(idx.at(c.in), idx.at(c.out), c.w >= 0.0, std::abs(c.w));
    }
    return net;
}

// ───────────────────────────── consultas ─────────────────────────────

int Genome::numHidden() const {
    return static_cast<int>(std::count_if(nodes_.begin(), nodes_.end(),
                                          [](const NodeGene& n) { return n.role == NodeRole::Hidden; }));
}

int Genome::numEnabledConns() const {
    return static_cast<int>(std::count_if(conns_.begin(), conns_.end(),
                                          [](const ConnGene& c) { return c.enabled; }));
}

bool Genome::hasConnection(int in, int out) const {
    return std::any_of(conns_.begin(), conns_.end(),
                       [&](const ConnGene& c) { return c.in == in && c.out == out; });
}

const NodeGene* Genome::findNode(int id) const {
    auto it = std::lower_bound(nodes_.begin(), nodes_.end(), id,
                               [](const NodeGene& n, int v) { return n.id < v; });
    return (it != nodes_.end() && it->id == id) ? &*it : nullptr;
}

void Genome::addNodeSorted(const NodeGene& n) {
    auto it = std::lower_bound(nodes_.begin(), nodes_.end(), n.id,
                               [](const NodeGene& a, int v) { return a.id < v; });
    nodes_.insert(it, n);
}

void Genome::addConnSorted(const ConnGene& c) {
    auto it = std::lower_bound(conns_.begin(), conns_.end(), c.innov,
                               [](const ConnGene& a, int v) { return a.innov < v; });
    conns_.insert(it, c);
}

bool Genome::isValid() const {
    for (size_t k = 1; k < nodes_.size(); k++)
        if (nodes_[k - 1].id >= nodes_[k].id) return false;
    for (size_t k = 1; k < conns_.size(); k++)
        if (conns_[k - 1].innov >= conns_[k].innov) return false;
    for (size_t a = 0; a < conns_.size(); a++) {
        const ConnGene& c = conns_[a];
        const NodeGene* dst = findNode(c.out);
        if (!findNode(c.in) || !dst || dst->role == NodeRole::Input || c.in == c.out) return false;
        for (size_t b = a + 1; b < conns_.size(); b++)
            if (conns_[b].in == c.in && conns_[b].out == c.out) return false;
    }
    return true;
}

}  // namespace neat
