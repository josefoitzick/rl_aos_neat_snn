#include "genome.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include "innovation.hpp"

namespace neat {

static double uniform(Rng& rng, double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
}

static double gauss(Rng& rng, double sigma) {
    return std::normal_distribution<double>(0.0, sigma)(rng);
}

static double clampWeight(double w, const NeatConfig& cfg) {
    return std::clamp(w, -cfg.weight_max, cfg.weight_max);
}

static size_t randomIndex(Rng& rng, size_t n) {
    return std::uniform_int_distribution<size_t>(0, n - 1)(rng);
}

// Cuántas conexiones tocan A2 y A3: 1, 2 o 3 con probabilidades 0,6 / 0,3 / 0,1 (como SharpNEAT 4)
static int pickCardinality(Rng& rng) {
    double u = uniform(rng, 0.0, 1.0);
    return u < 0.6 ? 1 : (u < 0.9 ? 2 : 3);
}

// k índices distintos de `pool` (Fisher–Yates parcial); devuelve todos si k >= tamaño
static std::vector<size_t> sampleDistinct(std::vector<size_t> pool, int k, Rng& rng) {
    const size_t n = std::min(pool.size(), static_cast<size_t>(k));
    for (size_t i = 0; i < n; i++) std::swap(pool[i], pool[i + randomIndex(rng, pool.size() - i)]);
    pool.resize(n);
    return pool;
}

// (a, b, c, d) efectivos de un nodo: los propios (B2) o los de su familia
static void effectiveParams(const NodeGene& n, double out[4]) {
    if (n.custom) {
        std::copy(n.abcd, n.abcd + 4, out);
        return;
    }
    IzhikevichNeuron tmp(n.type);
    out[0] = tmp.getA(); out[1] = tmp.getB(); out[2] = tmp.getC(); out[3] = tmp.getD();
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
        const NodeGene* o = other.findNode(n.id);
        child.nodes_.push_back((o && uniform(rng, 0.0, 1.0) < 0.5) ? *o : n);
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

    // Parámetros de nodo (catálogo ampliado), cada término llevado a ~[0, 1]. Con los valores por defecto
    // todos son 0 y δ es exactamente la de Stanley.
    static const IzhikevichBounds kB;
    double node_diff = 0.0;
    int node_matching = 0;
    for (size_t p = 0, q = 0; p < a.nodes_.size() && q < b.nodes_.size();) {
        const NodeGene& x = a.nodes_[p];
        const NodeGene& y = b.nodes_[q];
        if (x.id < y.id) { p++; continue; }
        if (y.id < x.id) { q++; continue; }
        if (x.role == NodeRole::Input) {
            if (x.gain != y.gain) node_diff += std::abs(std::log(x.gain / y.gain)) / std::log(cfg.gain_max);
        } else {
            node_diff += std::abs(x.bias - y.bias) / cfg.bias_max;
            if (x.custom || y.custom) {
                double px[4], py[4];
                effectiveParams(x, px);
                effectiveParams(y, py);
                for (int k = 0; k < 4; k++) node_diff += std::abs(px[k] - py[k]) / (kB.hi[k] - kB.lo[k]) / 4.0;
            } else if (x.type != y.type) {
                node_diff += 1.0;
            }
        }
        node_matching++; p++; q++;
    }
    double avg_node = node_matching > 0 ? node_diff / node_matching : 0.0;
    return cfg.c1 * excess / N + cfg.c2 * disjoint / N + cfg.c3 * avg_w + cfg.c4 * avg_node;
}

// ───────────────────────────── operadores ─────────────────────────────

bool Genome::apply(Operator op, const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
    switch (op) {
        case Operator::MutateWeights:    mutateWeights(cfg, rng); return true;
        case Operator::AddConnection:    return mutateAddConnection(cfg, innov, rng);
        case Operator::AddNode:          return mutateAddNode(cfg, innov, rng);
        case Operator::WeightsFine:      return mutateWeightsFine(cfg, rng);
        case Operator::WeightsReset:     return mutateWeightsReset(cfg, rng);
        case Operator::WeightsDirected:  return false;   // necesita otro genoma: mutateDirected
        case Operator::NeuronBias:       return mutateNeuronBias(cfg, rng);
        case Operator::NeuronDynamics:   return mutateNeuronDynamics(cfg, rng);
        case Operator::NeuronFamily:     return mutateNeuronFamily(cfg, rng);
        case Operator::InputGain:        return mutateInputGain(cfg, rng);
        case Operator::AddNodeFamily:    return mutateAddNodeFamily(cfg, innov, rng);
        case Operator::AddNodeMulti:     return mutateAddNodeMulti(cfg, innov, rng);
        case Operator::DuplicateNode:    return mutateDuplicateNode(cfg, innov, rng);
        case Operator::DeleteConnection: return mutateDeleteConnection(cfg, rng);
        case Operator::DeleteSimpleNode: return mutateDeleteSimpleNode(cfg, innov, rng);
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
    return addNodeOnRandomConnection(cfg.neuron_type, cfg, innov, rng);
}

bool Genome::addNodeOnRandomConnection(NeuronType type, const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
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
    addNodeSorted({s.node_id, NodeRole::Hidden, type});
    addConnSorted({s.innov_in,  old.in,     s.node_id, cfg.add_node_in_weight, true});
    addConnSorted({s.innov_out, s.node_id,  old.out,   old.w,                  true});
    return true;
}

// A2: perturbación chica de 1–3 conexiones habilitadas (gaussiana o, con fine_weight_cauchy, Cauchy)
bool Genome::mutateWeightsFine(const NeatConfig& cfg, Rng& rng) {
    std::vector<size_t> enabled;
    for (size_t k = 0; k < conns_.size(); k++)
        if (conns_[k].enabled) enabled.push_back(k);
    if (enabled.empty()) return false;
    for (size_t k : sampleDistinct(enabled, pickCardinality(rng), rng)) {
        double delta = cfg.fine_weight_cauchy
            ? std::cauchy_distribution<double>(0.0, cfg.fine_weight_sigma)(rng)
            : gauss(rng, cfg.fine_weight_sigma);
        conns_[k].w = clampWeight(conns_[k].w + delta, cfg);
    }
    return true;
}

// A3: valor nuevo ~ U(-init, init) para 1–3 conexiones habilitadas
bool Genome::mutateWeightsReset(const NeatConfig& cfg, Rng& rng) {
    std::vector<size_t> enabled;
    for (size_t k = 0; k < conns_.size(); k++)
        if (conns_[k].enabled) enabled.push_back(k);
    if (enabled.empty()) return false;
    for (size_t k : sampleDistinct(enabled, pickCardinality(rng), rng))
        conns_[k].w = uniform(rng, -cfg.weight_init_range, cfg.weight_init_range);
    return true;
}

// A7: Iso+LineDD (Vassiliades y Mouret 2018) sobre los genes coincidentes con `partner`:
//   w ← w + σ_iso·N(0,1) + σ_line·g·(w_partner − w),  con un solo g ~ N(0,1) para todo el genoma
// (el paso sigue la línea entre los dos genomas; el ruido isotrópico la engrosa).
bool Genome::mutateDirected(const Genome& partner, const NeatConfig& cfg, Rng& rng) {
    std::vector<std::pair<size_t, double>> matching;   // (índice en este genoma, peso del compañero)
    for (size_t i = 0, j = 0; i < conns_.size() && j < partner.conns_.size();) {
        if (conns_[i].innov < partner.conns_[j].innov) i++;
        else if (partner.conns_[j].innov < conns_[i].innov) j++;
        else { matching.emplace_back(i, partner.conns_[j].w); i++; j++; }
    }
    if (matching.empty()) return false;
    const double g = gauss(rng, 1.0);
    for (const auto& [k, wp] : matching) {
        double& w = conns_[k].w;
        w = clampWeight(w + gauss(rng, cfg.directed_iso_sigma) + cfg.directed_line_sigma * g * (wp - w), cfg);
    }
    return true;
}

// B1: corriente de reposo de una neurona oculta o de salida (las entradas usan su corriente para la observación)
bool Genome::mutateNeuronBias(const NeatConfig& cfg, Rng& rng) {
    std::vector<size_t> cand;
    for (size_t k = 0; k < nodes_.size(); k++)
        if (nodes_[k].role != NodeRole::Input) cand.push_back(k);
    if (cand.empty()) return false;
    NodeGene& n = nodes_[cand[randomIndex(rng, cand.size())]];
    n.bias = std::clamp(n.bias + gauss(rng, cfg.bias_sigma), -cfg.bias_max, cfg.bias_max);
    return true;
}

// B2: perturbación acotada de (a, b, c, d); la primera vez parte de los parámetros de su familia
bool Genome::mutateNeuronDynamics(const NeatConfig& cfg, Rng& rng) {
    (void)cfg;
    static const IzhikevichBounds kB;
    std::vector<size_t> cand;
    for (size_t k = 0; k < nodes_.size(); k++)
        if (nodes_[k].role != NodeRole::Input) cand.push_back(k);
    if (cand.empty()) return false;
    NodeGene& n = nodes_[cand[randomIndex(rng, cand.size())]];
    if (!n.custom) {
        effectiveParams(n, n.abcd);
        n.custom = true;
    }
    for (int k = 0; k < 4; k++)
        n.abcd[k] = std::clamp(n.abcd[k] + gauss(rng, kB.sigma[k]), kB.lo[k], kB.hi[k]);
    return true;
}

// B3: otra familia cortical para una neurona oculta o de salida (vuelve a los parámetros de la familia)
bool Genome::mutateNeuronFamily(const NeatConfig& cfg, Rng& rng) {
    std::vector<size_t> cand;
    for (size_t k = 0; k < nodes_.size(); k++)
        if (nodes_[k].role != NodeRole::Input) cand.push_back(k);
    if (cand.empty()) return false;
    NodeGene& n = nodes_[cand[randomIndex(rng, cand.size())]];
    std::vector<NeuronType> options;
    for (NeuronType t : cfg.families)
        if (t != n.type || n.custom) options.push_back(t);
    if (options.empty()) return false;
    n.type = options[randomIndex(rng, options.size())];
    n.custom = false;
    return true;
}

// E1: ganancia multiplicativa de una entrada, en escala logarítmica
bool Genome::mutateInputGain(const NeatConfig& cfg, Rng& rng) {
    std::vector<size_t> cand;
    for (size_t k = 0; k < nodes_.size(); k++)
        if (nodes_[k].role == NodeRole::Input) cand.push_back(k);
    if (cand.empty()) return false;
    NodeGene& n = nodes_[cand[randomIndex(rng, cand.size())]];
    n.gain = std::clamp(n.gain * std::exp(gauss(rng, cfg.gain_log_sigma)), cfg.gain_min, cfg.gain_max);
    return true;
}

// C4: como agregar nodo, pero la neurona nueva es de una familia al azar
bool Genome::mutateAddNodeFamily(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
    if (cfg.families.empty()) return false;
    NeuronType t = cfg.families[randomIndex(rng, cfg.families.size())];
    return addNodeOnRandomConnection(t, cfg, innov, rng);
}

// C9: hasta k nodos nuevos en una sola mutación (intensidad estructural, como las mutaciones múltiples de EONS)
bool Genome::mutateAddNodeMulti(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
    bool any = false;
    for (int k = 0; k < cfg.add_node_multi_k; k++)
        any = addNodeOnRandomConnection(cfg.neuron_type, cfg, innov, rng) || any;
    return any;
}

// C5: copia una oculta con sus mismas entradas y reparte a la mitad sus salidas (Net2WiderNet). Las dos copias
// reciben lo mismo y disparan igual, y las conductancias se suman: la red calcula lo mismo (salvo redondeo).
bool Genome::mutateDuplicateNode(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
    (void)cfg;
    std::vector<size_t> cand;
    for (size_t k = 0; k < nodes_.size(); k++) {
        if (nodes_[k].role != NodeRole::Hidden) continue;
        const int id = nodes_[k].id;
        if (std::any_of(conns_.begin(), conns_.end(),
                        [&](const ConnGene& c) { return c.enabled && c.in == id && c.out != id; }))
            cand.push_back(k);
    }
    if (cand.empty()) return false;
    const NodeGene orig = nodes_[cand[randomIndex(rng, cand.size())]];
    const int dup_id = innov.duplicate(orig.id);
    if (findNode(dup_id)) return false;

    NodeGene dup = orig;
    dup.id = dup_id;
    addNodeSorted(dup);
    const std::vector<ConnGene> snapshot = conns_;
    for (const ConnGene& c : snapshot) {
        if (!c.enabled || c.in == c.out) continue;
        if (c.out == orig.id) {
            addConnSorted({innov.connection(c.in, dup_id), c.in, dup_id, c.w, true});
        } else if (c.in == orig.id) {
            const double half = c.w / 2.0;
            for (ConnGene& o : conns_)
                if (o.innov == c.innov) o.w = half;
            addConnSorted({innov.connection(dup_id, c.out), dup_id, c.out, half, true});
        }
    }
    return true;
}

// D1: borra una conexión habilitada; una oculta que queda sin ningún gen de conexión se borra también
bool Genome::mutateDeleteConnection(const NeatConfig& cfg, Rng& rng) {
    (void)cfg;
    std::vector<size_t> enabled;
    for (size_t k = 0; k < conns_.size(); k++)
        if (conns_[k].enabled) enabled.push_back(k);
    if (enabled.empty()) return false;
    const ConnGene c = conns_[enabled[randomIndex(rng, enabled.size())]];
    conns_.erase(std::find_if(conns_.begin(), conns_.end(), [&](const ConnGene& x) { return x.innov == c.innov; }));
    for (int end : {c.in, c.out}) {
        const NodeGene* n = findNode(end);
        if (n && n->role == NodeRole::Hidden &&
            std::none_of(conns_.begin(), conns_.end(), [&](const ConnGene& x) { return x.in == end || x.out == end; }))
            removeNode(end);
    }
    return true;
}

// D2: borra una oculta "simple". Con exactamente 1 entrada (s → h) y 1 salida (h → t) habilitadas, la reemplaza
// por un puente s → t con el peso de salida (el inverso de agregar nodo). Sin entradas o sin salidas
// habilitadas ("callejón sin salida", como en SharpNEAT), solo la borra.
bool Genome::mutateDeleteSimpleNode(const NeatConfig& cfg, InnovationTracker& innov, Rng& rng) {
    struct Cand { int id; int src; int dst; double w_out; bool simple; };
    std::vector<Cand> cand;
    for (const NodeGene& n : nodes_) {
        if (n.role != NodeRole::Hidden) continue;
        int n_in = 0, n_out = 0, src = -1, dst = -1;
        double w_out = 0.0;
        for (const ConnGene& c : conns_) {
            if (!c.enabled || c.in == c.out) continue;
            if (c.out == n.id) { n_in++; src = c.in; }
            if (c.in == n.id)  { n_out++; dst = c.out; w_out = c.w; }
        }
        if (n_in == 1 && n_out == 1 && src != dst) cand.push_back({n.id, src, dst, w_out, true});
        else if (n_in == 0 || n_out == 0) cand.push_back({n.id, -1, -1, 0.0, false});
    }
    if (cand.empty()) return false;
    const Cand c = cand[randomIndex(rng, cand.size())];
    removeNode(c.id);
    if (!c.simple) return true;
    for (ConnGene& x : conns_)
        if (x.in == c.src && x.out == c.dst) {
            if (x.enabled) x.w = clampWeight(x.w + c.w_out, cfg);   // ruta paralela: se suman
            else { x.enabled = true; x.w = c.w_out; }
            return true;
        }
    addConnSorted({innov.connection(c.src, c.dst), c.src, c.dst, c.w_out, true});
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
            if (role == NodeRole::Input) continue;
            if (n.custom) net->setNeuronParams(k, n.abcd[0], n.abcd[1], n.abcd[2], n.abcd[3]);
            if (n.bias != 0.0) net->setNeuronIOffset(k, n.bias);
        }
    for (const ConnGene& c : conns_) {
        if (!c.enabled || c.in == c.out) continue;
        net->addSynapse(idx.at(c.in), idx.at(c.out), c.w >= 0.0, std::abs(c.w));
    }
    return net;
}

std::vector<double> Genome::inputGains() const {
    std::vector<double> g;
    bool all_one = true;
    for (const NodeGene& n : nodes_)
        if (n.role == NodeRole::Input) {
            g.push_back(n.gain);
            all_one = all_one && n.gain == 1.0;
        }
    if (all_one) g.clear();
    return g;
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

void Genome::removeNode(int id) {
    conns_.erase(std::remove_if(conns_.begin(), conns_.end(),
                                [&](const ConnGene& c) { return c.in == id || c.out == id; }),
                 conns_.end());
    nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(), [&](const NodeGene& n) { return n.id == id; }),
                 nodes_.end());
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
