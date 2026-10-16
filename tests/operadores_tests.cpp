// Tests del catálogo ampliado de operadores (docs/operadores_candidatos.md): cada operador hace lo que dice,
// deja genomas válidos, y los operadores "neutros" (duplicar neurona; agregar y luego borrar un nodo simple)
// conservan el fenotipo.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include "aos/adaptive_pursuit.hpp"
#include "aos/factory.hpp"
#include "neat/genome.hpp"
#include "neat/innovation.hpp"
#include "neat/population.hpp"
#include "neat/selector.hpp"

using namespace neat;
using Catch::Matchers::WithinAbs;

// Genoma con estructura: mínimo 3 → 2 y varios nodos y conexiones agregados
static Genome grown(NeatConfig& cfg, InnovationTracker& tr, Rng& rng, int nodes = 4, int conns = 4) {
    Genome g = Genome::minimal(0, 3, 2, cfg, tr, rng);
    for (int k = 0; k < nodes; k++) g.mutateAddNode(cfg, tr, rng);
    for (int k = 0; k < conns; k++) g.mutateAddConnection(cfg, tr, rng);
    REQUIRE(g.isValid());
    return g;
}

static std::map<int, double> weightsByInnov(const Genome& g) {
    std::map<int, double> m;
    for (const auto& c : g.conns()) m[c.innov] = c.w;
    return m;
}

// Fenotipo como conjunto de sinapsis habilitadas (origen, destino, peso)
static std::set<std::tuple<int, int, double>> phenotype(const Genome& g) {
    std::set<std::tuple<int, int, double>> s;
    for (const auto& c : g.conns()) if (c.enabled) s.insert({c.in, c.out, c.w});
    return s;
}

// Espigas de salida de la red durante `ms` con corrientes de entrada dadas
static std::vector<int> outputSpikes(const Genome& g, const NeatConfig& cfg, const std::vector<double>& currents, int ms) {
    auto net = g.buildNetwork(cfg);
    net->fastReset();
    net->setInputOffsets(currents);
    auto outs = net->getOutputNeuronIds();
    std::vector<int> count(outs.size(), 0);
    for (int t = 0; t < ms; t++) {
        net->step();
        for (size_t q = 0; q < outs.size(); q++) if (net->getNeuron(outs[q]).hasSpiked()) count[q]++;
    }
    return count;
}

TEST_CASE("Catálogo: nombres, parseo y errores") {
    REQUIRE(defaultCatalog().size() == static_cast<size_t>(kNumOperators));
    for (int k = 0; k < kNumOperatorTypes; k++) {
        Operator op = static_cast<Operator>(k);
        REQUIRE(parseCatalog(operatorName(op)) == std::vector<Operator>{op});
    }
    REQUIRE(parseCatalog("mutate_weights,add_connection,add_node") == defaultCatalog());
    REQUIRE_THROWS(parseCatalog("mutate_weights,no_existe"));
    REQUIRE_THROWS(parseCatalog("add_node,add_node"));
}

TEST_CASE("Pesos finos (A2): cambia 1–3 pesos habilitados, poco, y nada más") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(1);
    Genome g = grown(cfg, tr, rng);
    for (int rep = 0; rep < 50; rep++) {
        auto before = weightsByInnov(g);
        REQUIRE(g.mutateWeightsFine(cfg, rng));
        int changed = 0;
        for (const auto& c : g.conns()) {
            if (c.w == before[c.innov]) continue;
            changed++;
            REQUIRE(c.enabled);
            REQUIRE(std::abs(c.w - before[c.innov]) < 6.0 * cfg.fine_weight_sigma);
        }
        REQUIRE(changed >= 1);
        REQUIRE(changed <= 3);
    }
    REQUIRE(g.isValid());
}

TEST_CASE("Reasignar puntuales (A3): 1–3 pesos nuevos dentro del rango inicial") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(2);
    Genome g = grown(cfg, tr, rng);
    for (int rep = 0; rep < 50; rep++) {
        auto before = weightsByInnov(g);
        REQUIRE(g.mutateWeightsReset(cfg, rng));
        int changed = 0;
        for (const auto& c : g.conns())
            if (c.w != before[c.innov]) {
                changed++;
                REQUIRE(std::abs(c.w) <= cfg.weight_init_range);
            }
        REQUIRE(changed >= 1);
        REQUIRE(changed <= 3);
    }
}

TEST_CASE("Dirigido (A7): solo genes coincidentes; sin ruido el paso es colineal con (w_compañero − w)") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(3);
    Genome a = grown(cfg, tr, rng, 2, 2);
    Genome b = a;
    b.mutateWeights(cfg, rng);                 // mismos genes, otros pesos
    REQUIRE(b.mutateAddNode(cfg, tr, rng));    // genes que `a` no tiene
    cfg.directed_iso_sigma = 0.0;
    auto wa = weightsByInnov(a), wb = weightsByInnov(b);
    Genome child = a;
    REQUIRE(child.mutateDirected(b, cfg, rng));
    double factor = std::numeric_limits<double>::quiet_NaN();
    for (const auto& c : child.conns()) {
        REQUIRE(wb.count(c.innov));                // todos los genes de `a` están en `b`
        double d = wb[c.innov] - wa[c.innov];
        if (std::abs(d) < 1e-9) { REQUIRE(c.w == wa[c.innov]); continue; }
        double f = (c.w - wa[c.innov]) / d;        // = σ_line · g, el mismo para todos los genes
        if (std::isnan(factor)) factor = f;
        REQUIRE_THAT(f, WithinAbs(factor, 1e-9));
    }
    REQUIRE(child.conns().size() == a.conns().size());

    // Con un compañero idéntico y sin ruido, nada cambia
    Genome same = a;
    REQUIRE(same.mutateDirected(a, cfg, rng));
    REQUIRE(weightsByInnov(same) == wa);
}

TEST_CASE("Sesgo (B1): cambia un solo nodo no de entrada y la red lo aplica como corriente de reposo") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(4);
    Genome g = grown(cfg, tr, rng, 2, 0);
    REQUIRE(g.mutateNeuronBias(cfg, rng));
    int changed = 0;
    for (const auto& n : g.nodes()) {
        if (n.bias == 0.0) continue;
        changed++;
        REQUIRE(n.role != NodeRole::Input);
        REQUIRE(std::abs(n.bias) <= cfg.bias_max);
    }
    REQUIRE(changed == 1);
    // Con un sesgo grande, la salida dispara sin ninguna entrada
    Genome h = Genome::minimal(0, 2, 2, cfg, tr, rng);
    cfg.bias_sigma = 100.0;
    for (int k = 0; k < 20; k++) h.mutateNeuronBias(cfg, rng);
    bool any_pos = false;
    for (const auto& n : h.nodes()) any_pos = any_pos || n.bias > 3.0;
    if (any_pos) {
        auto spikes = outputSpikes(h, cfg, {0.0, 0.0}, 50);
        REQUIRE(std::accumulate(spikes.begin(), spikes.end(), 0) > 0);
    }
}

TEST_CASE("Dinámica (B2): (a, b, c, d) dentro de los rangos y aplicados en la red") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(5);
    Genome g = grown(cfg, tr, rng, 2, 0);
    const IzhikevichBounds B;
    for (int rep = 0; rep < 100; rep++) REQUIRE(g.mutateNeuronDynamics(cfg, rng));
    int custom = 0;
    for (const auto& n : g.nodes()) {
        if (!n.custom) continue;
        custom++;
        for (int k = 0; k < 4; k++) {
            REQUIRE(n.abcd[k] >= B.lo[k]);
            REQUIRE(n.abcd[k] <= B.hi[k]);
        }
    }
    REQUIRE(custom >= 1);
    auto net = g.buildNetwork(cfg);
    // Las salidas son las neuronas tras las entradas; las ocultas van al final (orden por id dentro de cada rol)
    std::vector<const NodeGene*> order;
    for (NodeRole r : {NodeRole::Input, NodeRole::Output, NodeRole::Hidden})
        for (const auto& n : g.nodes()) if (n.role == r) order.push_back(&n);
    for (size_t k = 0; k < order.size(); k++) {
        if (!order[k]->custom) continue;
        const auto& neu = net->getNeuron(static_cast<int>(k));
        REQUIRE(neu.getA() == order[k]->abcd[0]);
        REQUIRE(neu.getD() == order[k]->abcd[3]);
    }
}

TEST_CASE("Familia (B3) y agregar nodo de familia (C4): tipos dentro de las familias") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(6);
    Genome g = grown(cfg, tr, rng, 2, 0);
    std::set<NeuronType> fam(cfg.families.begin(), cfg.families.end());
    REQUIRE(g.mutateNeuronFamily(cfg, rng));
    int in_fam = 0;
    for (const auto& n : g.nodes()) if (fam.count(n.type)) in_fam++;
    REQUIRE(in_fam == 1);   // el resto sigue en TONIC_SPIKING

    std::set<int> ids_before;
    for (const auto& n : g.nodes()) ids_before.insert(n.id);
    int hidden = g.numHidden();
    REQUIRE(g.mutateAddNodeFamily(cfg, tr, rng));
    REQUIRE(g.numHidden() == hidden + 1);
    REQUIRE(g.isValid());
    int new_nodes = 0;
    for (const auto& n : g.nodes())
        if (!ids_before.count(n.id)) {
            new_nodes++;
            REQUIRE(fam.count(n.type));
        }
    REQUIRE(new_nodes == 1);
}

TEST_CASE("Ganancia de entrada (E1): solo entradas, acotada; inputGains vacío si todas valen 1") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(7);
    Genome g = Genome::minimal(0, 3, 2, cfg, tr, rng);
    REQUIRE(g.inputGains().empty());
    cfg.gain_log_sigma = 3.0;   // saltos grandes para tocar las cotas
    for (int k = 0; k < 50; k++) REQUIRE(g.mutateInputGain(cfg, rng));
    for (const auto& n : g.nodes()) {
        if (n.role != NodeRole::Input) REQUIRE(n.gain == 1.0);
        REQUIRE(n.gain >= cfg.gain_min);
        REQUIRE(n.gain <= cfg.gain_max);
    }
    auto gains = g.inputGains();
    REQUIRE(gains.size() == 3);
}

TEST_CASE("Agregar k nodos (C9): hasta k ocultas nuevas en una mutación") {
    NeatConfig cfg;
    cfg.add_node_multi_k = 3;
    InnovationTracker tr(5);
    Rng rng(8);
    Genome g = Genome::minimal(0, 3, 2, cfg, tr, rng);
    REQUIRE(g.mutateAddNodeMulti(cfg, tr, rng));
    REQUIRE(g.numHidden() >= 1);
    REQUIRE(g.numHidden() <= 3);
    REQUIRE(g.isValid());
}

TEST_CASE("Duplicar neurona (C5): estructura correcta y mismas espigas de salida") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(9);
    int done = 0;
    for (int trial = 0; trial < 20; trial++) {
        tr.newGeneration();
        Genome g = grown(cfg, tr, rng, 3, 3);
        Genome d = g;
        if (!d.mutateDuplicateNode(cfg, tr, rng)) continue;
        done++;
        REQUIRE(d.isValid());
        REQUIRE(d.numHidden() == g.numHidden() + 1);
        for (double x : {0.3, 0.7, 1.0})
            for (double y : {0.0, 0.5, 1.0}) {
                std::vector<double> cur = {x * 40.0, y * 40.0, (1.0 - x) * 40.0};
                REQUIRE(outputSpikes(g, cfg, cur, 100) == outputSpikes(d, cfg, cur, 100));
            }
    }
    REQUIRE(done >= 10);
}

TEST_CASE("Borrar conexión (D1): una habilitada menos y sin ocultas huérfanas") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(10);
    for (int trial = 0; trial < 30; trial++) {
        Genome g = grown(cfg, tr, rng, 3, 2);
        int before = g.numEnabledConns();
        REQUIRE(g.mutateDeleteConnection(cfg, rng));
        REQUIRE(g.isValid());
        REQUIRE(g.numEnabledConns() == before - 1);
        for (const auto& n : g.nodes()) {
            if (n.role != NodeRole::Hidden) continue;
            bool referenced = false;
            for (const auto& c : g.conns()) referenced = referenced || c.in == n.id || c.out == n.id;
            REQUIRE(referenced);
        }
    }
    // Sin conexiones habilitadas, falla y no cambia nada
    Genome empty = Genome::minimal(0, 1, 1, cfg, tr, rng);
    REQUIRE(empty.mutateDeleteConnection(cfg, rng));
    REQUIRE_FALSE(empty.mutateDeleteConnection(cfg, rng));
}

TEST_CASE("Borrar neurona simple (D2): agregar nodo y luego borrarlo recupera el fenotipo") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(11);
    Genome g = Genome::minimal(0, 3, 2, cfg, tr, rng);
    auto original = phenotype(g);
    Genome h = g;
    REQUIRE(h.mutateAddNode(cfg, tr, rng));
    REQUIRE(h.numHidden() == 1);
    REQUIRE(h.mutateDeleteSimpleNode(cfg, tr, rng));   // la única oculta es simple (1 entrada, 1 salida)
    REQUIRE(h.isValid());
    REQUIRE(h.numHidden() == 0);
    REQUIRE(phenotype(h) == original);

    // Sin ocultas simples ni callejones, falla
    Genome none = Genome::minimal(0, 2, 1, cfg, tr, rng);
    REQUIRE_FALSE(none.mutateDeleteSimpleNode(cfg, tr, rng));
}

TEST_CASE("Cruce y compatibilidad con parámetros de nodo") {
    NeatConfig cfg;
    InnovationTracker tr(5);
    Rng rng(12);
    Genome a = grown(cfg, tr, rng, 2, 0);
    REQUIRE_THAT(Genome::compatibility(a, a, cfg), WithinAbs(0.0, 1e-12));
    Genome b = a;
    cfg.bias_sigma = 5.0;
    REQUIRE(b.mutateNeuronBias(cfg, rng));
    REQUIRE(Genome::compatibility(a, b, cfg) > 0.0);   // mismos genes de conexión: la diferencia es del nodo
    cfg.c4 = 0.0;
    REQUIRE_THAT(Genome::compatibility(a, b, cfg), WithinAbs(0.0, 1e-12));

    // El hijo hereda los parámetros de nodo de alguno de los dos padres
    for (int k = 0; k < 20; k++) {
        Genome child = Genome::crossover(a, b, cfg, rng);
        REQUIRE(child.isValid());
        for (const auto& n : child.nodes()) {
            const NodeGene* na = a.findNode(n.id);
            const NodeGene* nb = b.findNode(n.id);
            REQUIRE((n.bias == na->bias || (nb && n.bias == nb->bias)));
        }
    }
}

TEST_CASE("Población con el catálogo completo: válida, tamaño constante y crédito por posición") {
    NeatConfig cfg;
    cfg.population_size = 60;
    cfg.catalog.clear();
    for (int k = 0; k < kNumOperatorTypes; k++) cfg.catalog.push_back(static_cast<Operator>(k));
    const size_t K = cfg.catalog.size();
    Population pop(cfg, 3, 2, 13);
    StaticSelector sel(std::vector<double>(K, 1.0));
    std::vector<int> used(K, 0);
    for (int gen = 0; gen < 25; gen++) {
        for (Genome& g : pop.genomes()) {
            double f = 0.0;
            for (const auto& c : g.conns()) if (c.enabled) f += c.w;
            g.fitness = f;
        }
        auto rep = pop.endGeneration();
        for (const auto& o : rep.outcomes) {
            REQUIRE(o.slot >= 0);
            REQUIRE(static_cast<size_t>(o.slot) < K);
            REQUIRE(cfg.catalog[o.slot] == o.op);
            used[o.slot]++;
        }
        pop.reproduce(sel.probabilities());
        REQUIRE(static_cast<int>(pop.genomes().size()) == cfg.population_size);
        for (const auto& g : pop.genomes()) REQUIRE(g.isValid());
    }
    for (size_t k = 0; k < K; k++) REQUIRE(used[k] > 0);
}

TEST_CASE("Adaptive Pursuit con un catálogo de K operadores") {
    std::vector<Operator> cat = {Operator::MutateWeights, Operator::WeightsFine, Operator::DeleteConnection,
                                 Operator::AddNode, Operator::NeuronBias};
    auto sel = aos::makeSelector("ap", aos::AdaptivePursuitConfig(), false, cat);
    auto p = sel->probabilities();
    REQUIRE(p.size() == cat.size());
    for (double x : p) REQUIRE_THAT(x, WithinAbs(1.0 / cat.size(), 1e-12));   // sin P de Stanley: uniforme

    NeatConfig cfg;
    cfg.population_size = 40;
    cfg.catalog = cat;
    Population pop(cfg, 2, 2, 17);
    for (int gen = 0; gen < 15; gen++) {
        for (Genome& g : pop.genomes()) {
            double f = 0.0;
            for (const auto& c : g.conns()) if (c.enabled) f += c.w;
            g.fitness = f;
        }
        sel->update(pop.endGeneration());
        p = sel->probabilities();
        REQUIRE_THAT(std::accumulate(p.begin(), p.end(), 0.0), WithinAbs(1.0, 1e-9));
        pop.reproduce(p);
    }
}

TEST_CASE("Determinismo con el catálogo completo") {
    auto run = [](uint64_t seed) {
        NeatConfig cfg;
        cfg.population_size = 40;
        cfg.catalog.clear();
        for (int k = 0; k < kNumOperatorTypes; k++) cfg.catalog.push_back(static_cast<Operator>(k));
        Population pop(cfg, 3, 2, seed);
        StaticSelector sel(std::vector<double>(cfg.catalog.size(), 1.0));
        double best = 0.0;
        for (int gen = 0; gen < 10; gen++) {
            for (Genome& g : pop.genomes()) {
                double f = 0.0;
                for (const auto& c : g.conns()) if (c.enabled) f += c.w;
                for (const auto& n : g.nodes()) f += n.bias + n.gain;
                g.fitness = f;
            }
            best = pop.endGeneration().stats.best;
            pop.reproduce(sel.probabilities());
        }
        return best;
    };
    REQUIRE(run(5) == run(5));
    REQUIRE(run(5) != run(6));
}

// ───────────── F1–F4: dirigidos por la actividad de la SNN (docs/experimentos/22_operadores_actividad.md) ─────────────

TEST_CASE("asignarTasas sigue el orden de buildNetwork (entradas, salidas, ocultas)", "[actividad]") {
    NeatConfig cfg; InnovationTracker tr; Rng rng(3);
    Genome g = grown(cfg, tr, rng, 3, 2);
    auto net = g.buildNetwork(cfg);
    const size_t N = net->getNeuronCount();
    std::vector<long> esp(N);
    for (size_t k = 0; k < N; k++) esp[k] = static_cast<long>(10 * k);
    g.asignarTasas(esp, 100);
    // la neurona de índice k de la red tiene tasa 0,1·k: comprobar por rol contra los ids de la red
    std::vector<int> ins = net->getInputNeuronIds(), outs = net->getOutputNeuronIds(), hid = net->getHiddenNeuronIds();
    size_t qi = 0, qo = 0, qh = 0;
    for (const auto& n : g.nodes()) {
        const int k = n.role == NodeRole::Input ? ins[qi++] : n.role == NodeRole::Output ? outs[qo++] : hid[qh++];
        REQUIRE_THAT(n.tasa, WithinAbs(0.1 * k, 1e-12));
    }
}

TEST_CASE("F1–F4: actúan solo sobre neuronas con la actividad indicada", "[actividad]") {
    NeatConfig cfg; InnovationTracker tr; Rng rng(5);
    Genome g = grown(cfg, tr, rng, 3, 3);
    // sin actividad medida: no se aplican y el genoma no cambia
    for (Operator op : {Operator::WakeOutputs, Operator::WakeHidden, Operator::CalmSaturated, Operator::GainMuteInputs}) {
        Genome h = g;
        REQUIRE_FALSE(h.apply(op, cfg, tr, rng));
        REQUIRE(weightsByInnov(h) == weightsByInnov(g));
    }
    // todas activas a tasa 0,05: ni mudas ni saturadas
    {
        Genome h = g;
        h.asignarTasas(std::vector<long>(h.buildNetwork(cfg)->getNeuronCount(), 5), 100);
        for (Operator op : {Operator::WakeOutputs, Operator::WakeHidden, Operator::CalmSaturated, Operator::GainMuteInputs})
            REQUIRE_FALSE(h.apply(op, cfg, tr, rng));
    }
    // todas mudas: despertar salida → sube el sesgo de una salida y no baja ningún peso excitatorio hacia ella
    {
        Genome h = g;
        h.asignarTasas(std::vector<long>(h.buildNetwork(cfg)->getNeuronCount(), 0), 100);
        const Genome antes = h;
        REQUIRE(h.apply(Operator::WakeOutputs, cfg, tr, rng));
        int cambiadas = 0;
        for (size_t k = 0; k < h.nodes().size(); k++) {
            const auto& n = h.nodes()[k]; const auto& m = antes.nodes()[k];
            if (n.bias != m.bias) { REQUIRE(n.role == NodeRole::Output); REQUIRE(n.bias > m.bias); cambiadas++; }
        }
        REQUIRE(cambiadas == 1);
        for (size_t k = 0; k < h.conns().size(); k++)
            if (antes.conns()[k].w > 0) REQUIRE(h.conns()[k].w >= antes.conns()[k].w);
        REQUIRE(h.isValid());
        Genome e = antes;
        REQUIRE(e.apply(Operator::GainMuteInputs, cfg, tr, rng));
        int gan = 0;
        for (size_t k = 0; k < e.nodes().size(); k++)
            if (e.nodes()[k].gain != antes.nodes()[k].gain) { REQUIRE(e.nodes()[k].role == NodeRole::Input); REQUIRE(e.nodes()[k].gain > 1.0); gan++; }
        REQUIRE(gan == 1);
    }
    // todas saturadas: calmar → baja el sesgo de una oculta o salida
    {
        Genome h = g;
        h.asignarTasas(std::vector<long>(h.buildNetwork(cfg)->getNeuronCount(), 50), 100);
        const Genome antes = h;
        REQUIRE(h.apply(Operator::CalmSaturated, cfg, tr, rng));
        int cambiadas = 0;
        for (size_t k = 0; k < h.nodes().size(); k++)
            if (h.nodes()[k].bias != antes.nodes()[k].bias) { REQUIRE(h.nodes()[k].role != NodeRole::Input); REQUIRE(h.nodes()[k].bias < antes.nodes()[k].bias); cambiadas++; }
        REQUIRE(cambiadas == 1);
    }
}
