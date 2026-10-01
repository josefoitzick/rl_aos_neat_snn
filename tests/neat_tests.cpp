// Tests del núcleo NEAT (src/neat): invariantes del genoma, marcas históricas,
// cruce, compatibilidad, reproducción y determinismo.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <set>
#include "neat/genome.hpp"
#include "neat/innovation.hpp"
#include "neat/population.hpp"
#include "neat/selector.hpp"

using namespace neat;
using Catch::Matchers::WithinAbs;

static std::set<int> innovs(const Genome& g) {
    std::set<int> s;
    for (const auto& c : g.conns()) s.insert(c.innov);
    return s;
}

TEST_CASE("Genoma mínimo: entradas × salidas conexiones, mismas innovaciones en toda la población") {
    NeatConfig cfg;
    InnovationTracker tr(3 + 2);
    Rng rng(1);
    Genome a = Genome::minimal(0, 3, 2, cfg, tr, rng);
    Genome b = Genome::minimal(1, 3, 2, cfg, tr, rng);
    REQUIRE(a.isValid());
    REQUIRE(a.conns().size() == 6);
    REQUIRE(a.numHidden() == 0);
    REQUIRE(innovs(a) == innovs(b));
}

TEST_CASE("Compatibilidad: δ=0 consigo mismo; un gen en exceso en genoma pequeño suma c1") {
    NeatConfig cfg;
    InnovationTracker tr(2 + 1);
    Rng rng(2);
    Genome a = Genome::minimal(0, 2, 1, cfg, tr, rng);
    REQUIRE_THAT(Genome::compatibility(a, a, cfg), WithinAbs(0.0, 1e-12));

    Genome b = a;
    REQUIRE(b.mutateAddNode(cfg, tr, rng));   // +2 genes nuevos (exceso), 1 deshabilitado
    // 2 genes en exceso, N = 1 (genomas < 20 genes), pesos homólogos iguales
    REQUIRE_THAT(Genome::compatibility(a, b, cfg), WithinAbs(2.0 * cfg.c1, 1e-12));
}

TEST_CASE("Agregar nodo: deshabilita la conexión, peso entrante de relevo y saliente original") {
    NeatConfig cfg;
    InnovationTracker tr(2 + 1);
    Rng rng(3);
    Genome g = Genome::minimal(0, 2, 1, cfg, tr, rng);
    auto before = g.conns();
    REQUIRE(g.mutateAddNode(cfg, tr, rng));
    REQUIRE(g.isValid());
    REQUIRE(g.numHidden() == 1);
    REQUIRE(g.conns().size() == before.size() + 2);
    REQUIRE(g.numEnabledConns() == static_cast<int>(before.size()) + 1);

    int hidden = g.nodes().back().id;
    const ConnGene *in = nullptr, *out = nullptr, *old = nullptr;
    for (const auto& c : g.conns()) {
        if (c.out == hidden) in = &c;
        if (c.in == hidden) out = &c;
        if (!c.enabled) old = &c;
    }
    REQUIRE((in && out && old));
    REQUIRE(in->in == old->in);
    REQUIRE(out->out == old->out);
    REQUIRE(in->w == cfg.add_node_in_weight);
    REQUIRE(out->w == old->w);
}

TEST_CASE("Marcas históricas: la misma división en la misma generación reutiliza nodo e innovaciones") {
    NeatConfig cfg;
    InnovationTracker tr(1 + 1);
    Rng rng(4);
    Genome a = Genome::minimal(0, 1, 1, cfg, tr, rng);   // una sola conexión: ambos dividen la misma
    Genome b = a;
    REQUIRE(a.mutateAddNode(cfg, tr, rng));
    REQUIRE(b.mutateAddNode(cfg, tr, rng));
    REQUIRE(innovs(a) == innovs(b));
    REQUIRE(a.nodes().back().id == b.nodes().back().id);

    tr.newGeneration();                                  // en otra generación: innovación nueva
    Genome c = Genome::minimal(2, 1, 1, cfg, tr, rng);
    REQUIRE(c.mutateAddNode(cfg, tr, rng));
    REQUIRE(c.nodes().back().id != a.nodes().back().id);
}

TEST_CASE("Agregar conexión: agrega exactamente una; falla si el grafo está completo") {
    NeatConfig cfg;
    InnovationTracker tr(2 + 1);
    Rng rng(5);
    Genome g = Genome::minimal(0, 2, 1, cfg, tr, rng);
    REQUIRE(g.mutateAddNode(cfg, tr, rng));
    size_t n = g.conns().size();
    REQUIRE(g.mutateAddConnection(cfg, tr, rng));
    REQUIRE(g.conns().size() == n + 1);
    REQUIRE(g.isValid());
    // Completar el grafo: en algún momento ya no hay pares libres
    int added = 0;
    while (g.mutateAddConnection(cfg, tr, rng)) added++;
    REQUIRE(g.isValid());
    REQUIRE_FALSE(g.mutateAddConnection(cfg, tr, rng));
}

TEST_CASE("Cruce: el hijo es válido y hereda exactamente las innovaciones del padre más apto") {
    NeatConfig cfg;
    InnovationTracker tr(3 + 2);
    Rng rng(6);
    Genome base = Genome::minimal(0, 3, 2, cfg, tr, rng);
    Genome fit = base, weak = base;
    tr.newGeneration();
    REQUIRE(fit.mutateAddNode(cfg, tr, rng));
    REQUIRE(fit.mutateAddConnection(cfg, tr, rng));
    tr.newGeneration();
    REQUIRE(weak.mutateAddNode(cfg, tr, rng));
    for (int rep = 0; rep < 50; rep++) {
        Genome child = Genome::crossover(fit, weak, cfg, rng);
        REQUIRE(child.isValid());
        REQUIRE(innovs(child) == innovs(fit));
    }
}

TEST_CASE("Fenotipo: neuronas y sinapsis coinciden con el genoma; el signo define exc/inh") {
    NeatConfig cfg;
    InnovationTracker tr(2 + 2);
    Rng rng(7);
    Genome g = Genome::minimal(0, 2, 2, cfg, tr, rng);
    REQUIRE(g.mutateAddNode(cfg, tr, rng));
    auto net = g.buildNetwork(cfg);
    REQUIRE(net->getInputCount() == 2);
    REQUIRE(net->getOutputCount() == 2);
    REQUIRE(net->getNeuronCount() == 5);
    REQUIRE(static_cast<int>(net->getSynapseCount()) == g.numEnabledConns());
    size_t s = 0;
    for (const auto& c : g.conns()) {
        if (!c.enabled) continue;
        REQUIRE(net->getSynapse(static_cast<int>(s)).isExcitatory() == (c.w >= 0.0));
        REQUIRE_THAT(net->getSynapse(static_cast<int>(s)).getWeight(), WithinAbs(std::abs(c.w), 1e-12));
        s++;
    }
}

// Fitness sintético y determinista: premia pesos grandes y excitatorios
static void syntheticEval(std::vector<Genome>& pop) {
    for (Genome& g : pop) {
        double f = 0.0;
        for (const auto& c : g.conns()) if (c.enabled) f += c.w;
        g.fitness = f - 100.0;   // valores negativos a propósito (como los retornos de control)
    }
}

TEST_CASE("Población: tamaño constante, genomas válidos y un resultado por hijo no élite") {
    NeatConfig cfg;
    cfg.population_size = 60;
    Population pop(cfg, 3, 2, 42);
    StaticSelector sel = StaticSelector::stanley2002();
    for (int gen = 0; gen < 15; gen++) {
        syntheticEval(pop.genomes());
        GenerationReport rep = pop.endGeneration();
        int with_op = 0;
        for (const auto& g : pop.genomes()) if (g.origin.has_operator) with_op++;
        REQUIRE(static_cast<int>(rep.outcomes.size()) == with_op);
        sel.update(rep);
        pop.reproduce(sel.probabilities());
        REQUIRE(static_cast<int>(pop.genomes().size()) == cfg.population_size);
        for (const auto& g : pop.genomes()) REQUIRE(g.isValid());
    }
}

TEST_CASE("Población: cruce interespecies forzado (tasa 1) con muchas especies") {
    NeatConfig cfg;
    cfg.population_size = 60;
    cfg.interspecies_rate = 1.0;
    cfg.compat_threshold = 0.5;    // muchas especies
    cfg.mutation_only_fraction = 0.0;
    Population pop(cfg, 3, 2, 11);
    StaticSelector sel = StaticSelector::stanley2002();
    for (int gen = 0; gen < 10; gen++) {
        syntheticEval(pop.genomes());
        auto rep = pop.endGeneration();
        REQUIRE(rep.stats.num_species > 1);
        pop.reproduce(sel.probabilities());
        REQUIRE(static_cast<int>(pop.genomes().size()) == cfg.population_size);
        for (const auto& g : pop.genomes()) REQUIRE(g.isValid());
    }
}

TEST_CASE("Especiación dinámica: δt se ajusta hacia el número objetivo de especies") {
    NeatConfig cfg;
    cfg.population_size = 100;
    cfg.target_species = 8;
    Population pop(cfg, 4, 2, 21);
    StaticSelector sel = StaticSelector::stanley2002();
    double sum = 0.0;
    for (int gen = 0; gen < 60; gen++) {
        syntheticEval(pop.genomes());
        auto rep = pop.endGeneration();
        REQUIRE(pop.compatThreshold() >= cfg.compat_threshold_min);
        if (gen >= 40) sum += rep.stats.num_species;
        pop.reproduce(sel.probabilities());
    }
    double mean = sum / 20.0;
    INFO("especies promedio en generaciones 40–59: " << mean);
    REQUIRE(mean >= 4.0);
    REQUIRE(mean <= 14.0);
}

TEST_CASE("Población: con P = (1,0,0) todos los hijos usan 'mutar pesos'") {
    NeatConfig cfg;
    cfg.population_size = 40;
    Population pop(cfg, 2, 1, 7);
    StaticSelector only_weights({1.0, 0.0, 0.0});
    for (int gen = 0; gen < 5; gen++) {
        syntheticEval(pop.genomes());
        auto rep = pop.endGeneration();
        for (const auto& o : rep.outcomes) REQUIRE(o.op == Operator::MutateWeights);
        pop.reproduce(only_weights.probabilities());
    }
    for (const auto& g : pop.genomes()) REQUIRE(g.numHidden() == 0);
}

TEST_CASE("Población: NEAT mejora el fitness sintético (con retornos negativos)") {
    NeatConfig cfg;
    cfg.population_size = 80;
    Population pop(cfg, 3, 2, 3);
    StaticSelector sel = StaticSelector::stanley2002();
    syntheticEval(pop.genomes());
    double first = pop.endGeneration().stats.best;
    for (int gen = 0; gen < 30; gen++) {
        pop.reproduce(sel.probabilities());
        syntheticEval(pop.genomes());
        pop.endGeneration();
    }
    INFO("mejor inicial " << first << ", mejor histórico " << pop.bestEver().fitness);
    REQUIRE(pop.bestEver().fitness > first + 50.0);
}

TEST_CASE("Determinismo: misma semilla → misma evolución") {
    auto run = [](uint64_t seed) {
        NeatConfig cfg;
        cfg.population_size = 50;
        Population pop(cfg, 3, 2, seed);
        StaticSelector sel = StaticSelector::stanley2002();
        for (int gen = 0; gen < 10; gen++) {
            syntheticEval(pop.genomes());
            pop.endGeneration();
            pop.reproduce(sel.probabilities());
        }
        std::vector<double> sig;
        for (const auto& g : pop.genomes())
            for (const auto& c : g.conns()) sig.push_back(c.w + 1000.0 * c.innov);
        return sig;
    };
    REQUIRE(run(99) == run(99));
    REQUIRE(run(99) != run(100));
}
