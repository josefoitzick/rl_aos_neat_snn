// Tests de Adaptive Pursuit (src/aos).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numeric>
#include "aos/adaptive_pursuit.hpp"
#include "neat/population.hpp"

using namespace neat;
using aos::AdaptivePursuitConfig;
using aos::AdaptivePursuitSelector;
using Catch::Matchers::WithinAbs;

// Reporte sintético: `n` hijos por operador; los del operador `good` mejoran a su padre en `gain`.
static GenerationReport makeReport(int good, double gain, int n = 20) {
    GenerationReport rep;
    rep.stats.best = 10.0;
    rep.stats.worst = 0.0;
    for (int i = 0; i < kNumOperators; i++)
        for (int k = 0; k < n; k++)
            rep.outcomes.push_back({static_cast<Operator>(i), 5.0 + (i == good ? gain : -1.0), 5.0});
    return rep;
}

static void requireValidDistribution(const std::vector<double>& p, double p_min) {
    REQUIRE(static_cast<int>(p.size()) == kNumOperators);
    REQUIRE_THAT(std::accumulate(p.begin(), p.end(), 0.0), WithinAbs(1.0, 1e-12));
    double p_max = 1.0 - (kNumOperators - 1) * p_min;
    for (double x : p) { REQUIRE(x >= p_min - 1e-12); REQUIRE(x <= p_max + 1e-12); }
}

TEST_CASE("AP: inicia uniforme o desde el P dado, siempre como distribución válida") {
    AdaptivePursuitSelector uni;
    for (double x : uni.probabilities()) REQUIRE_THAT(x, WithinAbs(1.0 / kNumOperators, 1e-12));

    AdaptivePursuitConfig cfg;
    cfg.p_init = {0.8, 0.05, 0.03};
    AdaptivePursuitSelector fromStanley(cfg);
    requireValidDistribution(fromStanley.probabilities(), cfg.p_min);
    REQUIRE(fromStanley.probabilities()[0] > 0.85);
}

TEST_CASE("AP: persigue al operador que mejora y converge a P_max / P_min") {
    AdaptivePursuitConfig cfg;
    AdaptivePursuitSelector ap(cfg);
    for (int g = 0; g < 30; g++) {
        ap.update(makeReport(/*good=*/2, /*gain=*/3.0));
        requireValidDistribution(ap.probabilities(), cfg.p_min);
    }
    auto p = ap.probabilities();
    double p_max = 1.0 - (kNumOperators - 1) * cfg.p_min;
    REQUIRE_THAT(p[2], WithinAbs(p_max, 1e-6));
    REQUIRE_THAT(p[0], WithinAbs(cfg.p_min, 1e-6));
    REQUIRE_THAT(p[1], WithinAbs(cfg.p_min, 1e-6));
}

TEST_CASE("AP: se adapta cuando cambia el operador útil (no estacionario)") {
    AdaptivePursuitSelector ap;
    for (int g = 0; g < 20; g++) ap.update(makeReport(0, 3.0));
    REQUIRE(ap.probabilities()[0] > 0.9);
    for (int g = 0; g < 20; g++) ap.update(makeReport(1, 3.0));
    REQUIRE(ap.probabilities()[1] > 0.9);
}

TEST_CASE("AP: empeorar no da crédito negativo; crédito normalizado por el rango") {
    OperatorOutcome worse{Operator::AddNode, 2.0, 5.0};
    OperatorOutcome better{Operator::AddNode, 7.0, 5.0};
    REQUIRE(AdaptivePursuitSelector::childReward(worse, 10.0, 0.0) == 0.0);
    REQUIRE_THAT(AdaptivePursuitSelector::childReward(better, 10.0, 0.0), WithinAbs(0.2, 1e-9));
    // Escala negativa (retornos de control): el mismo salto relativo da el mismo crédito
    OperatorOutcome better_neg{Operator::AddNode, -300.0, -320.0};
    // rango = -100 - (-300) = 200; mejora = 20 → 0,1
    REQUIRE_THAT(AdaptivePursuitSelector::childReward(better_neg, -100.0, -300.0), WithinAbs(0.1, 1e-9));
}

TEST_CASE("AP: sin resultados no cambia; operadores no usados no actualizan su Q") {
    AdaptivePursuitSelector ap;
    auto p0 = ap.probabilities();
    ap.update(GenerationReport{});
    REQUIRE(ap.probabilities() == p0);

    GenerationReport rep;
    rep.stats.best = 1.0; rep.stats.worst = 0.0;
    rep.outcomes.push_back({Operator::AddConnection, 1.0, 0.0});
    ap.update(rep);
    REQUIRE(ap.quality()[1] > 0.0);
    REQUIRE(ap.quality()[0] == 0.0);
    REQUIRE(std::isnan(ap.lastCredit()[0]));
}

TEST_CASE("AP integrado en NEAT: P sigue siendo distribución válida durante la evolución") {
    NeatConfig cfg;
    cfg.population_size = 60;
    Population pop(cfg, 3, 2, 5);
    AdaptivePursuitConfig apc;
    apc.p_init = {0.8, 0.05, 0.03};
    AdaptivePursuitSelector ap(apc);
    for (int gen = 0; gen < 20; gen++) {
        for (Genome& g : pop.genomes()) {          // fitness sintético
            double f = 0.0;
            for (const auto& c : g.conns()) if (c.enabled) f += c.w;
            g.fitness = f;
        }
        ap.update(pop.endGeneration());
        requireValidDistribution(ap.probabilities(), apc.p_min);
        pop.reproduce(ap.probabilities());
    }
}

TEST_CASE("AP modo SNAP-NEAT: período inicial uniforme y luego persigue al operador de mejor fitness") {
    AdaptivePursuitConfig cfg;
    cfg.snap = true;
    cfg.alpha = cfg.beta = 0.05;
    cfg.p_min = 0.10;
    cfg.init_generations = 3;
    AdaptivePursuitSelector ap(cfg);
    GenerationReport rep;
    rep.stats.best = 10.0; rep.stats.worst = 0.0;
    for (int k = 0; k < 30; k++) {           // los individuos cuyo último operador es AddNode rinden más
        rep.last_op_scores.push_back({Operator::MutateWeights, 2.0});
        rep.last_op_scores.push_back({Operator::AddConnection, 3.0});
        rep.last_op_scores.push_back({Operator::AddNode, 9.0});
    }
    for (int g = 0; g < 2; g++) {
        ap.update(rep);
        for (double x : ap.probabilities()) REQUIRE_THAT(x, WithinAbs(1.0 / 3.0, 1e-12));
    }
    for (int g = 0; g < 10; g++) {
        ap.update(rep);
        requireValidDistribution(ap.probabilities(), cfg.p_min);
    }
    REQUIRE(ap.probabilities()[2] > 0.75);   // P_max = 0,8
}
