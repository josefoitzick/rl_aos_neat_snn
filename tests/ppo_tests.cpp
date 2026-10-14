// Tests del agente PPO que no necesitan RLtools (docs/diseno_ppo.md §5, V0): acción → P, fronteras de decisión
// y el AOS sintético del paso 1.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include "aos/action_map.hpp"
#include "experiment/aos_toy.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("actionToP: con a = 0 reproduce P0 y respeta las cotas", "[ppo]") {
    const std::vector<double> P0{0.6, 0.2, 0.1, 0.05, 0.05};
    const auto lp = aos::logP0(P0);
    const std::vector<double> cero(5, 0.0);
    const auto P = aos::actionToP(cero.data(), lp, 3.0);
    for (size_t i = 0; i < P0.size(); i++) REQUIRE_THAT(P[i], WithinAbs(P0[i], 1e-12));

    const double c = 3.0;
    const std::vector<double> extrema{100.0, -100.0, 0.0, 50.0, -50.0};
    const auto Q = aos::actionToP(extrema.data(), lp, c);
    REQUIRE_THAT(std::accumulate(Q.begin(), Q.end(), 0.0), WithinAbs(1.0, 1e-12));
    for (size_t i = 0; i < Q.size(); i++) {
        REQUIRE(Q[i] > 0.0);
        for (size_t j = 0; j < Q.size(); j++)
            REQUIRE(Q[i] / Q[j] <= (P0[i] / P0[j]) * std::exp(2 * c) * (1 + 1e-9));
    }
}

TEST_CASE("logP0: piso, renormalización y errores", "[ppo]") {
    const auto lp = aos::logP0({1.0, 0.0, 0.0}, 0.002);
    REQUIRE_THAT(std::exp(lp[0]), WithinAbs(1.0 / 1.004, 1e-12));
    REQUIRE_THAT(std::exp(lp[1]), WithinAbs(0.002 / 1.004, 1e-12));
    REQUIRE_THROWS(aos::logP0({}));
    REQUIRE_THROWS(aos::logP0({0.5, -0.1}));
    REQUIRE_THROWS(aos::logP0({0.5, std::numeric_limits<double>::quiet_NaN()}));
    const std::vector<double> nan_a{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    REQUIRE_THROWS(aos::actionToP(nan_a.data(), lp, 3.0));
}

TEST_CASE("decisionBoundaries: ventanas que cubren todas las generaciones", "[ppo]") {
    const auto g = aos::decisionBoundaries(501, 50);
    for (int t = 0; t <= 50; t++) REQUIRE(g[t] == 10 * t);
    for (int G : {357, 577, 1071}) {
        const auto b = aos::decisionBoundaries(G, 50);
        REQUIRE(b.front() == 0);
        REQUIRE(b.back() == G - 1);
        for (int t = 0; t < 50; t++) REQUIRE(b[t + 1] - b[t] >= 1);
    }
    REQUIRE_THROWS(aos::decisionBoundaries(50, 50));
}

TEST_CASE("AOS sintético: determinista y de T decisiones", "[ppo]") {
    aos::toy::Config cfg;
    const std::vector<double> P{0.6, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0};
    aos::toy::Run a, b;
    a.reset(42, cfg);
    b.reset(42, cfg);
    while (!a.done()) {
        a.advance(P);
        b.advance(P);
        REQUIRE(a.obs() == b.obs());
        REQUIRE(a.phi() == b.phi());
    }
    REQUIRE(a.t() == cfg.T);
    REQUIRE(a.finalScore() == b.finalScore());
    REQUIRE(a.finalScore() > 0.0);
    REQUIRE(a.finalScore() < 1.0);
    REQUIRE_THROWS(a.advance(P));
}

TEST_CASE("AOS sintético: el pool lleva la contabilidad de corridas y semillas", "[ppo]") {
    aos::toy::Config cfg;
    cfg.T = 5;
    aos::toy::Pool pool(cfg, 3, 100, 0.5, 10.0);
    const std::vector<std::vector<double>> P(3, std::vector<double>(aos::toy::K, 1.0 / aos::toy::K));
    pool.start({0, 1, 2});
    REQUIRE_THROWS(pool.start({1}));   // reiniciar una corrida en curso es un error
    for (int t = 0; t < cfg.T; t++) pool.advance(P);
    const auto fin = pool.takeFinished();
    REQUIRE(fin.size() == 3);
    REQUIRE(fin[0].seed == 100);
    REQUIRE(fin[2].seed == 102);
    REQUIRE_THROWS(pool.advance(P));   // una decisión de más es un error
    pool.start({0, 1, 2});
    REQUIRE(pool.nextSeed() == 106);
    REQUIRE(pool.decisions() == 15);
}
