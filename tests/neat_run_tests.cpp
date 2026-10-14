// Tests del paso 2 del agente PPO (docs/diseno_ppo.md §3.3 y §5): crédito por rango, rasgos de estado, evaluador
// de varias corridas y NEAT como entorno (NeatRun / NeatPool). La comparación contra el binario de neat_control
// está en scripts/verificar_neat_run.sh; aquí NeatRun se compara con el mismo ciclo escrito a mano.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <vector>
#include "aos/action_map.hpp"
#include "aos/features.hpp"
#include "aos/neat_run.hpp"
#include "eval/evaluator.hpp"
#include "neat/population.hpp"
#include "neat/selector.hpp"

using Catch::Matchers::WithinAbs;

namespace {

aos::NeatRunConfig configChica() {
    aos::NeatRunConfig cfg;
    cfg.env_name = "pendulum";
    cfg.ncfg.population_size = 20;
    cfg.generations = 7;
    cfg.episodes = 2;
    cfg.test_episodes = 2;
    cfg.final_episodes = 3;
    cfg.reward_episodes = 3;
    cfg.T = 3;
    cfg.ref_trivial = -1500.0;
    cfg.ref_rl = -150.0;
    return cfg;
}

neat::Genome conOrigen(double fitness, int slot, bool applied = true) {
    neat::Genome g;
    g.fitness = fitness;
    g.origin.has_operator = slot >= 0;
    g.origin.slot = slot;
    g.origin.applied = applied;
    return g;
}

}  // namespace

TEST_CASE("rankCredit: percentiles con empates promediados, por operador", "[ppo][neat_run]") {
    // fitness 1, 2, 2, 3, 4 → percentiles 0, 0,375, 0,375, 0,75, 1
    std::vector<neat::Genome> G{conOrigen(2, 0), conOrigen(1, 1), conOrigen(4, 0, false), conOrigen(2, 2), conOrigen(3, -1)};
    const auto r = neat::rankCredit(G, 3);
    REQUIRE(r[0].n == 2);
    REQUIRE_THAT(r[0].rank_sum, WithinAbs(0.375 + 1.0, 1e-12));
    REQUIRE(r[0].top == 1);
    REQUIRE(r[0].fail == 1);
    REQUIRE(r[1].n == 1);
    REQUIRE_THAT(r[1].rank_sum, WithinAbs(0.0, 1e-12));
    REQUIRE(r[2].n == 1);
    REQUIRE_THAT(r[2].rank_sum, WithinAbs(0.375, 1e-12));
}

TEST_CASE("FeatureExtractor: rasgos de una ventana sintética", "[ppo][neat_run]") {
    aos::FeatureConfig fc;
    fc.K = 2;
    fc.budget = 100;
    fc.ref_trivial = 0.0;
    fc.ref_rl = 10.0;
    fc.target_species = 5;
    fc.compat_threshold0 = 2.0;
    fc.conns0 = 4;
    aos::FeatureExtractor fx(fc);
    REQUIRE(fx.dim() == 18);
    neat::GenerationReport rep;
    rep.stats.stdev = 3.0;
    rep.stats.num_species = 10;
    rep.stats.largest_species_share = 0.4;
    rep.stats.compat_threshold = 4.0;
    rep.stats.mean_conns = 9.0;
    rep.stats.mean_hidden = 1.0;
    rep.stats.champion_conns = 4;
    rep.stats.champion_hidden = 0;
    rep.op_rank = std::vector<neat::OpRank>(2);
    REQUIRE_THROWS(fx.addGeneration(rep));   // faltan las validaciones
    rep.champion_validation = 2.0;
    rep.best_validation = 2.0;
    rep.evaluations = 10;
    fx.addGeneration(rep);
    auto o = fx.closeWindow();
    REQUIRE(o.size() == 18);
    REQUIRE_THAT(o[0], WithinAbs(0.1, 1e-6));    // τ
    REQUIRE_THAT(o[1], WithinAbs(0.2, 1e-6));    // V*
    REQUIRE_THAT(o[4], WithinAbs(0.0, 1e-6));    // ΔV* en la primera ventana
    REQUIRE_THAT(o[6], WithinAbs(0.3, 1e-6));    // dispersión
    REQUIRE_THAT(o[7], WithinAbs(2.0, 1e-6));    // especies / objetivo
    REQUIRE_THAT(o[9], WithinAbs(std::log(2.0), 1e-6));
    REQUIRE_THAT(o[10], WithinAbs(std::log(10.0 / 5.0), 1e-6));
    // segunda ventana: dos generaciones, mejora de V*, hijos del operador 0 con percentil medio 0,9 (1 fallido de 2)
    rep.champion_validation = 4.0;
    rep.best_validation = 4.0;
    rep.evaluations = 20;
    rep.op_rank[0] = {2, 1.8, 2, 1};
    fx.addGeneration(rep);
    rep.champion_validation = 3.0;
    rep.evaluations = 30;
    rep.op_rank[0] = {0, 0.0, 0, 0};
    fx.addGeneration(rep);
    o = fx.closeWindow();
    REQUIRE_THAT(o[0], WithinAbs(0.3, 1e-6));
    REQUIRE_THAT(o[1], WithinAbs(0.4, 1e-6));
    REQUIRE_THAT(o[2], WithinAbs(0.35, 1e-6));   // media de 0,4 y 0,3
    REQUIRE_THAT(o[4], WithinAbs(0.2, 1e-6));    // ΔV*
    REQUIRE_THAT(o[5], WithinAbs(0.1, 1e-6));    // estancamiento: 0,3 − 0,2
    REQUIRE_THAT(o[14], WithinAbs(0.5 * (0.9 - 0.5), 1e-6));   // crédito del operador 0
    REQUIRE_THAT(o[15], WithinAbs(0.0, 1e-6));                 // operador 1 sin hijos: sin cambio
    REQUIRE_THAT(o[16], WithinAbs(0.5 * 0.5, 1e-6));           // fallos del operador 0
}

TEST_CASE("evaluatePopulations da el mismo fitness que evaluatePopulation", "[ppo][neat_run]") {
    neat::NeatConfig ncfg;
    ncfg.population_size = 12;
    eval::EvalConfig ecfg;
    auto env = envs::makeEnv("pendulum");
    neat::Population a(ncfg, eval::numInputs(*env), eval::numOutputs(*env), 3);
    neat::Population b(ncfg, eval::numInputs(*env), eval::numOutputs(*env), 4);
    auto ga = a.genomes(), gb = b.genomes();
    const auto sa = eval::episodeSeeds(3, 2, 0), sb = eval::episodeSeeds(4, 2, 0);
    eval::evaluatePopulation(a.genomes(), ncfg, "pendulum", {}, sa, ecfg);
    eval::evaluatePopulation(b.genomes(), ncfg, "pendulum", {}, sb, ecfg);
    std::vector<eval::PopulationJob> jobs(2);
    jobs[0] = {&ga, &ncfg, "pendulum", {}, &sa, &ecfg};
    jobs[1] = {&gb, &ncfg, "pendulum", {}, &sb, &ecfg};
    eval::evaluatePopulations(jobs);
    for (size_t i = 0; i < ga.size(); i++) REQUIRE(ga[i].fitness == a.genomes()[i].fitness);
    for (size_t i = 0; i < gb.size(); i++) REQUIRE(gb[i].fitness == b.genomes()[i].fitness);
}

TEST_CASE("NeatRun repite el ciclo de neat_control con rotación, bit a bit", "[ppo][neat_run]") {
    const auto cfg = configChica();
    const std::vector<double> P1{0.8, 0.12, 0.08}, P2{0.5, 0.3, 0.2};
    const uint64_t seed = 5;
    const auto bounds = aos::decisionBoundaries(cfg.generations, cfg.T);
    const int g_switch = bounds[1] + 1;   // desde la decisión 1 se usa P2

    // Referencia: el ciclo de neat_control (--resample-train, --p P1 --p2 P2 --p-switch g_switch)
    auto probe = envs::makeEnv(cfg.env_name, envs::EnvOptions{1.0, true});
    neat::Population pop(cfg.ncfg, eval::numInputs(*probe), eval::numOutputs(*probe), seed);
    const auto val = eval::episodeSeeds(seed, cfg.test_episodes, 1);
    double best_val = -std::numeric_limits<double>::infinity();
    neat::Genome best_genome;
    std::vector<neat::PopulationStats> ref_stats;
    std::vector<double> ref_val;
    for (int gen = 0; gen < cfg.generations; gen++) {
        eval::evaluatePopulation(pop.genomes(), cfg.ncfg, cfg.env_name, cfg.env_opt, eval::episodeSeeds(seed, cfg.episodes, 1000 + gen), cfg.ecfg);
        const auto& G = pop.genomes();
        size_t champ = 0;
        for (size_t i = 1; i < G.size(); i++)
            if (G[i].fitness > G[champ].fitness) champ = i;
        const double ct = eval::evaluateGenome(G[champ], cfg.ncfg, *probe, val, cfg.ecfg);
        if (ct > best_val) {
            best_val = ct;
            best_genome = G[champ];
        }
        ref_val.push_back(ct);
        ref_stats.push_back(pop.endGeneration().stats);
        if (gen == cfg.generations - 1) break;
        pop.reproduce(gen + 1 >= g_switch ? P2 : P1);
    }
    const double ref_final = eval::evaluateGenome(best_genome, cfg.ncfg, *probe, eval::episodeSeeds(seed, cfg.final_episodes, 2), cfg.ecfg);

    // NeatRun por ventanas de decisión
    aos::NeatRun run(cfg, seed);
    run.startSequential();
    REQUIRE(run.t() == 0);
    REQUIRE(static_cast<int>(run.obs().size()) == run.obsDim());
    REQUIRE(std::isfinite(run.phi()));
    std::vector<neat::PopulationStats> st{run.lastReport().stats};
    std::vector<double> vals{run.lastReport().champion_validation};
    for (int t = 0; t < cfg.T; t++) {
        const auto& P = t >= 1 ? P2 : P1;
        // registrar cada generación cerrada en la ventana
        while (run.needsGeneration()) {
            run.reproduce(P);
            eval::evaluatePopulation(run.genomes(), cfg.ncfg, cfg.env_name, cfg.env_opt, run.trainingSeeds(), cfg.ecfg);
            run.closeGeneration();
            st.push_back(run.lastReport().stats);
            vals.push_back(run.lastReport().champion_validation);
        }
        run.closeWindow();
        REQUIRE(run.t() == t + 1);
    }
    REQUIRE(run.done());
    REQUIRE_THROWS(run.reproduce(P1));
    REQUIRE(st.size() == ref_stats.size());
    for (size_t g = 0; g < st.size(); g++) {
        REQUIRE(st[g].best == ref_stats[g].best);
        REQUIRE(st[g].mean == ref_stats[g].mean);
        REQUIRE(st[g].stdev == ref_stats[g].stdev);
        REQUIRE(st[g].num_species == ref_stats[g].num_species);
        REQUIRE(st[g].compat_threshold == ref_stats[g].compat_threshold);
        REQUIRE(st[g].mean_conns == ref_stats[g].mean_conns);
        REQUIRE(vals[g] == ref_val[g]);
    }
    REQUIRE(run.bestValidation() == best_val);
    REQUIRE(run.finalTest() == ref_final);
}

TEST_CASE("NeatPool en paralelo = las mismas corridas por separado", "[ppo][neat_run]") {
    const auto cfg = configChica();
    const std::vector<double> P{0.7, 0.2, 0.1};
    aos::NeatPool pool(cfg, 3, 11, 0.5, 10.0, true);
    pool.start({0, 1, 2});
    REQUIRE_THROWS(pool.start({1}));
    std::vector<aos::NeatRun> solas;
    for (int s = 0; s < 3; s++) {
        solas.emplace_back(cfg, 11 + s);
        solas.back().startSequential();
        REQUIRE(pool.obs(s) == solas.back().obs());
        REQUIRE(pool.phi(s) == solas.back().phi());
    }
    for (int t = 0; t < cfg.T; t++) {
        pool.advance(std::vector<std::vector<double>>(3, P));
        for (int s = 0; s < 3; s++) {
            solas[s].advanceSequential(P);
            REQUIRE(pool.t(s) == t + 1);
            REQUIRE(pool.obs(s) == solas[s].obs());
            REQUIRE(pool.phi(s) == solas[s].phi());
        }
    }
    const auto fin = pool.takeFinished();
    REQUIRE(fin.size() == 3);
    for (int s = 0; s < 3; s++) {
        REQUIRE(fin[s].seed == static_cast<uint64_t>(11 + s));
        REQUIRE(fin[s].final_test == solas[s].finalTest());
    }
    REQUIRE_THROWS(pool.advance(std::vector<std::vector<double>>(3, P)));   // una decisión de más
    REQUIRE(pool.decisions() == 3L * cfg.T);
}
