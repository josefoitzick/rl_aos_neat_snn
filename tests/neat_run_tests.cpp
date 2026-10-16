// Tests del paso 2 del agente PPO (docs/diseno_ppo.md §3.3 y §5): crédito por rango, rasgos de estado, evaluador
// de varias corridas y NEAT como entorno (NeatRun / NeatPool). La comparación contra el binario de neat_control
// está en scripts/verificar_neat_run.sh; aquí NeatRun se compara con el mismo ciclo escrito a mano.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
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

TEST_CASE("NeatPool con entornos mezclados = las mismas corridas por separado", "[ppo][neat_run]") {
    auto pend = configChica();
    auto acro = configChica();
    acro.env_name = "acrobot-swingup";
    acro.ref_trivial = -18.04437;
    acro.ref_rl = -6.8019405;
    const std::vector<aos::NeatRunConfig> cfgs{pend, pend, acro, acro};
    const std::vector<double> P{0.6, 0.3, 0.1};
    aos::NeatPool pool(cfgs, 21, 0.5, 10.0, true);
    pool.start({0, 1, 2, 3});
    std::vector<aos::NeatRun> solas;
    for (int s = 0; s < 4; s++) {
        solas.emplace_back(cfgs[s], 21 + s);
        solas.back().startSequential();
    }
    for (int t = 0; t < pend.T; t++) {
        pool.advance(std::vector<std::vector<double>>(4, P));
        for (int s = 0; s < 4; s++) {
            solas[s].advanceSequential(P);
            REQUIRE(pool.obs(s) == solas[s].obs());
            REQUIRE(pool.phi(s) == solas[s].phi());
        }
    }
    const auto fin = pool.takeFinished();
    REQUIRE(fin.size() == 4);
    for (int s = 0; s < 4; s++) {
        REQUIRE(fin[s].env == cfgs[s].env_name);
        REQUIRE(fin[s].final_test == solas[s].finalTest());
    }
    auto distinto = pend;
    distinto.T = 2;
    REQUIRE_THROWS(aos::NeatPool(std::vector<aos::NeatRunConfig>{pend, distinto}, 1, 0.5, 10.0));
}

// Plan B (docs/experimentos/15_margen_intensidades.md): intensidades de variación por ventana y estado compacto

TEST_CASE("MapaAccion: con a = 0 da exactamente P₀ y las intensidades base, y respeta los rangos", "[ppo][neat_run][planB]") {
    const std::vector<double> P0{0.5, 0.3, 0.2};
    neat::Variacion base;
    base.weight_perturb_power = 5.0;
    base.weight_perturb_prob = 0.98;
    base.mutation_only_fraction = 0.25;
    for (aos::ModoAccion modo : {aos::ModoAccion::P, aos::ModoAccion::Intensidades, aos::ModoAccion::Ambas}) {
        const auto m = aos::MapaAccion::crear(modo, P0, 3.0, base);
        REQUIRE(m.dim() == (modo == aos::ModoAccion::P ? 3 : modo == aos::ModoAccion::Intensidades ? 3 : 6));
        const std::vector<double> cero(m.dim(), 0.0);
        std::vector<double> P;
        neat::Variacion v;
        m.decodificar(cero.data(), P, v);
        REQUIRE(v.weight_perturb_power == base.weight_perturb_power);
        REQUIRE(v.weight_perturb_prob == base.weight_perturb_prob);
        REQUIRE(v.mutation_only_fraction == base.mutation_only_fraction);
        if (modo == aos::ModoAccion::Intensidades) REQUIRE(P == P0);
        else for (size_t k = 0; k < 3; k++) REQUIRE_THAT(P[k], WithinAbs(P0[k], 1e-12));
        if (modo == aos::ModoAccion::P) continue;
        // extremos: dentro de los rangos de Optuna
        for (double s : {-1e6, 1e6}) {
            std::vector<double> a(m.dim(), s);
            m.decodificar(a.data(), P, v);
            REQUIRE(v.weight_perturb_power >= 0.5);
            REQUIRE(v.weight_perturb_power <= 15.0);
            REQUIRE(1.0 - v.weight_perturb_prob >= 0.0);
            REQUIRE(1.0 - v.weight_perturb_prob <= 0.2 + 1e-12);
            REQUIRE(v.mutation_only_fraction >= 0.1);
            REQUIRE(v.mutation_only_fraction <= 0.6);
        }
        std::vector<double> a(m.dim(), 0.0);
        a[m.dim() - 3] = 1.0;   // más potencia
        m.decodificar(a.data(), P, v);
        REQUIRE(v.weight_perturb_power > base.weight_perturb_power);
        a[m.dim() - 3] = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS(m.decodificar(a.data(), P, v));
    }
}

TEST_CASE("NeatRun con las intensidades base es idéntico a sin intensidades; con otras, cambia", "[ppo][neat_run][planB]") {
    auto cfg = configChica();
    const uint64_t seed = 11;
    const std::vector<double> P{0.8, 0.1, 0.1};
    const neat::Variacion base = neat::variacionDe(cfg.ncfg);
    neat::Variacion otra = base;
    otra.weight_perturb_power = 0.7;
    otra.mutation_only_fraction = 0.5;
    aos::NeatRun a(cfg, seed), b(cfg, seed), c(cfg, seed);
    a.startSequential();
    b.startSequential();
    c.startSequential();
    for (int t = 0; t < cfg.T; t++) {
        while (a.needsGeneration()) {
            a.reproduce(P);
            b.reproduce(P, &base);
            c.reproduce(P, &otra);
            for (aos::NeatRun* r : {&a, &b, &c}) {
                eval::evaluatePopulation(r->genomes(), cfg.ncfg, cfg.env_name, cfg.env_opt, r->trainingSeeds(), cfg.ecfg);
                r->closeGeneration();
            }
            REQUIRE(a.lastReport().stats.best == b.lastReport().stats.best);
            REQUIRE(a.lastReport().stats.mean == b.lastReport().stats.mean);
            REQUIRE(a.lastReport().champion_validation == b.lastReport().champion_validation);
        }
        a.closeWindow();
        b.closeWindow();
        c.closeWindow();
        REQUIRE(a.obs() == b.obs());
    }
    REQUIRE(a.finalTest() == b.finalTest());
    REQUIRE(c.variacion().weight_perturb_power == 0.7);
    REQUIRE(c.lastReport().stats.mean != a.lastReport().stats.mean);   // control negativo: otras intensidades, otra corrida
}

TEST_CASE("Estado compacto: 9 rasgos finitos; la potencia relativa sigue a la intensidad vigente", "[ppo][neat_run][planB]") {
    auto cfg = configChica();
    cfg.estado_compacto = true;
    aos::NeatRun r(cfg, 3);
    REQUIRE(r.obsDim() == aos::FeatureExtractor::DIM_COMPACTO);
    r.startSequential();
    REQUIRE(static_cast<int>(r.obs().size()) == aos::FeatureExtractor::DIM_COMPACTO);
    REQUIRE(r.obs()[8] == 0.0f);   // potencia base
    neat::Variacion v = neat::variacionDe(cfg.ncfg);
    v.weight_perturb_power = cfg.ncfg.weight_perturb_power * 2.0;
    while (r.needsGeneration()) {
        r.reproduce({0.8, 0.1, 0.1}, &v);
        eval::evaluatePopulation(r.genomes(), cfg.ncfg, cfg.env_name, cfg.env_opt, r.trainingSeeds(), cfg.ecfg);
        r.closeGeneration();
    }
    r.closeWindow();
    for (float x : r.obs()) REQUIRE(std::isfinite(x));
    REQUIRE_THAT(r.obs()[8], WithinAbs(std::log(2.0), 1e-6));
    aos::NeatPool pool(cfg, 2, 1, 0.5, 10.0);
    REQUIRE(pool.obsDim() == aos::FeatureExtractor::DIM_COMPACTO);
}

// Estado SNN (docs/experimentos/18_analisis_efectos_acciones.md): medir la actividad no cambia nada y da fracciones válidas

TEST_CASE("Actividad de la SNN: medirla no cambia el retorno y da fracciones en [0, 1]", "[ppo][neat_run][snn]") {
    auto cfg = configChica();
    aos::NeatRun r(cfg, 5);
    r.startSequential();
    const neat::Genome& g = r.champion();
    const auto seeds = eval::episodeSeeds(5, 3, 1);
    eval::ActividadSNN act;
    eval::GenomeJob con, sin;
    for (eval::GenomeJob* j : {&con, &sin}) {
        j->genome = &g;
        j->ncfg = &cfg.ncfg;
        j->env_name = cfg.env_name;
        j->env_opt = cfg.env_opt;
        j->seeds = &seeds;
        j->cfg = &cfg.ecfg;
    }
    con.actividad = &act;
    const auto v = eval::evaluateGenomes({con, sin});
    REQUIRE(v[0] == v[1]);
    REQUIRE(act.pasos_snn > 0);
    REQUIRE(act.acciones > 0);
    const auto red = g.buildNetwork(cfg.ncfg);
    const auto res = eval::resumirActividad(act, *red);
    for (double x : {res.ocultas_mudas, res.ocultas_saturadas, res.entradas_mudas, res.salidas_mudas, res.accion_saturada}) {
        REQUIRE(x >= 0.0);
        REQUIRE(x <= 1.0);
    }
}

TEST_CASE("NeatRun con estado SNN: 10 rasgos finitos y la misma evolución que sin medir", "[ppo][neat_run][snn]") {
    auto cfg = configChica();
    aos::NeatRun a(cfg, 9);
    cfg.estado_snn = true;
    aos::NeatRun b(cfg, 9);
    REQUIRE(b.obsDim() == aos::FeatureExtractor::DIM_SNN);
    a.startSequential();
    b.startSequential();
    const std::vector<double> P{0.8, 0.1, 0.1};
    for (int t = 0; t < cfg.T; t++) {
        a.advanceSequential(P);
        b.advanceSequential(P);
        REQUIRE(static_cast<int>(b.obs().size()) == aos::FeatureExtractor::DIM_SNN);
        for (float x : b.obs()) REQUIRE(std::isfinite(x));
        REQUIRE(a.lastReport().stats.mean == b.lastReport().stats.mean);
        REQUIRE(a.lastReport().champion_validation == b.lastReport().champion_validation);
    }
    REQUIRE(a.finalTest() == b.finalTest());
    aos::NeatPool pool(cfg, 2, 1, 0.5, 10.0);
    REQUIRE(pool.obsDim() == aos::FeatureExtractor::DIM_SNN);
}

TEST_CASE("Medir la actividad por genoma (F1–F4) no cambia el fitness y escribe tasas", "[actividad]") {
    neat::NeatConfig a_cfg, b_cfg;
    a_cfg.population_size = b_cfg.population_size = 12;
    b_cfg.medir_actividad = true;
    eval::EvalConfig ecfg;
    auto env = envs::makeEnv("pendulum");
    neat::Population a(a_cfg, eval::numInputs(*env), eval::numOutputs(*env), 3);
    auto gb = a.genomes(), gc = a.genomes();
    const auto s = eval::episodeSeeds(3, 2, 0);
    eval::evaluatePopulation(a.genomes(), a_cfg, "pendulum", {}, s, ecfg);
    eval::evaluatePopulation(gb, b_cfg, "pendulum", {}, s, ecfg);
    std::vector<eval::PopulationJob> jobs(1);
    jobs[0] = {&gc, &b_cfg, "pendulum", {}, &s, &ecfg};
    eval::evaluatePopulations(jobs);
    for (size_t i = 0; i < gb.size(); i++) {
        REQUIRE(gb[i].fitness == a.genomes()[i].fitness);
        REQUIRE(gc[i].fitness == a.genomes()[i].fitness);
        for (size_t k = 0; k < gb[i].nodes().size(); k++) {
            REQUIRE(a.genomes()[i].nodes()[k].tasa < 0.0);   // sin medir: desconocida
            REQUIRE(gb[i].nodes()[k].tasa >= 0.0);
            REQUIRE(gb[i].nodes()[k].tasa == gc[i].nodes()[k].tasa);
        }
    }
}

// Diagnóstico (no corre por defecto: ctest no lo incluye; correr con `neat_run_tests "[.diag_tasas]"`): cuantiles de la tasa
// de disparo por rol en una corrida corta, para calibrar los umbrales de F1–F4
TEST_CASE("Diagnóstico de tasas por rol", "[.diag_tasas]") {
    for (const char* env_name : {"pendulum-meta", "acrobot-swingup"}) {
        neat::NeatConfig ncfg;
        ncfg.population_size = 60;
        ncfg.medir_actividad = true;
        ncfg.catalog = neat::parseCatalog("mutate_weights,add_connection,add_node,neuron_bias,neuron_dynamics");
        eval::EvalConfig ecfg;
        envs::EnvOptions opt;
        auto env = envs::makeEnv(env_name, opt);
        neat::Population pop(ncfg, eval::numInputs(*env), eval::numOutputs(*env), 7);
        const std::vector<double> P(5, 0.2);
        for (int gen = 0; gen <= 40; gen++) {
            eval::evaluatePopulation(pop.genomes(), ncfg, env_name, opt, eval::episodeSeeds(7, 4, 1000 + gen), ecfg);
            if (gen % 20 == 0) {
                for (neat::NodeRole role : {neat::NodeRole::Input, neat::NodeRole::Output, neat::NodeRole::Hidden}) {
                    std::vector<double> t;
                    for (const auto& g : pop.genomes())
                        for (const auto& n : g.nodes()) if (n.role == role) t.push_back(n.tasa);
                    if (t.empty()) continue;
                    std::sort(t.begin(), t.end());
                    auto q = [&](double f) { return t[std::min(t.size() - 1, static_cast<size_t>(f * t.size()))]; };
                    const double cero = std::count(t.begin(), t.end(), 0.0) / static_cast<double>(t.size());
                    std::printf("%s gen %d rol %d: n %zu, ceros %.2f, q05 %.4f q25 %.4f q50 %.4f q75 %.4f q95 %.4f máx %.4f\n", env_name, gen,
                                static_cast<int>(role), t.size(), cero, q(0.05), q(0.25), q(0.5), q(0.75), q(0.95), t.back());
                }
            }
            pop.endGeneration();
            pop.reproduce(P);
        }
    }
}
