// Tests de entornos (RLtools) y evaluación (src/envs, src/eval).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <random>
#include "core/network.hpp"
#include "eval/evaluator.hpp"
#include "neat/innovation.hpp"
#include "neat/population.hpp"

using namespace neat;

TEST_CASE("Entornos: todos reproducibles por semilla, observación en [-1, 1] y dimensiones coherentes") {
    for (const std::string& name : envs::envNames()) {
        INFO("entorno " << name);
        auto a = envs::makeEnv(name), b = envs::makeEnv(name);
        const size_t O = a->numAgents() * a->obsDim(), A = a->numAgents() * a->actionDim();
        REQUIRE(a->maxSteps() > 0);
        a->reset(123); b->reset(123);
        REQUIRE(a->observe().size() == O);
        REQUIRE(a->rawObserve().size() == O);
        REQUIRE(a->observe() == b->observe());
        std::mt19937_64 rng(5);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        for (int t = 0; t < 30; t++) {
            for (double x : a->observe()) { REQUIRE(x >= -1.0); REQUIRE(x <= 1.0); }
            std::vector<double> act(A);
            for (double& x : act) x = U(rng);
            auto ra = a->step(act), rb = b->step(act);
            REQUIRE(ra.reward == rb.reward);
            REQUIRE(std::isfinite(ra.reward));
            REQUIRE(ra.terminated == rb.terminated);
            if (ra.terminated) break;
        }
    }
}

TEST_CASE("Entornos: semillas distintas dan episodios distintos (incluye parámetros, p. ej. banderas)") {
    for (const std::string& name : envs::envNames()) {
        if (name == "memory") continue;   // estado inicial casi determinista: un bit al azar
        INFO("entorno " << name);
        auto a = envs::makeEnv(name);
        a->reset(1);
        auto o1 = a->rawObserve();
        a->reset(2);
        REQUIRE(o1 != a->rawObserve());
    }
}

TEST_CASE("Entornos: multiagente con política compartida (bottleneck)") {
    auto env = envs::makeEnv("bottleneck");
    REQUIRE(env->numAgents() == 5);
    REQUIRE(env->actionDim() == 2);
    REQUIRE(env->obsDim() == 12);
    NeatConfig cfg;
    InnovationTracker tr(2 * 12 + 2 * 2);
    Rng rng(3);
    Genome g = Genome::minimal(0, 2 * 12, 2 * 2, cfg, tr, rng);
    eval::EvalConfig ecfg;
    auto seeds = eval::episodeSeeds(1, 2, 0);
    double r1 = eval::evaluateGenome(g, cfg, *env, seeds, ecfg);
    REQUIRE(std::isfinite(r1));
    REQUIRE(r1 == eval::evaluateGenome(g, cfg, *env, seeds, ecfg));
}

TEST_CASE("Evaluación: red sin conexiones (salidas mudas) equivale a acción 0") {
    // Pesos 0 → ninguna salida dispara → acción 0
    NeatConfig zero;
    zero.weight_init_range = 0.0;
    InnovationTracker tr(6 + 2);
    Rng rng(1);
    Genome silent = Genome::minimal(0, 6, 2, zero, tr, rng);

    auto env = envs::makeEnv("pendulum");
    eval::EvalConfig ecfg;
    auto seeds = eval::episodeSeeds(5, 3, 0);
    double snn_ret = eval::evaluateGenome(silent, zero, *env, seeds, ecfg);

    double ref = 0.0;
    for (uint64_t s : seeds) {
        env->reset(s);
        for (int t = 0; t < env->maxSteps(); t++) {
            auto r = env->step({0.0});
            ref += r.reward;
            if (r.terminated) break;
        }
    }
    ref /= seeds.size();
    REQUIRE_THAT(snn_ret, Catch::Matchers::WithinAbs(ref, 1e-9));   // mismo valor; otro orden de suma
}

TEST_CASE("Evaluación: determinista y sin dependencia del número de hilos") {
    NeatConfig cfg;
    cfg.population_size = 24;
    Population pop(cfg, 6, 2, 9);          // pendulum: 3 obs → 6 entradas, 1 acción → 2 salidas
    auto seeds = eval::episodeSeeds(9, 2, 0);
    eval::EvalConfig ecfg;

    auto run = [&](int threads) {
        omp_set_num_threads(threads);
        std::vector<Genome> G = pop.genomes();
        eval::evaluatePopulation(G, cfg, "pendulum", envs::EnvOptions(), seeds, ecfg);
        std::vector<double> f;
        for (const auto& g : G) f.push_back(g.fitness);
        return f;
    };
    auto one = run(1);
    REQUIRE(one == run(1));
    REQUIRE(one == run(4));
}

TEST_CASE("Semillas de episodio: reproducibles y distintas entre streams") {
    auto a = eval::episodeSeeds(3, 5, 0);
    REQUIRE(a == eval::episodeSeeds(3, 5, 0));
    REQUIRE(a != eval::episodeSeeds(3, 5, 1));
    REQUIRE(a != eval::episodeSeeds(4, 5, 0));
}

// Implementación de referencia de la versión anterior del evaluador (reinicio en cada paso, un agente)
static double referenceEpisode(Network& net, envs::ControlEnv& env, uint64_t seed, const eval::EvalConfig& cfg) {
    const std::vector<int> outputs = net.getOutputNeuronIds();
    const int steps = static_cast<int>(cfg.step_ms / net.getTimestep());
    std::vector<double> currents(2 * env.obsDim()), action(env.actionDim());
    std::vector<int> count(outputs.size());
    env.reset(seed);
    double ret = 0.0;
    for (int t = 0; t < env.maxSteps(); t++) {
        const std::vector<double> obs = env.observe();
        for (int i = 0; i < env.obsDim(); i++) {
            currents[2 * i]     = obs[i] < 0.0 ? -obs[i] * cfg.input_gain : 0.0;
            currents[2 * i + 1] = obs[i] >= 0.0 ? obs[i] * cfg.input_gain : 0.0;
        }
        net.fastReset();
        net.setInputOffsets(currents);
        std::fill(count.begin(), count.end(), 0);
        for (int k = 0; k < steps; k++) {
            net.step();
            for (size_t o = 0; o < outputs.size(); o++)
                if (net.getNeuron(outputs[o]).hasSpiked()) count[o]++;
        }
        for (int j = 0; j < env.actionDim(); j++)
            action[j] = std::clamp((count[2 * j] - count[2 * j + 1]) / cfg.spikes_full_action, -1.0, 1.0);
        auto r = env.step(action);
        ret += r.reward;
        if (r.terminated) break;
    }
    return ret;
}

TEST_CASE("Evaluación: sin keep_state reproduce exactamente al evaluador anterior; con keep_state cambia") {
    NeatConfig cfg;
    cfg.population_size = 12;
    Population pop(cfg, 6, 2, 21);
    // Algunas mutaciones estructurales para tener ocultas y recurrencia
    for (int gen = 0; gen < 5; gen++) {
        for (Genome& g : pop.genomes()) g.fitness = static_cast<double>(g.numEnabledConns());
        pop.endGeneration();
        pop.reproduce({0.4, 0.3, 0.3});
    }
    auto env = envs::makeEnv("pendulum");
    auto seeds = eval::episodeSeeds(4, 2, 0);
    eval::EvalConfig reactive, memory;
    memory.keep_state = true;
    int differ = 0;
    for (const Genome& g : pop.genomes()) {
        double ref = 0.0;
        auto net = g.buildNetwork(cfg);
        for (uint64_t s : seeds) ref += referenceEpisode(*net, *env, s, reactive);
        ref /= seeds.size();
        REQUIRE(eval::evaluateGenome(g, cfg, *env, seeds, reactive) == ref);
        differ += eval::evaluateGenome(g, cfg, *env, seeds, memory) != ref;
    }
    REQUIRE(differ > 0);
}

TEST_CASE("Evaluación: ganancia de entrada (E1) = 2 en todas las entradas equivale a duplicar input_gain") {
    NeatConfig cfg;
    cfg.population_size = 8;
    Population pop(cfg, 6, 2, 31);
    for (int gen = 0; gen < 3; gen++) {
        for (Genome& g : pop.genomes()) g.fitness = static_cast<double>(g.numEnabledConns());
        pop.endGeneration();
        pop.reproduce({0.6, 0.2, 0.2});
    }
    auto env = envs::makeEnv("pendulum");
    auto seeds = eval::episodeSeeds(5, 2, 0);
    eval::EvalConfig base, doubled;
    doubled.input_gain = 2.0 * base.input_gain;
    NeatConfig fixed2 = cfg;
    fixed2.gain_min = fixed2.gain_max = 2.0;   // la mutación queda forzada a ganancia 2
    Rng rng(3);
    int checked = 0;
    for (const Genome& g0 : pop.genomes()) {
        Genome g = g0;
        for (int k = 0; k < 200; k++) g.mutateInputGain(fixed2, rng);
        auto gains = g.inputGains();
        REQUIRE(gains.size() == 6);
        bool all2 = true;
        for (double x : gains) all2 = all2 && x == 2.0;
        if (!all2) continue;
        checked++;
        REQUIRE(eval::evaluateGenome(g, cfg, *env, seeds, base) == eval::evaluateGenome(g0, cfg, *env, seeds, doubled));
    }
    REQUIRE(checked >= 4);
}
