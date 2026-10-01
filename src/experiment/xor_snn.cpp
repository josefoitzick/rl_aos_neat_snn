// Verificación de NEAT + SNN en XOR (análogo a Stanley & Miikkulainen 2002, §4.2).
//
// Entradas: 2 neuronas; bit 1 → corriente constante `--current`, bit 0 → 0.
// Salida: 1 neurona; se cuentan sus spikes en una ventana de `--window` ms.
//   Predicción = 1 si dispara al menos una vez.
//   Error por caso: 1 - min(n, n_obj)/n_obj si el objetivo es 1; min(n, n_obj)/n_obj si es 0.
//   Fitness = (4 - Σ error)² (como en el paper), máximo 16.
// No hay neurona de sesgo, así que (0,0) → 0 sale gratis; (1,1) → 0 exige una
// inhibición que solo aparezca con ambas entradas activas (necesita estructura oculta).
#include <CLI/CLI.hpp>
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "aos/factory.hpp"
#include "neat/population.hpp"
#include "neat/selector.hpp"

using namespace neat;

struct XorParams {
    double current = 20.0;
    double window = 100.0;
    int target_spikes = 3;
};

struct XorResult { double fitness; bool solved; };

static XorResult evaluateXor(const Genome& g, const NeatConfig& cfg, const XorParams& xp) {
    static const int cases[4][3] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    auto net = g.buildNetwork(cfg);
    const int out = net->getOutputNeuronIds()[0];
    double err = 0.0;
    bool solved = true;
    for (const auto& c : cases) {
        net->fastReset();
        net->setInputOffsets({c[0] * xp.current, c[1] * xp.current});
        int n = 0;
        for (int t = 0; t < static_cast<int>(xp.window / cfg.dt); t++) {
            net->step();
            if (net->getNeuron(out).hasSpiked()) n++;
        }
        double y = std::min(n, xp.target_spikes) / static_cast<double>(xp.target_spikes);
        err += c[2] ? 1.0 - y : y;
        solved = solved && ((n > 0) == (c[2] == 1));
    }
    return {(4.0 - err) * (4.0 - err), solved};
}

int main(int argc, char** argv) {
    CLI::App app{"NEAT + SNN (Izhikevich) en XOR"};
    uint64_t seed = 1;
    int runs = 10, generations = 200;
    NeatConfig cfg;
    XorParams xp;
    bool verbose = false;
    std::string selector_name = "static";
    aos::AdaptivePursuitConfig apc;
    app.add_option("--seed", seed, "Semilla de la primera corrida (las siguientes usan seed+1, …)");
    app.add_option("--runs", runs, "Corridas independientes");
    app.add_option("--generations", generations, "Máximo de generaciones por corrida");
    app.add_option("--pop", cfg.population_size, "Tamaño de población");
    app.add_option("--current", xp.current, "Corriente de entrada para bit 1");
    app.add_option("--window", xp.window, "Ventana de simulación por caso (ms)");
    app.add_flag("-v,--verbose", verbose, "Imprimir cada generación");
    app.add_option("--selector", selector_name, "static | ap")->check(CLI::IsMember({"static", "ap"}));
    app.add_option("--ap-alpha", apc.alpha, "AP: tasa de adaptación de Q");
    app.add_option("--ap-beta", apc.beta, "AP: tasa de persecución de P");
    app.add_option("--ap-pmin", apc.p_min, "AP: probabilidad mínima");
    app.add_flag("--ap-extreme", apc.extreme_credit, "AP: crédito extremo (máximo) en vez de media");
    CLI11_PARSE(app, argc, argv);

    // Cada corrida parte con un selector nuevo (Adaptive Pursuit arranca desde el P de Stanley)
    auto makeSelector = [&]() { return aos::makeSelector(selector_name, apc); };
    std::printf("Selector: %s  hilos OpenMP: %d\n", makeSelector()->name().c_str(), omp_get_max_threads());

    int solved_runs = 0;
    std::vector<int> gens_to_solve, hidden_at_solve, conns_at_solve;
    for (int r = 0; r < runs; r++) {
        Population pop(cfg, 2, 1, seed + r);
        auto sel = makeSelector();
        int solved_gen = -1;
        const Genome* solver = nullptr;
        for (int gen = 0; gen < generations && solved_gen < 0; gen++) {
            auto& G = pop.genomes();
            std::vector<char> ok(G.size(), 0);
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < static_cast<int>(G.size()); i++) {
                XorResult res = evaluateXor(G[i], cfg, xp);
                G[i].fitness = res.fitness;
                ok[i] = res.solved;
            }
            for (size_t i = 0; i < G.size(); i++)
                if (ok[i]) { solved_gen = gen; solver = &G[i]; break; }

            GenerationReport rep = pop.endGeneration();
            if (solved_gen >= 0) break;
            sel->update(rep);
            auto P = sel->probabilities();
            if (verbose)
                std::printf("  run %d gen %3d  best %6.3f  mean %6.3f  especies %2d  campeón: %d ocultas, %d conexiones  "
                            "P=(%.3f %.3f %.3f)\n",
                            r, gen, rep.stats.best, rep.stats.mean, rep.stats.num_species,
                            rep.stats.champion_hidden, rep.stats.champion_conns, P[0], P[1], P[2]);
            pop.reproduce(P);
        }
        if (solved_gen >= 0) {
            solved_runs++;
            gens_to_solve.push_back(solved_gen);
            hidden_at_solve.push_back(solver->numHidden());
            conns_at_solve.push_back(solver->numEnabledConns());
            std::printf("run %2d (seed %llu): RESUELTO en generación %3d — %d ocultas, %d conexiones habilitadas\n",
                        r, static_cast<unsigned long long>(seed + r), solved_gen,
                        solver->numHidden(), solver->numEnabledConns());
        } else {
            std::printf("run %2d (seed %llu): no resuelto en %d generaciones (mejor fitness %.3f / 16)\n",
                        r, static_cast<unsigned long long>(seed + r), generations, pop.bestEver().fitness);
        }
    }

    std::printf("\nResueltas %d/%d", solved_runs, runs);
    if (!gens_to_solve.empty()) {
        auto mean = [](const std::vector<int>& v) {
            double s = 0; for (int x : v) s += x; return s / v.size();
        };
        std::printf("  |  generaciones media %.1f  |  ocultas media %.2f  |  conexiones media %.2f",
                    mean(gens_to_solve), mean(hidden_at_solve), mean(conns_at_solve));
    }
    std::printf("\n");
    return 0;
}
