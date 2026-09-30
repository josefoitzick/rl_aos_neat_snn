#include <CLI/CLI.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>
#include "../../include/core/network.hpp"
#include "../../include/core/simulator.hpp"

static Network* buildBaseNetwork(double dt, double w) {
    Network* net = new Network(dt);
    int in1 = net->addInputNeuron(NeuronType::REGULAR_SPIKING);
    int in2 = net->addInputNeuron(NeuronType::REGULAR_SPIKING);
    int h1  = net->addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h2  = net->addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int out = net->addOutputNeuron(NeuronType::REGULAR_SPIKING);
    net->addSynapse(in1, h1,  true, w);
    net->addSynapse(in1, h2,  true, w);
    net->addSynapse(in2, h1,  true, w);
    net->addSynapse(in2, h2,  true, w);
    net->addSynapse(h1,  out, true, w);
    net->addSynapse(h2,  out, true, w);
    return net;
}

int main(int argc, char** argv) {
    CLI::App app{"SNN parallel simulation demo"};

    double duration   = 100.0;
    double dt         = 1.0;
    double weight     = 20.0;
    int    pop_size   = 8;

    app.add_option("-t,--duration", duration,  "Duración de simulación (ms)")->check(CLI::PositiveNumber);
    app.add_option("-d,--timestep", dt,         "Timestep (ms)")->check(CLI::PositiveNumber);
    app.add_option("-w,--weight",   weight,     "Peso de todas las sinapsis")->check(CLI::Number);
    app.add_option("-p,--pop",      pop_size,   "Tamaño de la población")->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);

    // ── Construir población ────────────────────────────────────────────────
    SimulationConfig cfg;
    cfg.dt                  = dt;
    cfg.evaluation_duration = duration;

    Simulator sim(cfg);

    // Red base: igual para todos; cada clon es independiente
    Network* base = buildBaseNetwork(dt, weight);
    sim.addNetwork(base, /*take_ownership=*/true);

    for (int i = 1; i < pop_size; i++)
        sim.cloneNetwork(0);

    // Aplicar una mutación distinta a cada individuo para que no sean idénticas
    for (int i = 0; i < pop_size; i++) {
        // Alterna tipo de neurona oculta según el índice
        NeuronType t = (i % 2 == 0) ? NeuronType::FAST_SPIKING
                                     : NeuronType::CHATTERING;
        sim.getNetwork(i)->setNeuronType(2, t);   // neurona oculta h1
    }

    std::cout << "Población: " << sim.getNetworkCount() << " redes\n";
    std::cout << "Hebras OpenMP disponibles: " << omp_get_max_threads() << "\n\n";

    std::vector<double> inputs = {10.0, 10.0};

    // ── Evaluación secuencial (referencia) ────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<SimulationResult> seq_results(pop_size);
    for (int i = 0; i < pop_size; i++)
        seq_results[i] = sim.simulate(i, inputs, duration);

    double seq_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // ── Evaluación paralela ────────────────────────────────────────────────
    sim.resetAll();

    auto t1 = std::chrono::high_resolution_clock::now();

    auto par_results = sim.simulateAll(inputs, duration);

    double par_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t1).count();

    // ── Resultados ─────────────────────────────────────────────────────────
    std::cout << "Red | Spikes (seq) | Spikes (par)\n";
    std::cout << "----+-------------+-------------\n";
    for (int i = 0; i < pop_size; i++) {
        std::cout << "  " << i
                  << " |     " << static_cast<int>(seq_results[i].total_spikes)
                  << "       |     " << static_cast<int>(par_results[i].total_spikes)
                  << "\n";
    }

    std::cout << "\nTiempo secuencial: " << seq_ms << " ms\n";
    std::cout << "Tiempo paralelo:   " << par_ms << " ms\n";
    std::cout << "Speedup:           " << seq_ms / par_ms << "x\n";

    return 0;
}
