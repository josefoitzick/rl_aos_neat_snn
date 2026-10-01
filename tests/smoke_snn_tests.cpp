// Test de humo: el simulador SNN compila y funciona integrado en el repo de la tesis.
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "core/network.hpp"
#include "core/simulator.hpp"
#include "encoding/rlEncoder.hpp"

// 1 variable de observación → 2 entradas (SMALL: [neg, pos]) → 1 salida
static Network* buildNet() {
    Network* net = new Network(1.0);
    int in_neg = net->addInputNeuron(NeuronType::TONIC_SPIKING);
    int in_pos = net->addInputNeuron(NeuronType::TONIC_SPIKING);
    int out    = net->addOutputNeuron(NeuronType::TONIC_SPIKING);
    net->addSynapse(in_neg, out, true, 20.0);
    net->addSynapse(in_pos, out, true, 20.0);
    return net;
}

TEST_CASE("SNN integrada: una observación codificada con SMALL produce spikes de salida") {
    Simulator sim;
    sim.addNetwork(buildNet(), /*take_ownership=*/true);

    RLEncoder enc(RLEncoder::EncodingType::SMALL);
    auto currents = enc.encodeObservationSmall({0.8}, /*weight=*/20.0);

    auto result = sim.simulate(0, currents, /*duration=*/50.0);
    REQUIRE(result.output_spike_times.size() == 1);
    REQUIRE(result.total_spikes > 0);
}

TEST_CASE("SNN integrada: simulación paralela idéntica a la secuencial") {
    Simulator sim;
    sim.addNetwork(buildNet(), /*take_ownership=*/true);
    for (int i = 1; i < 8; i++) sim.cloneNetwork(0);

    std::vector<double> currents = {0.0, 16.0};
    std::vector<double> seq;
    for (int i = 0; i < 8; i++) seq.push_back(sim.simulate(i, currents, 100.0).total_spikes);

    auto par = sim.simulateAll(currents, 100.0);
    for (int i = 0; i < 8; i++) REQUIRE(par[i].total_spikes == seq[i]);
}
