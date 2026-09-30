#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include "../../include/core/neuron.hpp"

TEST_CASE("Excitatory conductance causes spiking") {
    double simulationTime = 200.0;
    double dt = 1.0;
    int spikeCount = 0;

    IzhikevichNeuron neuron(NeuronType::REGULAR_SPIKING);
    for (double t = 0; t < simulationTime; t += dt) {
        neuron.addExcConductance(10.0);
        neuron.step(dt);
        if (neuron.hasSpiked()) spikeCount++;
    }

    std::cout << "Exc spikes: " << spikeCount << std::endl;
    REQUIRE(spikeCount > 0);
}

TEST_CASE("Inhibitory conductance suppresses spiking") {
    double simulationTime = 200.0;
    double dt = 1.0;
    int spikeCountExc = 0;
    int spikeCountInh = 0;

    // Solo excitatorio
    IzhikevichNeuron n1(NeuronType::REGULAR_SPIKING);
    for (double t = 0; t < simulationTime; t += dt) {
        n1.addExcConductance(10.0);
        n1.step(dt);
        if (n1.hasSpiked()) spikeCountExc++;
    }

    // Excitatorio + inhibitorio fuerte
    IzhikevichNeuron n2(NeuronType::REGULAR_SPIKING);
    for (double t = 0; t < simulationTime; t += dt) {
        n2.addExcConductance(10.0);
        n2.addInhConductance(20.0);
        n2.step(dt);
        if (n2.hasSpiked()) spikeCountInh++;
    }

    std::cout << "Exc-only spikes: " << spikeCountExc
              << "  Exc+Inh spikes: " << spikeCountInh << std::endl;
    REQUIRE(spikeCountInh < spikeCountExc);
}

TEST_CASE("Neuron with i_offset fires without external input") {
    double simulationTime = 200.0;
    double dt = 1.0;
    int spikeCount = 0;

    IzhikevichNeuron neuron(NeuronType::REGULAR_SPIKING, /*i_offset=*/8.0);
    for (double t = 0; t < simulationTime; t += dt) {
        neuron.step(dt);
        if (neuron.hasSpiked()) spikeCount++;
    }

    std::cout << "i_offset spikes: " << spikeCount << std::endl;
    REQUIRE(spikeCount > 0);
}
