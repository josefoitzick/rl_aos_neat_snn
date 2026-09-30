#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>
#include "../../include/core/network.hpp"

TEST_CASE("Neural network creation") {
    double dt = 1.0;
    Network net(dt);

    int in1 = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
    int in2 = net.addInputNeuron(NeuronType::REGULAR_SPIKING);

    int h1 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h2 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);

    int out = net.addOutputNeuron(NeuronType::REGULAR_SPIKING);

    net.addSynapse(in1, h1, /*excitatory=*/true);
    net.addSynapse(in1, h2, /*excitatory=*/true);
    net.addSynapse(in2, h1, /*excitatory=*/true);
    net.addSynapse(in2, h2, /*excitatory=*/true);
    net.addSynapse(h1, out, /*excitatory=*/true);
    net.addSynapse(h2, out, /*excitatory=*/true);

    REQUIRE(net.getNeuronCount() == 5);
    REQUIRE(net.getSynapseCount() == 6);
    REQUIRE(net.getInputCount() == 2);
    REQUIRE(net.getOutputCount() == 1);
}

TEST_CASE("Synapse g_type is stored correctly") {
    Network net(1.0);
    int a = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
    int b = net.addOutputNeuron(NeuronType::REGULAR_SPIKING);

    int exc_id = net.addSynapse(a, b, /*excitatory=*/true);
    int inh_id = net.addSynapse(b, a, /*excitatory=*/false);

    REQUIRE(net.getSynapse(exc_id).isExcitatory() == true);
    REQUIRE(net.getSynapse(inh_id).isExcitatory() == false);
}

TEST_CASE("Excitatory synapse propagates spikes to output") {
    double dt = 1.0;
    double I = 10.0;
    double w = 20.0;
    double simulationTime = 100.0;
    int spikeCount = 0;

    Network net(dt);
    int input = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
    int output = net.addOutputNeuron(NeuronType::REGULAR_SPIKING);
    net.addSynapse(input, output, /*excitatory=*/true, w);

    for (double t = 0; t < simulationTime; t += dt) {
        net.setInputCurrent(input, I);
        net.step();
        if (net.getOutputSpikes()[0]) spikeCount++;
    }

    std::cout << "Exc synapse spikes: " << spikeCount << std::endl;
    REQUIRE(spikeCount > 0);
}

TEST_CASE("Inhibitory synapse reduces output spiking") {
    double dt = 1.0;
    double w = 20.0;
    double simulationTime = 100.0;

    // Red con sinapsis excitatoria
    int excSpikes = 0;
    {
        Network net(dt);
        int input = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
        int output = net.addOutputNeuron(NeuronType::REGULAR_SPIKING);
        net.addSynapse(input, output, /*excitatory=*/true, w);

        for (double t = 0; t < simulationTime; t += dt) {
            net.setInputCurrent(input, 10.0);
            net.step();
            if (net.getOutputSpikes()[0]) excSpikes++;
        }
    }

    // Red con sinapsis excitatoria + inhibitoria fuerte
    int inhSpikes = 0;
    {
        Network net(dt);
        int input = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
        int inh   = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
        int output = net.addOutputNeuron(NeuronType::REGULAR_SPIKING);
        net.addSynapse(input, output, /*excitatory=*/true, w);
        net.addSynapse(inh, output,   /*excitatory=*/false, w);

        for (double t = 0; t < simulationTime; t += dt) {
            net.setInputCurrent(input, 10.0);
            net.setInputCurrent(inh, 10.0);
            net.step();
            if (net.getOutputSpikes()[0]) inhSpikes++;
        }
    }

    std::cout << "Exc-only spikes: " << excSpikes
              << "  With inh spikes: " << inhSpikes << std::endl;
    REQUIRE(inhSpikes < excSpikes);
}