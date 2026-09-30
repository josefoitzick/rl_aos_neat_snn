// Tests de los cambios de M0: estado de reposo, pesos por sinapsis,
// entrada como corriente constante y determinismo del ruido.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>
#include "../../include/core/network.hpp"
#include "../../include/core/simulator.hpp"

using Catch::Matchers::WithinAbs;

// Los 8 modelos del paper Izhikevich (IZ-A..IZ-H)
static const std::vector<NeuronType> kPaperTypes = {
    NeuronType::TONIC_SPIKING, NeuronType::PHASIC_SPIKING, NeuronType::TONIC_BURSTING,
    NeuronType::PHASIC_BURSTING, NeuronType::MIXED_MODE, NeuronType::SPIKE_FREQUENCY_ADAPTATION,
    NeuronType::CLASS1_EXCITABLE, NeuronType::CLASS2_EXCITABLE};

static int countSpikes(IzhikevichNeuron& n, int steps) {
    int s = 0;
    for (int k = 0; k < steps; k++) { n.step(1.0); if (n.hasSpiked()) s++; }
    return s;
}

// ─── Estado de reposo ────────────────────────────────────────────────────────

TEST_CASE("Reposo: el potencial inicial es el punto fijo estable (b=0.2 → -70 mV)") {
    IzhikevichNeuron n(NeuronType::TONIC_SPIKING);
    REQUIRE_THAT(n.getVoltage(), WithinAbs(-70.0, 1e-9));
    REQUIRE_THAT(n.getRecovery(), WithinAbs(-14.0, 1e-9));
}

TEST_CASE("Reposo: los 8 tipos del paper no disparan ni derivan sin entrada") {
    for (NeuronType t : kPaperTypes) {
        IzhikevichNeuron n(t);
        double v0 = n.getVoltage();
        INFO("tipo " << static_cast<int>(t) << " v0=" << v0);
        REQUIRE(countSpikes(n, 500) == 0);
        REQUIRE_THAT(n.getVoltage(), WithinAbs(v0, 1e-6));
    }
}

TEST_CASE("Reposo: TONIC_BURSTING ya no dispara con una perturbación mínima") {
    // Antes (v=c=-50, punto fijo inestable) g=0.5 bastaba para un spike.
    IzhikevichNeuron n(NeuronType::TONIC_BURSTING);
    n.addExcConductance(0.5);
    REQUIRE(countSpikes(n, 50) == 0);
}

TEST_CASE("Reposo: reset() vuelve al reposo y conserva i_offset") {
    IzhikevichNeuron n(NeuronType::TONIC_SPIKING, /*i_offset=*/10.0);
    countSpikes(n, 100);
    n.reset();
    REQUIRE_THAT(n.getVoltage(), WithinAbs(-70.0, 1e-9));
    REQUIRE(n.getIOffset() == 10.0);
}

// ─── Pesos por sinapsis ──────────────────────────────────────────────────────

// input (corriente constante) → output con peso w; cuenta spikes de salida en 200 ms
static int outputSpikesWithWeight(double w) {
    Network net(1.0);
    int in  = net.addInputNeuron(NeuronType::TONIC_SPIKING);
    int out = net.addOutputNeuron(NeuronType::TONIC_SPIKING);
    net.addSynapse(in, out, true, w);
    net.setInputOffsets({10.0});
    int s = 0;
    for (int k = 0; k < 200; k++) { net.step(); if (net.getOutputSpikes()[0]) s++; }
    return s;
}

TEST_CASE("Pesos: peso 0 no transmite; más peso produce más spikes de salida") {
    REQUIRE(outputSpikesWithWeight(0.0) == 0);
    int low  = outputSpikesWithWeight(10.0);
    int high = outputSpikesWithWeight(40.0);
    INFO("w=10: " << low << " spikes, w=40: " << high << " spikes");
    REQUIRE(high > low);
}

TEST_CASE("Pesos: cada sinapsis guarda su propio peso") {
    Network net(1.0);
    int a = net.addInputNeuron(NeuronType::TONIC_SPIKING);
    int b = net.addOutputNeuron(NeuronType::TONIC_SPIKING);
    int c = net.addOutputNeuron(NeuronType::TONIC_SPIKING);
    int s1 = net.addSynapse(a, b, true, 3.5);
    int s2 = net.addSynapse(a, c, false, 7.0);
    REQUIRE(net.getSynapse(s1).getWeight() == 3.5);
    REQUIRE(net.getSynapse(s2).getWeight() == 7.0);
    net.setSynapseWeight(s1, 1.25);
    REQUIRE(net.getSynapse(s1).getWeight() == 1.25);
}

TEST_CASE("Pesos: insertNeuron asigna 1 a la sinapsis entrante y el peso original a la saliente (NEAT)") {
    Network net(1.0);
    int a = net.addInputNeuron(NeuronType::TONIC_SPIKING);
    int b = net.addOutputNeuron(NeuronType::TONIC_SPIKING);
    int s = net.addSynapse(a, b, true, 12.0);
    auto r = net.insertNeuron(s, NeuronType::TONIC_SPIKING);
    REQUIRE(r.neuron_id >= 0);
    REQUIRE_FALSE(net.getSynapse(s).isEnabled());
    REQUIRE(net.getSynapse(r.syn_in).getWeight() == 1.0);
    REQUIRE(net.getSynapse(r.syn_out).getWeight() == 12.0);
}

// Relevo in→h (TONIC_SPIKING): con peso 1 el nodo queda mudo; con ~30 retransmite 1:1.
// Justifica NeatConfig::add_node_in_weight (en NEAT de Stanley el peso entrante es 1).
TEST_CASE("Pesos: peso 1 no alcanza para relevar spikes; ~30 releva 1:1") {
    auto relay = [](double w) {
        Network net(1.0);
        int in = net.addInputNeuron(NeuronType::TONIC_SPIKING);
        int h  = net.addOutputNeuron(NeuronType::TONIC_SPIKING);
        net.addSynapse(in, h, true, w);
        net.setInputOffsets({10.0});
        int s_in = 0, s_h = 0;
        for (int k = 0; k < 200; k++) { net.step(); auto s = net.getSpikes(); s_in += s[in]; s_h += s[h]; }
        return std::make_pair(s_in, s_h);
    };
    REQUIRE(relay(1.0).second == 0);
    auto [s_in, s_h] = relay(30.0);
    REQUIRE(s_in > 0);
    REQUIRE(s_h == s_in);
}

// ─── Entrada como corriente constante ───────────────────────────────────────

TEST_CASE("Entrada: setInputOffsets equivale a una neurona con i_offset constante") {
    Network net(1.0);
    net.addInputNeuron(NeuronType::TONIC_SPIKING);
    net.setInputOffsets({10.0});
    int net_spikes = 0;
    for (int k = 0; k < 300; k++) { net.step(); if (net.getSpikes()[0]) net_spikes++; }

    IzhikevichNeuron ref(NeuronType::TONIC_SPIKING, /*i_offset=*/10.0);
    int ref_spikes = countSpikes(ref, 300);

    REQUIRE(net_spikes > 0);
    REQUIRE(net_spikes == ref_spikes);
}

TEST_CASE("Entrada: el Simulator en modo corriente constante no deja i_offset residual") {
    Network* net = new Network(1.0);
    int in  = net->addInputNeuron(NeuronType::TONIC_SPIKING);
    int out = net->addOutputNeuron(NeuronType::TONIC_SPIKING);
    net->addSynapse(in, out, true, 30.0);

    SimulationConfig cfg;
    cfg.constant_input_current = true;
    Simulator sim(cfg);
    sim.addNetwork(net, /*take_ownership=*/true);

    auto r1 = sim.simulate(0, {10.0}, 100.0);
    REQUIRE(r1.total_spikes > 0);
    REQUIRE(sim.getNetwork(0)->getNeuron(in).getIOffset() == 0.0);

    auto r0 = sim.simulate(0, {0.0}, 100.0);   // sin entrada: silencio
    REQUIRE(r0.total_spikes == 0);
}

// ─── Determinismo ────────────────────────────────────────────────────────────

TEST_CASE("Ruido: misma semilla produce la misma secuencia de spikes") {
    auto run = [](uint32_t seed) {
        IzhikevichNeuron::seedNoise(seed);
        IzhikevichNeuron n(NeuronType::TONIC_SPIKING, /*i_offset=*/3.0, /*noise=*/20.0);
        std::vector<bool> s;
        for (int k = 0; k < 300; k++) { n.step(1.0); s.push_back(n.hasSpiked()); }
        return s;
    };
    REQUIRE(run(123) == run(123));
    REQUIRE(run(123) != run(456));
}
