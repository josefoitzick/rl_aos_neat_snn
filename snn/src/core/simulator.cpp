// src/core/simulator.cpp
#include "../../include/core/simulator.hpp"
#include <iostream>
#include <chrono>
#include <limits>
#include <omp.h>

// ===== CONSTRUCTORES =====

Simulator::Simulator(const SimulationConfig& cfg)
    : config(cfg) {}

Simulator::Simulator(Network* net, const SimulationConfig& cfg)
    : config(cfg) {
    addNetwork(net, /*take_ownership=*/false);
}

Simulator::~Simulator() {
    for (size_t i = 0; i < networks.size(); i++) {
        if (owns_networks[i]) delete networks[i];
    }
}

// ===== POBLACIÓN =====

Network* Simulator::getNet(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(networks.size())) {
        std::cerr << "Simulator: índice de red inválido " << idx << std::endl;
        return nullptr;
    }
    return networks[idx];
}

int Simulator::addNetwork(Network* net, bool take_ownership) {
    int idx = networks.size();
    networks.push_back(net);
    owns_networks.push_back(take_ownership);
    return idx;
}

void Simulator::removeNetwork(int idx) {
    if (idx < 0 || idx >= static_cast<int>(networks.size())) return;
    if (owns_networks[idx]) delete networks[idx];
    networks.erase(networks.begin() + idx);
    owns_networks.erase(owns_networks.begin() + idx);
}

void Simulator::replaceNetwork(int idx, Network* net, bool take_ownership) {
    if (idx < 0 || idx >= static_cast<int>(networks.size())) return;
    if (owns_networks[idx]) delete networks[idx];
    networks[idx] = net;
    owns_networks[idx] = take_ownership;
}

int Simulator::cloneNetwork(int src_idx) {
    Network* src = getNet(src_idx);
    if (!src) return -1;
    Network* copy = src->clone();
    copy->fastReset();
    return addNetwork(copy, /*take_ownership=*/true);
}

Simulator* Simulator::clone() const {
    Simulator* copy = new Simulator(config);
    for (size_t i = 0; i < networks.size(); i++) {
        Network* net_copy = networks[i]->clone();
        net_copy->fastReset();
        copy->addNetwork(net_copy, /*take_ownership=*/true);
    }
    return copy;
}

Network* Simulator::getNetwork(int idx) {
    return getNet(idx);
}

const Network* Simulator::getNetwork(int idx) const {
    return getNet(idx);
}

// ===== CONFIGURACIÓN =====

void Simulator::setEncoder(std::unique_ptr<Encoder> enc) {
    encoder = std::move(enc);
}

void Simulator::setDecoder(std::unique_ptr<RLDecoder> dec) {
    decoder = std::move(dec);
}

void Simulator::setConfig(const SimulationConfig& cfg) {
    config = cfg;
}

// ===== SIMULACIÓN (lógica central) =====

SimulationResult Simulator::runSimulation(
    Network* net,
    const std::vector<double>& input_currents,
    double duration) {

    auto start = std::chrono::high_resolution_clock::now();

    SimulationResult result;
    net->fastReset();

    size_t num_outputs = net->getOutputCount();
    result.output_spike_times.resize(num_outputs);
    double total_spikes = 0;

    if (config.constant_input_current)
        net->setInputOffsets(input_currents);

    for (double t = 0; t < duration; t += config.dt) {
        if (!config.constant_input_current)
            net->setInputCurrents(input_currents);
        net->step();

        auto output_spikes = net->getOutputSpikes();
        for (size_t i = 0; i < output_spikes.size(); i++) {
            if (output_spikes[i]) {
                result.output_spike_times[i].push_back(t);
                total_spikes++;
            }
        }
    }

    if (config.constant_input_current)
        net->setInputOffsets(std::vector<double>(input_currents.size(), 0.0));

    auto voltages = net->getVoltages();
    for (int id : net->getOutputNeuronIds())
        result.output_voltages.push_back(voltages[id]);

    result.total_spikes = total_spikes;
    result.simulation_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();

    return result;
}

SimulationResult Simulator::runSimulationEncoded(
    Network* net,
    const std::vector<double>& input_values,
    double duration) {

    if (!encoder) {
        std::cerr << "Error: no hay encoder configurado" << std::endl;
        return SimulationResult();
    }

    auto encoded_inputs = encoder->encodeVector(input_values, config.encoding_duration, config.dt);

    if (encoded_inputs.size() != net->getInputCount()) {
        std::cerr << "Error: mismatch de entradas. Esperado "
                  << net->getInputCount() << ", recibido "
                  << encoded_inputs.size() << std::endl;
        return SimulationResult();
    }

    auto start = std::chrono::high_resolution_clock::now();

    SimulationResult result;
    net->fastReset();

    size_t num_outputs = net->getOutputCount();
    result.output_spike_times.resize(num_outputs);
    double total_spikes = 0;

    auto input_ids = net->getInputNeuronIds();

    for (double t = 0; t < duration; t += config.dt) {
        for (size_t i = 0; i < encoded_inputs.size(); i++) {
            for (double spike_time : encoded_inputs[i]) {
                if (std::abs(spike_time - t) < config.dt / 2.0)
                    net->setInputCurrent(input_ids[i], 100.0);
            }
        }

        net->step();

        auto output_spikes = net->getOutputSpikes();
        for (size_t i = 0; i < output_spikes.size(); i++) {
            if (output_spikes[i]) {
                result.output_spike_times[i].push_back(t);
                total_spikes++;
            }
        }
    }

    auto voltages = net->getVoltages();
    for (int id : net->getOutputNeuronIds())
        result.output_voltages.push_back(voltages[id]);

    result.total_spikes = total_spikes;
    result.simulation_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();

    return result;
}

// ===== SIMULACIÓN PÚBLICA =====

SimulationResult Simulator::simulate(
    int net_idx,
    const std::vector<double>& input_currents,
    double duration) {

    Network* net = getNet(net_idx);
    if (!net) return SimulationResult();
    if (duration <= 0) duration = config.evaluation_duration;

    auto result = runSimulation(net, input_currents, duration);
    if (config.verbose) printStats(result);
    return result;
}

SimulationResult Simulator::simulateEncoded(
    int net_idx,
    const std::vector<double>& input_values,
    double duration) {

    Network* net = getNet(net_idx);
    if (!net) return SimulationResult();
    if (duration <= 0) duration = config.evaluation_duration;

    auto result = runSimulationEncoded(net, input_values, duration);
    if (config.verbose) printStats(result);
    return result;
}

std::vector<SimulationResult> Simulator::simulateAll(
    const std::vector<double>& input_currents,
    double duration) {

    if (duration <= 0) duration = config.evaluation_duration;
    int n = static_cast<int>(networks.size());
    std::vector<SimulationResult> results(n);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
        results[i] = runSimulation(networks[i], input_currents, duration);

    return results;
}

std::vector<SimulationResult> Simulator::simulateAllEncoded(
    const std::vector<double>& input_values,
    double duration) {

    if (duration <= 0) duration = config.evaluation_duration;
    int n = static_cast<int>(networks.size());
    std::vector<SimulationResult> results(n);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
        results[i] = runSimulationEncoded(networks[i], input_values, duration);

    return results;
}

// ===== REINFORCEMENT LEARNING =====

double Simulator::runEpisode(
    int net_idx,
    const std::vector<double>& initial_state,
    int max_steps,
    std::function<bool(std::vector<double>&, int, double&)> step_callback) {

    Network* net = getNet(net_idx);
    if (!net || !encoder || !decoder) {
        std::cerr << "Error: red, encoder o decoder no configurados" << std::endl;
        return 0.0;
    }

    double total_reward = 0.0;
    std::vector<double> current_state = initial_state;

    for (int step = 0; step < max_steps; step++) {
        auto result = runSimulationEncoded(net, current_state, config.evaluation_duration);
        int action = decoder->decodeDiscreteAction(result.output_spike_times);

        double reward = 0.0;
        bool done = step_callback(current_state, action, reward);  // actualiza current_state
        total_reward += reward;

        if (done) break;
    }

    return total_reward;
}

double Simulator::evaluateFitness(
    int net_idx,
    const std::vector<std::vector<double>>& test_cases,
    std::function<double(const SimulationResult&)> fitness_func) {

    Network* net = getNet(net_idx);
    if (!net || test_cases.empty()) return 0.0;

    double total = 0.0;
    for (const auto& tc : test_cases)
        total += fitness_func(runSimulationEncoded(net, tc, config.evaluation_duration));
    return total / test_cases.size();
}

// ===== RESET =====

void Simulator::reset(int net_idx) {
    Network* net = getNet(net_idx);
    if (net) net->fastReset();
}

void Simulator::resetAll() {
    for (Network* net : networks) net->fastReset();
}

// ===== UTILIDADES =====

void Simulator::printStats(const SimulationResult& result) const {
    std::cout << "\n=== Simulation Statistics ===" << std::endl;
    std::cout << "Total output spikes: " << result.total_spikes << std::endl;
    std::cout << "Simulation time: " << result.simulation_time_ms << " ms" << std::endl;

    for (size_t i = 0; i < result.output_spike_times.size(); i++) {
        std::cout << "  Output " << i << ": " << result.output_spike_times[i].size() << " spikes";
        if (!result.output_spike_times[i].empty())
            std::cout << " (first at " << result.output_spike_times[i][0] << "ms)";
        std::cout << std::endl;
    }

    for (size_t i = 0; i < result.output_voltages.size(); i++)
        std::cout << "  Output " << i << ": " << result.output_voltages[i] << " mV" << std::endl;

    std::cout << "=========================" << std::endl;
}
