#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "network.hpp"
#include "../encoding/encoder.hpp"
#include "../decoding/decoder.hpp"
#include <cstddef>
#include <vector>
#include <memory>
#include <functional>

struct SimulationConfig {
    double dt;
    double encoding_duration;
    double evaluation_duration;
    // false: las corrientes de entrada se SUMAN a g_exc en cada paso (setInputCurrents).
    // true:  se aplican como corriente constante i_offset (setInputOffsets), como en ANNarchy.
    bool constant_input_current;
    bool verbose;

    SimulationConfig()
        : dt(1.0),
          encoding_duration(50.0),
          evaluation_duration(100.0),
          constant_input_current(false),
          verbose(false) {}
};

struct SimulationResult {
    std::vector<std::vector<double>> output_spike_times;
    std::vector<std::vector<double>> all_spike_times;
    std::vector<double> output_voltages;
    double total_spikes;
    double simulation_time_ms;

    SimulationResult() : total_spikes(0), simulation_time_ms(0) {}
};

// Contenedor de una población de redes con simulación en paralelo (OpenMP).
// Cada red usa los pesos de sus propias sinapsis.
class Simulator {
private:
    std::vector<Network*> networks;
    std::vector<bool> owns_networks;
    std::unique_ptr<Encoder> encoder;
    std::unique_ptr<RLDecoder> decoder;
    SimulationConfig config;

    Network* getNet(int idx) const;
    SimulationResult runSimulation(Network* net, const std::vector<double>& input_currents, double duration);
    SimulationResult runSimulationEncoded(Network* net, const std::vector<double>& input_values, double duration);

public:
    explicit Simulator(const SimulationConfig& cfg = SimulationConfig());
    explicit Simulator(Network* net, const SimulationConfig& cfg = SimulationConfig());
    ~Simulator();

    // ===== POBLACIÓN =====

    int addNetwork(Network* net, bool take_ownership = false);
    void removeNetwork(int idx);
    void replaceNetwork(int idx, Network* net, bool take_ownership = false);
    int cloneNetwork(int src_idx);
    Simulator* clone() const;

    Network* getNetwork(int idx = 0);
    const Network* getNetwork(int idx = 0) const;
    size_t getNetworkCount() const { return networks.size(); }

    // ===== CONFIGURACIÓN =====

    void setEncoder(std::unique_ptr<Encoder> enc);
    void setDecoder(std::unique_ptr<RLDecoder> dec);
    void setConfig(const SimulationConfig& cfg);
    const SimulationConfig& getConfig() const { return config; }

    // ===== SIMULACIÓN =====

    SimulationResult simulate(int net_idx, const std::vector<double>& input_currents, double duration = 0.0);
    SimulationResult simulateEncoded(int net_idx, const std::vector<double>& input_values, double duration = 0.0);

    std::vector<SimulationResult> simulateAll(const std::vector<double>& input_currents, double duration = 0.0);
    // Ojo: todos los hilos comparten el encoder; con encoders estocásticos
    // (Poisson, Rate) eso es una condición de carrera. Usar encoders deterministas.
    std::vector<SimulationResult> simulateAllEncoded(const std::vector<double>& input_values, double duration = 0.0);

    // ===== REINFORCEMENT LEARNING =====

    // step_callback(state, action, reward) recibe el estado por referencia: debe
    // actualizarlo con la nueva observación del entorno y devolver true si terminó.
    double runEpisode(int net_idx, const std::vector<double>& initial_state, int max_steps,
                      std::function<bool(std::vector<double>&, int, double&)> step_callback);

    double evaluateFitness(int net_idx, const std::vector<std::vector<double>>& test_cases,
                           std::function<double(const SimulationResult&)> fitness_func);

    // ===== RESET =====

    void reset(int net_idx);
    void resetAll();

    // ===== UTILIDADES =====

    void printStats(const SimulationResult& result) const;
};

#endif
