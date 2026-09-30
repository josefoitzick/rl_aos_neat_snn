#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include "../../include/core/network.hpp"
#include "../../include/core/simulator.hpp"
#include "../../include/decoding/decoder.hpp"
#include <iostream>
#include <vector>

namespace rlt = rl_tools;

using T   = double;
using TI  = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using PendulumSpec = rlt::rl::environments::pendulum::Specification<T, TI>;
using Env  = rlt::rl::environments::Pendulum<PendulumSpec>;
using RNG  = rlt::devices::random::CPU::ENGINE<>;

// Matrices estáticas (stack) para acción y observación
using ActionMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>>;
using ObsMatrix    = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 3, false>>;

int main() {
    DEVICE device;
    RNG rng;
    rlt::init(device, rng, 42);

    Env env;
    Env::Parameters params;
    rlt::initial_parameters(device, env, params);

    // Construir SNN: 3 inputs (cos θ, sin θ, θ̇) → 2 hidden → 1 output
    Network net(1.0);
    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // cos(theta)
    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // sin(theta)
    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // theta_dot

    int h0 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h1 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int out = net.addOutputNeuron(NeuronType::REGULAR_SPIKING);

    auto inputs = net.getInputNeuronIds();
    for (int in_id : inputs) {
        net.addSynapse(in_id, h0);
        net.addSynapse(in_id, h1);
    }
    net.addSynapse(h0, out);
    net.addSynapse(h1, out);

    Simulator sim(&net);
    // Decodificador por tasa de disparo, ventana de 50 ms
    RLDecoder decoder(RLDecoder::DecodingType::RATE, 50.0);

    std::vector<double> test_weights = {5.0, 10.0, 20.0};

    for (double weight : test_weights) {
        // Mismo peso en todas las sinapsis (barrido de escala)
        for (size_t s = 0; s < net.getSynapseCount(); s++)
            net.setSynapseWeight(static_cast<int>(s), weight);

        Env::State state;
        rlt::sample_initial_state(device, env, params, state, rng);

        double total_reward = 0.0;

        for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; step++) {
            // Obtener observación: [cos θ, sin θ, θ̇]
            ObsMatrix obs_mat;
            rlt::rl::environments::pendulum::ObservationFourier<TI> obs_type;
            rlt::observe(device, env, params, state, obs_type, obs_mat, rng);

            // Mapear observación a corrientes de entrada [0, 20] mA
            std::vector<double> currents = {
                (rlt::get(obs_mat, 0, 0) + 1.0) * 10.0,          // cos(θ) ∈ [-1,1] → [0,20]
                (rlt::get(obs_mat, 0, 1) + 1.0) * 10.0,          // sin(θ) ∈ [-1,1] → [0,20]
                (rlt::get(obs_mat, 0, 2) / 8.0 + 1.0) * 10.0     // θ̇    ∈ [-8,8] → [0,20]
            };

            // Simular SNN durante 50 ms
            auto result = sim.simulate(0, currents, 50.0);

            // Decodificar: tasa de spikes → torque normalizado en [-1, 1]
            double action_value = 0.0;
            if (!result.output_spike_times.empty()) {
                // decodeContinuousAction devuelve [0, 1]; reescalar a [-1, 1]
                double rate_decoded = decoder.decodeContinuousAction(
                    result.output_spike_times[0]);
                action_value = rate_decoded * 2.0 - 1.0;
            }

            // Aplicar acción y avanzar el entorno
            ActionMatrix action_mat;
            rlt::set(action_mat, 0, 0, action_value);

            Env::State next_state;
            rlt::step(device, env, params, state, action_mat, next_state, rng);
            T reward = rlt::reward(device, env, params, state, action_mat, next_state, rng);

            total_reward += reward;
            state = next_state;
        }

        std::cout << "Peso: " << weight
                  << " | Recompensa total: " << total_reward
                  << " | Promedio: " << total_reward / static_cast<double>(Env::EPISODE_STEP_LIMIT)
                  << std::endl;
    }

    return 0;
}
