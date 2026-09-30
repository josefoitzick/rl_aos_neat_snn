#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_cpu.h>
#include "../../include/core/network.hpp"
#include "../../include/core/simulator.hpp"
#include "../../include/decoding/decoder.hpp"
#include <cmath>
#include <iostream>
#include <vector>

namespace rlt = rl_tools;

using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using AcrobotSpec = rlt::rl::environments::acrobot::Specification<T, TI>;
using Env    = rlt::rl::environments::Acrobot<AcrobotSpec>;
using Params = Env::Parameters;
using RNG    = rlt::devices::random::CPU::ENGINE<>;

using ActionMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>>;
using ObsMatrix    = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 6, false>>;

// Observación (6 dims) → corrientes [0, 20] mA
//   obs[0] = cos(θ₁)      ∈ [-1, 1]
//   obs[1] = sin(θ₁)      ∈ [-1, 1]
//   obs[2] = cos(θ₂)      ∈ [-1, 1]
//   obs[3] = sin(θ₂)      ∈ [-1, 1]
//   obs[4] = θ₁_dot       ∈ [-4π, 4π]
//   obs[5] = θ₂_dot       ∈ [-9π, 9π]
static std::vector<double> obsToCurrents(const ObsMatrix& obs) {
    constexpr double MAX_VEL_1 = 4 * M_PI;
    constexpr double MAX_VEL_2 = 9 * M_PI;
    return {
        (rlt::get(obs, 0, 0) + 1.0) * 10.0,                          // cos(θ₁)
        (rlt::get(obs, 0, 1) + 1.0) * 10.0,                          // sin(θ₁)
        (rlt::get(obs, 0, 2) + 1.0) * 10.0,                          // cos(θ₂)
        (rlt::get(obs, 0, 3) + 1.0) * 10.0,                          // sin(θ₂)
        (rlt::get(obs, 0, 4) / MAX_VEL_1 + 1.0) * 10.0,             // θ₁_dot
        (rlt::get(obs, 0, 5) / MAX_VEL_2 + 1.0) * 10.0,             // θ₂_dot
    };
}

int main() {
    DEVICE device;
    RNG rng;
    rlt::init(device, rng, 42);

    Env env;
    Params params;
    rlt::initial_parameters(device, env, params);

    // SNN: 6 inputs → 4 hidden → 1 output
    Network net(1.0);

    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // cos(θ₁)
    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // sin(θ₁)
    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // cos(θ₂)
    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // sin(θ₂)
    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // θ₁_dot
    net.addInputNeuron(NeuronType::REGULAR_SPIKING);  // θ₂_dot

    int h0  = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h1  = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h2  = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h3  = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int out = net.addOutputNeuron(NeuronType::REGULAR_SPIKING);

    for (int in_id : net.getInputNeuronIds()) {
        net.addSynapse(in_id, h0);
        net.addSynapse(in_id, h1);
        net.addSynapse(in_id, h2);
        net.addSynapse(in_id, h3);
    }
    net.addSynapse(h0, out);
    net.addSynapse(h1, out);
    net.addSynapse(h2, out);
    net.addSynapse(h3, out);

    Simulator sim(&net);
    RLDecoder decoder(RLDecoder::DecodingType::RATE, 50.0);

    std::vector<double> test_weights = {5.0, 10.0, 20.0};
    rlt::rl::environments::acrobot::Observation<TI> obs_type;

    for (double weight : test_weights) {
        // Mismo peso en todas las sinapsis (barrido de escala)
        for (size_t s = 0; s < net.getSynapseCount(); s++)
            net.setSynapseWeight(static_cast<int>(s), weight);

        Env::State state;
        rlt::sample_initial_state(device, env, params, state, rng);

        double total_reward = 0.0;
        TI steps_taken = 0;

        for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; step++) {
            // Observar y mapear a corrientes
            ObsMatrix obs_mat;
            rlt::observe(device, env, params, state, obs_type, obs_mat, rng);
            auto currents = obsToCurrents(obs_mat);

            // Simular SNN 50 ms
            auto result = sim.simulate(0, currents, 50.0);

            // Decodificar tasa de spikes → acción ∈ [-1, 1]
            double action_value = 0.0;
            if (!result.output_spike_times.empty()) {
                double rate = decoder.decodeContinuousAction(result.output_spike_times[0]);
                action_value = rate * 2.0 - 1.0;
            }

            // Avanzar el entorno
            ActionMatrix action_mat;
            rlt::set(action_mat, 0, 0, action_value);

            Env::State next_state;
            rlt::step(device, env, params, state, action_mat, next_state, rng);
            T reward_val = rlt::reward(device, env, params, state, action_mat, next_state, rng);

            total_reward += reward_val;
            state = next_state;
            steps_taken++;

            // Terminación anticipada: extremo del segundo eslabón sobre el pivote
            if (rlt::terminated(device, env, params, state, rng)) {
                std::cout << "  [peso " << weight << "] Objetivo alcanzado en paso " << step + 1 << "\n";
                break;
            }
        }

        std::cout << "Peso: "          << weight
                  << " | Pasos: "      << steps_taken
                  << " | Recompensa: " << total_reward
                  << "\n";
    }

    return 0;
}
