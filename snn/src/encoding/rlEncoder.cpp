#include "../../include/encoding/rlEncoder.hpp"
#include <cmath>
#include <algorithm>

RLEncoder::RLEncoder(EncodingType type, double max_rate, size_t population_size, size_t neurons_per_var)
    : type(type), max_rate(max_rate), population_size(population_size), neurons_per_var(neurons_per_var) {
}

std::vector<double> RLEncoder::encode(double value, double duration, double dt) const {
    (void)dt;
    if (!isValid(value)) {
        return {};
    }

    std::vector<double> spike_times;

    switch (type) {
        case EncodingType::RATE: {
            double firing_rate = value * max_rate;
            double inter_spike_interval = 1000.0 / firing_rate;
            for (double t = 0; t < duration; t += inter_spike_interval) {
                spike_times.push_back(t);
            }
            break;
        }
        case EncodingType::TEMPORAL: {
            double spike_time = (1.0 - value) * duration * 0.8;
            spike_times.push_back(spike_time);
            break;
        }
        case EncodingType::POPULATION:
        case EncodingType::SMALL:
        case EncodingType::LARGE:
        case EncodingType::INTERVAL:
            // Estos tipos usan métodos propios
            break;
    }

    return spike_times;
}

// Population coding: múltiples neuronas con receptive fields gaussianos solapados
std::vector<std::vector<double>> RLEncoder::encodePopulation(
    double value,
    double duration,
    double dt) const {

    (void)dt;
    if (!isValid(value)) {
        return std::vector<std::vector<double>>(population_size);
    }

    std::vector<std::vector<double>> population_spikes(population_size);

    for (size_t i = 0; i < population_size; i++) {
        double center = static_cast<double>(i) / (population_size - 1);
        double sigma = 0.3;
        double distance = std::abs(value - center);
        double activation = std::exp(-(distance * distance) / (2 * sigma * sigma));
        double firing_rate = activation * max_rate;

        if (firing_rate > 1.0) {
            double inter_spike_interval = 1000.0 / firing_rate;
            for (double t = 0; t < duration; t += inter_spike_interval) {
                population_spikes[i].push_back(t);
            }
        }
    }

    return population_spikes;
}

// Small encoding — traducción directa de ANNarchy-NEAT (bastianloyola).
// Devuelve {I_neg, I_pos}: corriente a inyectar en la neurona negativa y positiva.
//   value < 0  →  neurona negativa recibe |value|×weight, positiva recibe 0
//   value ≥ 0  →  neurona negativa recibe 0, positiva recibe value×weight
// Usar resultado con Network::setInputCurrents() en cada paso de simulación.
std::pair<double, double> RLEncoder::encodeSmall(double value, double weight) const {
    double magnitude = std::abs(value) * weight;
    if (value < 0)
        return {magnitude, 0.0};
    else
        return {0.0, magnitude};
}

// Codifica observación completa con SMALL.
// Resultado: [I_neg_0, I_pos_0, I_neg_1, I_pos_1, ...]  (2×N elementos)
std::vector<double> RLEncoder::encodeObservationSmall(
    const std::vector<double>& observation,
    double weight) const {

    std::vector<double> currents;
    currents.reserve(observation.size() * 2);
    for (double val : observation) {
        auto [neg, pos] = encodeSmall(val, weight);
        currents.push_back(neg);
        currents.push_back(pos);
    }
    return currents;
}

// Large encoding — traducción de cartpole2/cartpole3 de ANNarchy-NEAT (bastianloyola).
// neurons_per_var neuronas por feature. Solo la del intervalo activo recibe current_val;
// el resto recibe 0. La neurona dispara múltiples veces por dinámica Izhikevich.
// Usar resultado con Network::setInputCurrents() en cada paso de simulación.
std::vector<double> RLEncoder::encodeLarge(
        double value, double low, double high, double current_val) const {

    std::vector<double> currents(neurons_per_var, 0.0);
    if (neurons_per_var == 0 || high <= low) return currents;

    double step = (high - low) / static_cast<double>(neurons_per_var);
    double clamped = std::clamp(value, low, high);
    int idx = static_cast<int>((clamped - low) / step);
    if (idx >= static_cast<int>(neurons_per_var))
        idx = static_cast<int>(neurons_per_var) - 1;

    currents[idx] = current_val;
    return currents;
}

// Codifica observación completa con LARGE (neurons_per_var × N corrientes).
std::vector<double> RLEncoder::encodeObservationLarge(
        const std::vector<double>& observation,
        const std::vector<std::pair<double, double>>& limits,
        double current_val) const {

    std::vector<double> all_currents;
    all_currents.reserve(observation.size() * neurons_per_var);

    for (size_t i = 0; i < observation.size(); i++) {
        double low  = (i < limits.size()) ? limits[i].first  : 0.0;
        double high = (i < limits.size()) ? limits[i].second : 1.0;
        for (double c : encodeLarge(observation[i], low, high, current_val))
            all_currents.push_back(c);
    }
    return all_currents;
}

// Interval coding: neurons_per_var spike trains por feature
// Solo dispara la neurona cuyo intervalo uniforme contiene el valor
std::vector<std::vector<double>> RLEncoder::encodeInterval(
    double value,
    double duration,
    double dt,
    double low,
    double high) const {

    (void)dt;
    std::vector<std::vector<double>> spikes(neurons_per_var);

    if (neurons_per_var == 0 || high <= low) {
        return spikes;
    }

    // Límites de intervalo uniformemente distribuidos en [low, high]
    double step = (high - low) / static_cast<double>(neurons_per_var);

    // Clamp value al rango para asegurar que siempre cae en algún intervalo
    double clamped = std::max(low, std::min(value, high));

    int idx = static_cast<int>((clamped - low) / step);
    // La neurona del último intervalo incluye el valor máximo
    if (idx >= static_cast<int>(neurons_per_var)) {
        idx = static_cast<int>(neurons_per_var) - 1;
    }

    // La neurona activa dispara a max_rate de forma determinista
    double inter_spike_interval = 1000.0 / max_rate;
    for (double t = 0; t < duration; t += inter_spike_interval) {
        spikes[idx].push_back(t);
    }

    return spikes;
}

// Codifica observación completa con INTERVAL (neurons_per_var * N spike trains)
std::vector<std::vector<double>> RLEncoder::encodeObservationInterval(
    const std::vector<double>& observation,
    const std::vector<std::pair<double,double>>& limits,
    double duration,
    double dt) const {

    std::vector<std::vector<double>> all_spikes;
    all_spikes.reserve(observation.size() * neurons_per_var);

    for (size_t i = 0; i < observation.size(); i++) {
        double low  = (i < limits.size()) ? limits[i].first  : 0.0;
        double high = (i < limits.size()) ? limits[i].second : 1.0;
        auto var_spikes = encodeInterval(observation[i], duration, dt, low, high);
        for (auto& s : var_spikes) {
            all_spikes.push_back(std::move(s));
        }
    }

    return all_spikes;
}

// Codificar una observación completa (tipos RATE/TEMPORAL/POPULATION)
std::vector<std::vector<double>> RLEncoder::encodeObservation(
    const std::vector<double>& observation,
    double duration,
    double dt) const {

    std::vector<std::vector<double>> all_spikes;

    if (type == EncodingType::POPULATION) {
        for (double value : observation) {
            auto pop_spikes = encodePopulation(value, duration, dt);
            all_spikes.insert(all_spikes.end(), pop_spikes.begin(), pop_spikes.end());
        }
    } else {
        for (double value : observation) {
            all_spikes.push_back(encode(value, duration, dt));
        }
    }

    return all_spikes;
}
