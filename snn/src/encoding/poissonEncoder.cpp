#include "../../include/encoding/poissonEncoder.hpp"
#include <algorithm>

PoissonEncoder::PoissonEncoder(double max_rate, bool use_refractory, double refractory_period,
                               uint32_t seed)
    : max_rate(max_rate), use_refractory(use_refractory),
      refractory_period(refractory_period), rng_(seed) {}

std::vector<double> PoissonEncoder::encode(double value, double duration, double dt) const {
    value = std::clamp(value, 0.0, 1.0);
    std::vector<double> spike_times;
    double spike_prob = (value * max_rate / 1000.0) * dt;
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    double last_spike_time = -refractory_period - 1.0;

    for (double t = 0; t < duration; t += dt) {
        if (use_refractory && (t - last_spike_time) < refractory_period)
            continue;
        if (dis(rng_) < spike_prob) {
            spike_times.push_back(t);
            last_spike_time = t;
        }
    }
    return spike_times;
}

std::vector<double> PoissonEncoder::encodeTemporalPattern(
    const std::vector<double>& rate_pattern, double dt) const {
    std::vector<double> spike_times;
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    double last_spike_time = -refractory_period - 1.0;

    for (size_t i = 0; i < rate_pattern.size(); i++) {
        double t = i * dt;
        double firing_rate = std::clamp(rate_pattern[i], 0.0, 1.0) * max_rate;
        double spike_prob = (firing_rate / 1000.0) * dt;

        if (use_refractory && (t - last_spike_time) < refractory_period)
            continue;
        if (dis(rng_) < spike_prob) {
            spike_times.push_back(t);
            last_spike_time = t;
        }
    }
    return spike_times;
}
