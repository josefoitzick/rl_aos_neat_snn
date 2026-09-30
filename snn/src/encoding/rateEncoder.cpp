#include "../../include/encoding/rateEncoder.hpp"

RateEncoder::RateEncoder(double max_rate, uint32_t seed)
    : max_rate(max_rate), rng_(seed) {}

std::vector<double> RateEncoder::encode(double value, double duration, double dt) const {
    if (!isValid(value)) return {};

    std::vector<double> spike_times;
    double spike_prob = (value * max_rate / 1000.0) * dt;
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (double t = 0; t < duration; t += dt) {
        if (dis(rng_) < spike_prob)
            spike_times.push_back(t);
    }
    return spike_times;
}
