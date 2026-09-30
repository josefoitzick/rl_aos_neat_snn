#include "../../include/encoding/encoder.hpp"
#include <algorithm>
#include <stdexcept>

std::vector<std::vector<double>> Encoder::encodeVector(
        const std::vector<double>& values,
        double duration,
        double dt) const {
    std::vector<std::vector<double>> result;
    result.reserve(values.size());
    for (double v : values)
        result.push_back(encode(v, duration, dt));
    return result;
}

double Encoder::normalize(double value, double min_val, double max_val) const {
    if (max_val <= min_val)
        throw std::invalid_argument("normalize: max_val must be > min_val");
    return std::clamp((value - min_val) / (max_val - min_val), 0.0, 1.0);
}

// Valor válido para codificación de tasa: debe estar en [0, 1].
// Un valor negativo produciría una probabilidad de spike negativa, lo que
// hace que la condición dis(gen) < spike_prob nunca se cumpla → 0 spikes,
// comportamiento incorrecto y silencioso.
bool Encoder::isValid(double value) const {
    return value >= 0.0 && value <= 1.0;
}
