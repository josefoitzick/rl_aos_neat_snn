#include "../../include/encoding/ttfsEncoder.hpp"
#include <cmath>

TTFSEncoder::TTFSEncoder(Mapping mapping, double threshold, double t_max_ratio)
    : mapping(mapping), threshold(threshold), t_max_ratio(t_max_ratio) {}

std::vector<double> TTFSEncoder::encode(double value, double duration, double dt) const {
    if (!isValid(value) || value < threshold) {
        return {};
    }

    double t_max = (duration - dt) * t_max_ratio;
    double t_spike = 0.0;

    switch (mapping) {
        case Mapping::LINEAR:
            // Valor 1.0 → t=0, valor 0.0 → t=t_max
            t_spike = (1.0 - value) * t_max;
            break;

        case Mapping::LOGARITHMIC:
            // Comprime el rango: mayor resolución para valores bajos
            // log(1 + v*(e-1)) mapea [0,1] → [0,1] con curvatura logarítmica
            t_spike = (1.0 - std::log1p(value * (M_E - 1.0))) * t_max;
            break;
    }

    // Alinear al paso de tiempo más cercano
    t_spike = std::round(t_spike / dt) * dt;
    t_spike = std::max(0.0, std::min(t_spike, duration - dt));

    return {t_spike};
}