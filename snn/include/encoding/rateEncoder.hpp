#ifndef RATE_ENCODER_HPP
#define RATE_ENCODER_HPP

#include "encoder.hpp"
#include <cstdint>
#include <random>

class RateEncoder : public Encoder {
private:
    double max_rate;
    mutable std::mt19937 rng_;

public:
    // Semilla fija por defecto (reproducible). Ojo: el RNG es interno y no es
    // seguro compartir un mismo encoder entre hilos.
    explicit RateEncoder(double max_rate = 100.0, uint32_t seed = 0x5EEDu);

    std::vector<double> encode(double value, double duration, double dt) const override;

    void reseed(uint32_t seed) { rng_.seed(seed); }
    void setMaxRate(double rate) { max_rate = rate; }
    double getMaxRate() const { return max_rate; }
};

#endif
