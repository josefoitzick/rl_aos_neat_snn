#ifndef TTFS_ENCODER_HPP
#define TTFS_ENCODER_HPP

#include "encoder.hpp"

// Time-to-First-Spike encoder
// El valor se codifica como el tiempo del único spike dentro de la ventana:
//   valor alto  →  spike temprano
//   valor bajo  →  spike tardío (o ninguno si cae bajo el umbral)
//
// Fórmula lineal:  t_spike = (1 - value) * t_max
// Fórmula log:     t_spike = (1 - log(1 + value * (e-1))) * t_max
class TTFSEncoder : public Encoder {
public:
    enum class Mapping {
        LINEAR,       // t = (1 - v) * t_max
        LOGARITHMIC   // mayor resolución para valores bajos
    };

private:
    Mapping mapping;
    double threshold;  // Valores por debajo no generan spike [0, 1]
    double t_max_ratio; // Fracción de duration usada como t_max (default 1.0)

public:
    TTFSEncoder(Mapping mapping = Mapping::LINEAR,
                double threshold = 0.0,
                double t_max_ratio = 1.0);

    // Devuelve un vector con un único spike (o vacío si value < threshold)
    std::vector<double> encode(double value, double duration, double dt) const override;

    void setMapping(Mapping m) { mapping = m; }
    void setThreshold(double t) { threshold = t; }
    void setTMaxRatio(double r) { t_max_ratio = r; }
};

#endif