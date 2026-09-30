// include/encoding/rl_encoder.hpp
#ifndef RL_ENCODER_HPP
#define RL_ENCODER_HPP

#include "encoder.hpp"
#include <cstddef>
#include <utility>

// Encoder optimizado para Reinforcement Learning
// Determinista y rápido, diseñado para observaciones continuas
class RLEncoder : public Encoder {
public:
    enum class EncodingType {
        POPULATION, // Múltiples neuronas por feature con campos receptivos gaussianos
        RATE,       // Frecuencia constante (determinista)
        TEMPORAL,   // Timing-based: un spike por feature, tiempo ∝ (1 - value)
        SMALL,      // 2 neuronas por feature: corriente directa ∝ |value|×weight,
                    // neurona negativa activa si value<0, positiva si value≥0
        LARGE,      // N neuronas por feature (neurons_per_var): corriente directa a la
                    // neurona del intervalo activo → múltiples spikes por dinámica Izhikevich
        INTERVAL    // N neuronas por feature: spike trains a tasa max_rate (pre-computados)
    };

private:
    EncodingType type;
    double max_rate;
    size_t population_size;   // Para POPULATION coding
    size_t neurons_per_var;   // Para INTERVAL coding (neuronas por variable)

public:
    RLEncoder(EncodingType type = EncodingType::POPULATION,
              double max_rate = 100.0,
              size_t population_size = 5,
              size_t neurons_per_var = 20);

    // Encoding básico (para una sola feature, tipos RATE/TEMPORAL)
    std::vector<double> encode(double value, double duration, double dt) const override;

    // Codificar observación completa del ambiente (RATE/TEMPORAL/POPULATION)
    std::vector<std::vector<double>> encodeObservation(
        const std::vector<double>& observation,
        double duration,
        double dt) const;

    // Codificar con population coding (múltiples neuronas por feature con Gaussian RFs)
    std::vector<std::vector<double>> encodePopulation(
        double value,
        double duration,
        double dt) const;

    // --- SMALL ---
    // Traducción directa del encoding de ANNarchy-NEAT (bastianloyola).
    // 2 neuronas por feature: [negativa, positiva].
    // Devuelve la corriente a inyectar en cada neurona vía setInputCurrents():
    //   value < 0  →  {|value|×weight,        0}
    //   value ≥ 0  →  {0,              value×weight}
    // A diferencia de SIGN_MAGNITUDE (rate coding), aquí no se generan spike
    // trains: la corriente se pasa directamente a la dinámica del neurona.
    std::pair<double, double> encodeSmall(double value, double weight = 1.0) const;

    // Codifica observación completa con SMALL (2×N corrientes).
    // Resultado: [I_neg_0, I_pos_0, I_neg_1, I_pos_1, ...]
    // Usar con Network::setInputCurrents() en cada paso de simulación.
    std::vector<double> encodeObservationSmall(
        const std::vector<double>& observation,
        double weight = 1.0) const;

    // --- LARGE ---
    // Traducción de cartpole2/cartpole3 de ANNarchy-NEAT (bastianloyola).
    // neurons_per_var neuronas por feature; el rango [low, high] se divide en
    // neurons_per_var intervalos uniformes. Solo la neurona del intervalo activo
    // recibe corriente directa (current_val), el resto recibe 0.
    // La neurona dispara múltiples veces de forma natural (dinámica Izhikevich),
    // a diferencia de INTERVAL que pre-computa spike trains a tasa fija.
    // Usar resultado con Network::setInputCurrents() en cada paso de simulación.
    std::vector<double> encodeLarge(
        double value,
        double low, double high,
        double current_val = 20.0) const;

    // Codifica observación completa con LARGE (neurons_per_var × N corrientes).
    std::vector<double> encodeObservationLarge(
        const std::vector<double>& observation,
        const std::vector<std::pair<double, double>>& limits,
        double current_val = 20.0) const;

    // --- INTERVAL ---
    // N neuronas por feature con spike trains pre-computados a tasa max_rate.
    // Solo la neurona del intervalo activo dispara; los intervalos son uniformes en [low, high].
    std::vector<std::vector<double>> encodeInterval(
        double value,
        double duration,
        double dt,
        double low,
        double high) const;

    // Codifica observación completa con INTERVAL (neurons_per_var × N spike trains).
    std::vector<std::vector<double>> encodeObservationInterval(
        const std::vector<double>& observation,
        const std::vector<std::pair<double,double>>& limits,
        double duration,
        double dt) const;

    void setType(EncodingType t) { type = t; }
    void setPopulationSize(size_t size) { population_size = size; }
    void setNeuronsPerVar(size_t n) { neurons_per_var = n; }
    size_t getNeuronsPerVar() const { return neurons_per_var; }
};

#endif