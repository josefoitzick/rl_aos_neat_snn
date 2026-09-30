#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <vector>

// Clase base abstracta para todos los encoders
class Encoder {
public:
    virtual ~Encoder() = default;
    
    // Codificar un valor escalar en un patrón de spikes
    // value: valor a codificar [0, 1]
    // duration: duración de la codificación en ms
    // dt: paso de tiempo en ms
    // return: vector de tiempos de spike en ms
    virtual std::vector<double> encode(double value, double duration, double dt) const = 0;
    
    // Codificar un vector de valores
    virtual std::vector<std::vector<double>> encodeVector(
        const std::vector<double>& values, 
        double duration, 
        double dt) const;
    
protected:
    // Normalizar valor al rango [0, 1]
    double normalize(double value, double min_val, double max_val) const;
    
    // Validar que el valor esté en rango
    bool isValid(double value) const;
};

#endif