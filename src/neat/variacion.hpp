#ifndef NEAT_VARIACION_HPP
#define NEAT_VARIACION_HPP

namespace neat {

// Intensidades de variación que un AOS puede mover entre generaciones (plan B, docs/experimentos/15_margen_intensidades.md).
// Son campos de NeatConfig (variacionDe / aplicarVariacion en config.hpp) y se copian tal cual: aplicar las de la
// configuración no cambia nada. Sin dependencias, para que el mapa de acciones no arrastre el simulador.
struct Variacion {
    double weight_perturb_power = 5.0;     // potencia de la perturbación de pesos
    double weight_perturb_prob = 0.9;      // 1 − probabilidad de reasignar un peso
    double mutation_only_fraction = 0.25;  // fracción de hijos sin cruce
};

}  // namespace neat

#endif
