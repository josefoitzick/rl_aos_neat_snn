#ifndef NEAT_CONFIG_HPP
#define NEAT_CONFIG_HPP

#include "core/neuron.hpp"

namespace neat {

// Parámetros de NEAT. Los valores por defecto son los de Stanley & Miikkulainen (2002), §4.1,
// salvo los marcados [SNN], que no existen en el paper y se calibran para redes de espigas.
struct NeatConfig {
    // ── Población y especiación (Stanley 2002 §3.3 y §4.1) ──
    int    population_size        = 150;
    double c1                     = 1.0;    // peso de genes en exceso
    double c2                     = 1.0;    // peso de genes disjuntos
    double c3                     = 0.4;    // peso de la diferencia media de pesos
    double compat_threshold       = 3.0;    // δt
    // [SNN] W̄ se divide por esta escala antes de multiplicar por c3. Stanley usa pesos de
    // orden 1 ("una diferencia media de 3,0 equivale a un gen"); aquí los pesos son
    // conductancias de orden 10–50. Sin reescalar, casi cada genoma forma su propia especie.
    double compat_weight_scale    = 10.0;
    int    small_genome_genes     = 20;     // si ambos genomas tienen menos genes, N = 1
    // [Calibración] Umbral dinámico: si target_species > 0, δt se ajusta cada generación en
    // ±compat_threshold_step para acercar el número de especies al objetivo (no está en el
    // paper de 2002, que usa δt fijo; es práctica común en implementaciones posteriores).
    int    target_species         = 0;
    double compat_threshold_step  = 0.1;   // con 0,3 el conteo de especies oscila (6–19 para objetivo 8)
    double compat_threshold_min   = 0.3;
    int    species_stagnation     = 15;     // generaciones sin mejora → la especie no se reproduce
    int    global_stagnation      = 20;     // sin mejora global → solo las 2 mejores especies (nota 2 del paper)
    int    elitism_min_size       = 6;      // campeón copiado si la especie tiene MÁS de 5 miembros
    double survival_fraction      = 0.2;    // fracción de cada especie que puede reproducirse (no está en el paper)

    // ── Reproducción ──
    double mutation_only_fraction = 0.25;   // 25 % de los hijos sin cruce
    double interspecies_rate      = 0.001;
    double disable_inherit_prob   = 0.75;   // gen deshabilitado en algún padre → deshabilitado con p=0,75

    // ── Operador "mutar pesos" ──
    double weight_perturb_prob    = 0.9;    // por peso: 90 % perturbación uniforme, 10 % valor nuevo
    double weight_perturb_power   = 5.0;    // [SNN] perturbación ~ U(-power, power)
    double weight_init_range      = 30.0;   // [SNN] peso inicial / nuevo ~ U(-range, range)
    double weight_max             = 60.0;   // [SNN] |w| <= weight_max

    // ── Operador "agregar nodo" ──
    // Stanley: entrante con peso 1 y saliente con el peso original, para alterar poco la red.
    // [SNN] Con conductancias, peso 1 deja mudo al nodo nuevo (cortaría la ruta); ~30 hace que
    // una neurona TONIC_SPIKING retransmita 1:1 (test "peso 1 no alcanza…" en snn/tests).
    double add_node_in_weight     = 30.0;

    // ── Red ──
    NeuronType neuron_type        = NeuronType::TONIC_SPIKING;  // IZ-A del paper
    double     dt                 = 1.0;                        // ms
};

}  // namespace neat

#endif
