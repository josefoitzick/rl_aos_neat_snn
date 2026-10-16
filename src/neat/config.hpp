#ifndef NEAT_CONFIG_HPP
#define NEAT_CONFIG_HPP

#include <vector>
#include "core/neuron.hpp"
#include "operators.hpp"
#include "variacion.hpp"

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

    // ── Catálogo ampliado (docs/operadores_candidatos.md). Con el catálogo por defecto nada de esto se usa ──
    std::vector<Operator> catalog = defaultCatalog();
    double c4                     = 0.4;    // peso de la diferencia media de parámetros de nodo en δ
    double fine_weight_sigma      = 1.0;    // A2: N(0, σ) sobre 1–3 conexiones (SharpNEAT: σ chico, 1–3 conexiones)
    bool   fine_weight_cauchy     = false;  // A2 con cola pesada: Cauchy de escala σ (Yao et al. 1999)
    double directed_iso_sigma     = 0.6;    // A7: ruido isotrópico (~1 % del ancho del rango inicial de pesos)
    double directed_line_sigma    = 0.2;    // A7: paso sobre la línea entre élites (Vassiliades y Mouret 2018)
    double bias_sigma             = 1.0;    // B1: N(0, σ) sobre la corriente de reposo
    double bias_max               = 10.0;   // B1: |sesgo| máximo (con ≈3 una TONIC_SPIKING ya dispara sola)
    double gain_log_sigma         = 0.3;    // E1: ganancia × exp(N(0, σ))
    double gain_min               = 0.25;
    double gain_max               = 4.0;
    int    add_node_multi_k       = 3;      // C9: nodos agregados por aplicación
    // F1–F4, dirigidos por actividad: el evaluador mide la tasa de disparo de cada neurona de cada genoma (solo observa)
    bool   medir_actividad        = false;  // lo activa el catálogo si contiene alguno de F1–F4 (ver activarActividad)
    // Umbrales en espigas por paso de simulación, calibrados con los cuantiles de la tasa por rol en pendulum-meta y
    // acrobot-swingup (máximo observado ~0,16; ver el diagnóstico [.diag_tasas] de tests/neat_run_tests.cpp)
    double umbral_muda            = 0.005;  // F1, F2, F4: muda o casi muda (≤ cuantil 5 % de salidas y entradas)
    double umbral_saturacion      = 0.08;   // F3: saturada (~cuantil 95 % de ocultas y salidas)
    double actividad_bias_sigma   = 2.0;    // F1–F3: |N(0, σ)| sobre la corriente de reposo
    double actividad_peso_sigma   = 0.5;    // F1–F3: entradas excitatorias × exp(±|N(0, σ)|)
    double actividad_gain_sigma   = 0.6;    // F4: ganancia × exp(|N(0, σ)|)
    // B3 y C4: familias corticales de Izhikevich (2004, Fig. 2)
    std::vector<NeuronType> families = {NeuronType::REGULAR_SPIKING, NeuronType::FAST_SPIKING,
                                        NeuronType::INTRINSICALLY_BURSTING, NeuronType::CHATTERING};
};

// Activa la medición de actividad si el catálogo tiene operadores dirigidos por actividad
inline void activarActividad(NeatConfig& c) {
    for (Operator op : c.catalog)
        if (usaActividad(op)) c.medir_actividad = true;
}

// Intensidades de variación de la configuración (struct Variacion en variacion.hpp)
inline Variacion variacionDe(const NeatConfig& c) {
    return {c.weight_perturb_power, c.weight_perturb_prob, c.mutation_only_fraction};
}

inline void aplicarVariacion(NeatConfig& c, const Variacion& v) {
    c.weight_perturb_power = v.weight_perturb_power;
    c.weight_perturb_prob = v.weight_perturb_prob;
    c.mutation_only_fraction = v.mutation_only_fraction;
}

// B2: rangos de (a, b, c, d) para la perturbación continua; cubren RS, FS, IB, CH, LTS, RZ y TONIC
// (Izhikevich 2004) sin llegar a regímenes que divergen con Euler (p. ej. BISTABILITY, a = 1, b = 1,5).
struct IzhikevichBounds {
    double lo[4] = {0.01, 0.15, -70.0, 0.5};
    double hi[4] = {0.12, 0.30, -45.0, 10.0};
    double sigma[4] = {0.01, 0.01, 2.0, 0.5};
};

}  // namespace neat

#endif
