#include "../../include/core/neuron.hpp"

#include <algorithm>
#include <cmath>

// Definición de miembros estáticos thread-local
// Semilla fija por defecto (reproducibilidad); cambiar con IzhikevichNeuron::seedNoise().
thread_local std::mt19937 IzhikevichNeuron::rng{0x5EEDu};
thread_local std::uniform_real_distribution<double> IzhikevichNeuron::uniform{-0.5, 0.5};

IzhikevichNeuron::IzhikevichNeuron(NeuronType type, double i_offset, double noise,
                                   double tau_exc, double tau_inh)
        : a(0.0), b(0.0), c(0.0), d(0.0), v(0.0), u(0.0),
          i_offset(i_offset), noise(noise), spiked(false), enabled(true),
          g_exc(0.0), g_inh(0.0),
          tau_exc(tau_exc), tau_inh(tau_inh) {
    setNeuronType(type);
}

IzhikevichNeuron::IzhikevichNeuron(double a, double b, double c, double d,
                                   double i_offset, double noise,
                                   double tau_exc, double tau_inh)
        : a(a), b(b), c(c), d(d), v(0.0), u(0.0),
          i_offset(i_offset), noise(noise), spiked(false), enabled(true),
          g_exc(0.0), g_inh(0.0),
          tau_exc(tau_exc), tau_inh(tau_inh) {
    reset();
}

void IzhikevichNeuron::setNeuronType(NeuronType type) {
    switch(type) {
        case NeuronType::REGULAR_SPIKING:
            a = 0.02; b = 0.2; c = -65.0; d = 8.0; break;
        case NeuronType::FAST_SPIKING:
            a = 0.1;  b = 0.2; c = -65.0; d = 2.0; break;
        case NeuronType::INTRINSICALLY_BURSTING:
            a = 0.02; b = 0.2; c = -55.0; d = 4.0; break;
        case NeuronType::CHATTERING:
            a = 0.02; b = 0.2; c = -50.0; d = 2.0; break;
        case NeuronType::LOW_THRESHOLD_SPIKING:
            a = 0.02; b = 0.25; c = -65.0; d = 2.0; break;
        case NeuronType::RESONATOR:
            a = 0.1;  b = 0.26; c = -65.0; d = 2.0; break;

        // Izhikevich (2004), Fig. 1 — 20 patrones de disparo.
        case NeuronType::TONIC_SPIKING:
            a = 0.02; b = 0.2;  c = -65.0; d = 6.0;   break;
        case NeuronType::PHASIC_SPIKING:
            a = 0.02; b = 0.25; c = -65.0; d = 6.0;   break;
        case NeuronType::TONIC_BURSTING:
            a = 0.02; b = 0.2;  c = -50.0; d = 2.0;   break;
        case NeuronType::PHASIC_BURSTING:
            a = 0.02; b = 0.25; c = -55.0; d = 0.05;  break;
        case NeuronType::MIXED_MODE:
            a = 0.02; b = 0.2;  c = -55.0; d = 4.0;   break;
        case NeuronType::SPIKE_FREQUENCY_ADAPTATION:
            a = 0.01; b = 0.2;  c = -65.0; d = 8.0;   break;
        case NeuronType::CLASS1_EXCITABLE:
            a = 0.02; b = -0.1; c = -55.0; d = 6.0;   break;
        case NeuronType::CLASS2_EXCITABLE:
            a = 0.2;  b = 0.26; c = -65.0; d = 0.0;   break;
        case NeuronType::SPIKE_LATENCY:
            a = 0.02; b = 0.2;  c = -65.0; d = 6.0;   break;
        case NeuronType::SUBTHRESHOLD_OSCILLATIONS:
            a = 0.05; b = 0.26; c = -60.0; d = 0.0;   break;
        case NeuronType::INTEGRATOR:
            a = 0.02; b = -0.1; c = -55.0; d = 6.0;   break;
        case NeuronType::REBOUND_SPIKE:
            a = 0.03; b = 0.25; c = -60.0; d = 4.0;   break;
        case NeuronType::REBOUND_BURST:
            a = 0.03; b = 0.25; c = -52.0; d = 0.0;   break;
        case NeuronType::THRESHOLD_VARIABILITY:
            a = 0.03; b = 0.25; c = -60.0; d = 4.0;   break;
        case NeuronType::BISTABILITY:
            a = 1.0;  b = 1.5;  c = -60.0; d = 0.0;   break;
        case NeuronType::DEPOLARIZING_AFTERPOTENTIAL:
            a = 1.0;  b = 0.2;  c = -60.0; d = -21.0; break;
        case NeuronType::ACCOMMODATION:
            a = 0.02; b = 1.0;  c = -55.0; d = 4.0;   break;
        case NeuronType::INHIBITION_INDUCED_SPIKING:
            a = -0.02;  b = -1.0; c = -60.0; d = 8.0;  break;
        case NeuronType::INHIBITION_INDUCED_BURSTING:
            a = -0.026; b = -1.0; c = -45.0; d = -2.0; break;

        default:
            a = 0.02; b = 0.2; c = -65.0; d = 8.0; break;
    }
    reset();
}

/**
 *  Método de Euler (dos sub-pasos de dt/2 por precisión, como Izhikevich recomienda)
 *
 *  Corriente total (como ANNarchy):
 *    I_total = g_exc - g_inh + I_syn + i_offset + noise * Uniform(-0.5, 0.5)
 *
 *  Decaimiento de conductancias (Euler de tau*dg/dt = -g):
 *    g_exc -= g_exc / tau_exc * dt
 *    g_inh -= g_inh / tau_inh * dt
 */
void IzhikevichNeuron::step(double dt) {
    spiked = false;


    // Decaimiento exponencial de conductancias (tau * dg/dt = -g)
    g_exc *= exp( -dt / tau_exc);
    g_inh *= exp( -dt / tau_inh);

    // Integración de v y u (dos sub-pasos de dt/2)
    for (int i = 0; i < 2; i++) {
        // Corriente total: excitatoria - inhibitoria + directa + offset + ruido
        double I_total = g_exc - g_inh + i_offset + noise * uniform(rng);

        double dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I_total) * (dt / 2.0);
        v += dv;
        // Red de seguridad, no el límite de disparo: ver comentario junto a
        // V_SAFETY_MAX/MIN en neuron.hpp. No usar V_THRESHOLD acá.
        v = std::clamp(v, V_SAFETY_MIN, V_SAFETY_MAX);

        double du = a * (b * v - u) * (dt / 2.0);
        u += du;
        u = std::clamp(u, -U_SAFETY_LIMIT, U_SAFETY_LIMIT);
    }

    if (v >= V_THRESHOLD) {
        spiked = true;
        v = c;
        u += d;
    }

}

/**
 *  Acumula conductancia excitatoria (AMPA) — llamar antes de step()
 *  cuando dispara una neurona presináptica excitatoria
 */
void IzhikevichNeuron::addExcConductance(double g) {
    g_exc += g;
}

/**
 *  Acumula conductancia inhibitoria (INH) — llamar antes de step()
 *  cuando dispara una neurona presináptica inhibitoria
 */
void IzhikevichNeuron::addInhConductance(double g) {
    g_inh += g;
}

/**
 *  Estado de reposo: punto fijo ESTABLE del modelo con I = 0.
 *  Con u = b·v, v' = 0 da 0.04v² + (5-b)v + 140 = 0; el reposo es la raíz menor
 *  (la mayor es el umbral, un punto fijo inestable). Existe si (5-b)² >= 22.4,
 *  es decir b <= ~0.267 (p. ej. b=0.2 → v=-70; b=-0.1 → v=-87.5).
 *  Si no existe (tipos como BISTABILITY o ACCOMMODATION), se usa v=-65, u=b·v.
 *
 *  Antes se iniciaba en v=c, u=b·c: tipos con c=-50 (TONIC_BURSTING, CHATTERING)
 *  quedaban exactamente en el punto fijo inestable y disparaban con cualquier
 *  perturbación mínima (ver docs/diseno_neat_ap.md §3).
 */
void IzhikevichNeuron::setRestState() {
    const double k = 5.0 - b;
    const double disc = k * k - 4.0 * 0.04 * 140.0;
    v = (disc >= 0.0) ? (-k - std::sqrt(disc)) / (2.0 * 0.04) : -65.0;
    u = b * v;
}

/**
 *  Restaura el estado inicial (reposo) de la neurona. No cambia i_offset.
 */
void IzhikevichNeuron::reset() {
    setRestState();
    g_exc = 0.0;
    g_inh = 0.0;
    spiked = false;
}

void IzhikevichNeuron::seedNoise(uint32_t seed) {
    rng.seed(seed);
}
