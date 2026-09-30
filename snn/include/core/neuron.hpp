#ifndef NEURON_HPP
#define NEURON_HPP

#include <cstdint>
#include <random>

// Los primeros 6 son las clases corticales/talámicas clásicas de Izhikevich
// (Fig. 2 de Izhikevich, "Which Model to Use for Cortical Spiking Neurons?",
// IEEE Trans. Neural Networks 15(5), 2004). Los siguientes 19 cubren los 20
// patrones de disparo de la Fig. 1 del mismo trabajo (RESONATOR ya cubre el
// patrón "Resonator" de esa figura, con (a,b,c,d) casi idénticos, así que no
// se duplica).
//
// Nota: algunos pares comparten (a,b,c,d) exactos en la tabla original de
// Izhikevich — el comportamiento distinto que da nombre a cada uno depende
// del protocolo de corriente de entrada (pulso breve, rampa, pulso negativo
// seguido de liberación), no solo de los parámetros del modelo. Con la
// corriente que efectivamente reciban de la red evolucionada pueden
// comportarse de forma idéntica a su par:
//   SPIKE_LATENCY == TONIC_SPIKING (mismos a,b,c,d)
//   INTEGRATOR == CLASS1_EXCITABLE (mismos a,b,c,d)
//   THRESHOLD_VARIABILITY == REBOUND_SPIKE (mismos a,b,c,d)
enum class NeuronType {
    REGULAR_SPIKING,
    FAST_SPIKING,
    INTRINSICALLY_BURSTING,
    CHATTERING,
    LOW_THRESHOLD_SPIKING,
    RESONATOR,
    TONIC_SPIKING,
    PHASIC_SPIKING,
    TONIC_BURSTING,
    PHASIC_BURSTING,
    MIXED_MODE,
    SPIKE_FREQUENCY_ADAPTATION,
    CLASS1_EXCITABLE,
    CLASS2_EXCITABLE,
    SPIKE_LATENCY,
    SUBTHRESHOLD_OSCILLATIONS,
    INTEGRATOR,
    REBOUND_SPIKE,
    REBOUND_BURST,
    THRESHOLD_VARIABILITY,
    BISTABILITY,
    DEPOLARIZING_AFTERPOTENTIAL,
    ACCOMMODATION,
    INHIBITION_INDUCED_SPIKING,
    INHIBITION_INDUCED_BURSTING
};

class IzhikevichNeuron {
private:
    // Parámetros de las ecuaciones diferenciales
    double a; // Velocidad de recuperación
    double b; // Sensibilidad a v
    double c; // Reset del voltaje
    double d; // Aumento de u tras la espiga
    double v; // Potencial de membrana (voltaje)
    double u; // Recuperación
    double i_offset; // Corriente de offset constante (mA)
    double noise;    // Amplitud de ruido uniforme [-0.5, 0.5] * noise
    bool spiked;     // Indica si la neurona se dispara o no
    bool enabled;    // false = deshabilitada (no participa en la simulación)
    double g_exc;   // Conductancia excitatoria (AMPA)
    double g_inh;   // Conductancia inhibitoria (GABA)
    double tau_exc; // Constante de tiempo AMPA (ms)
    double tau_inh; // Constante de tiempo GABA (ms)

    static constexpr double V_THRESHOLD = 30.0;
    // Red de seguridad numérica (NO el límite de disparo — eso sigue siendo
    // V_THRESHOLD): algunas combinaciones (a,b,c,d) — p.ej. BISTABILITY
    // (a=1, b=1.5) — pueden hacer que el término 0.04*v*v del modelo de
    // Izhikevich diverja a Inf/NaN bajo Euler con dt=1ms. v/u se clampean a
    // estos rangos tras cada sub-paso SOLO para cortar esa divergencia.
    // Deliberadamente muy por fuera de cualquier trayectoria normal — un
    // spike sano puede pasarse de V_THRESHOLD por unas pocas decenas dentro
    // de un sub-paso (verificado con los 6 tipos originales: conteo de
    // spikes y v idénticos con y sin este clamp) y estas cotas no lo tocan.
    // No usar V_THRESHOLD como cota superior: eso SÍ altera la dinámica de
    // neuronas bien portadas cerca del umbral (probado empíricamente).
    static constexpr double V_SAFETY_MAX = 1.0e4;
    static constexpr double V_SAFETY_MIN = -1.0e4;
    static constexpr double U_SAFETY_LIMIT = 1.0e6;

    // Generador de ruido (thread-local para evitar condiciones de carrera)
    static thread_local std::mt19937 rng;
    static thread_local std::uniform_real_distribution<double> uniform;

    void setRestState();  // v, u ← punto fijo estable con I = 0 (ver neuron.cpp)

public:
    IzhikevichNeuron(NeuronType type, double i_offset = 0.0, double noise = 0.0,
                     double tau_exc = 5.0, double tau_inh = 10.0);
    IzhikevichNeuron(double a, double b, double c, double d,
                     double i_offset = 0.0, double noise = 0.0,
                     double tau_exc = 5.0, double tau_inh = 10.0);

    void step(double dt);
    void addExcConductance(double g);  // Conductancia excitatoria: g_exc += g
    void addInhConductance(double g);  // Conductancia inhibitoria: g_inh += g
    void reset();  // estado de reposo, conductancias en 0

    // Siembra el generador de ruido del hilo actual (thread_local)
    static void seedNoise(uint32_t seed);

    bool hasSpiked() const { return spiked; }
    bool isEnabled() const { return enabled; }
    void enable()          { enabled = true; }
    void disable()         { enabled = false; }
    double getVoltage() const { return v; }
    double getRecovery() const { return u; }
    double getGExc() const { return g_exc; }
    double getGInh() const { return g_inh; }

    void setNeuronType(NeuronType type);

    void setVoltage(double voltage) { v = voltage; }
    void setRecovery(double recovery) { u = recovery; }
    void setIOffset(double offset) { i_offset = offset; }
    void setNoise(double n) { noise = n; }
    void setTauExc(double tau) { tau_exc = tau; }
    void setTauInh(double tau) { tau_inh = tau; }

    double getIOffset() const { return i_offset; }
    double getNoise() const { return noise; }
};

#endif
