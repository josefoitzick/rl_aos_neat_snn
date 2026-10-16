#ifndef EVAL_EVALUATOR_HPP
#define EVAL_EVALUATOR_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "core/network.hpp"
#include "envs/control_env.hpp"
#include "neat/genome.hpp"

namespace eval {

// Interfaz SNN ↔ entorno (docs/diseno_neat_ap.md §8):
//  - Codificación SMALL (paper): 2 entradas por variable [neg, pos], corriente constante |x|·input_gain.
//    Con observación no negativa (imágenes) basta 1 entrada por variable: x·input_gain.
//  - Decodificación diferencial: 2 salidas por dimensión de acción [+, −];
//    a = clamp((n₊ − n₋) / spikes_full_action, −1, 1). Silencio → a = 0.
//  - Por paso de control se simula `step_ms` ms. Por defecto la red se reinicia antes de cada paso
//    (política reactiva). Con keep_state se reinicia solo al comienzo del episodio y el estado de
//    las neuronas (v, u, conductancias) pasa de un paso al siguiente (política con memoria).
//  - Multiagente: política compartida, una copia de la red por agente (cada una con su estado).
struct EvalConfig {
    double step_ms = 50.0;
    double input_gain = 40.0;          // |x| = 1 → ~9 spikes / 50 ms en TONIC_SPIKING (curva f-I medida)
    double spikes_full_action = 5.0;
    bool keep_state = false;
};

inline int numInputs(const envs::ControlEnv& env) { return (env.nonNegativeObs() ? 1 : 2) * env.obsDim(); }
inline int numOutputs(const envs::ControlEnv& env) { return 2 * env.actionDim(); }

// Actividad de la SNN durante episodios (estado del agente con rasgos de actividad, docs/experimentos/18_analisis_efectos_acciones.md).
// Solo observa: medirla no cambia ningún retorno. Se suman las espigas por neurona (índice de la red, sumado sobre agentes y
// episodios) y, por dimensión de acción y paso de control, si ambas salidas callaron (acción nula) o si la acción quedó en ±1.
struct ActividadSNN {
    std::vector<long> espigas;
    long pasos_snn = 0;             // pasos de simulación (por neurona), sumados sobre agentes y episodios
    long acciones = 0;              // dimensiones de acción × agentes × pasos de control
    long acciones_nulas = 0;
    long acciones_saturadas = 0;
    void sumar(const ActividadSNN& o);
};

// Fracciones para el estado: ocultas que nunca dispararon, ocultas saturadas (tasa ≥ umbral espigas por paso), entradas que
// nunca dispararon, acciones nulas por silencio de las salidas y acciones en el tope. Sin neuronas de un tipo, 0.
struct ResumenActividad {
    double ocultas_mudas = 0.0, ocultas_saturadas = 0.0, entradas_mudas = 0.0, salidas_mudas = 0.0, accion_saturada = 0.0;
};
ResumenActividad resumirActividad(const ActividadSNN& a, const Network& red, double umbral_saturacion = 0.2);

// Semillas de episodio deterministas. stream 0 = entrenamiento, 1 = test.
std::vector<uint64_t> episodeSeeds(uint64_t run_seed, int n, uint64_t stream);

// Retorno de un episodio. `nets` tiene una red por agente (copias del mismo genoma).
// `gains`: ganancia por neurona de entrada (operador E1); vacío = todas 1, sin multiplicar.
double runEpisode(std::vector<std::unique_ptr<Network>>& nets, envs::ControlEnv& env, uint64_t seed,
                  const EvalConfig& cfg, const std::vector<double>& gains = {}, ActividadSNN* actividad = nullptr);

// Retorno medio de un genoma en los episodios dados
double evaluateGenome(const neat::Genome& g, const neat::NeatConfig& ncfg, envs::ControlEnv& env,
                      const std::vector<uint64_t>& seeds, const EvalConfig& cfg);

// Retorno medio de un genoma con los episodios repartidos entre hilos (mismo resultado que evaluateGenome)
double evaluateGenomeParallel(const neat::Genome& g, const neat::NeatConfig& ncfg, const std::string& env_name,
                              const envs::EnvOptions& env_opt, const std::vector<uint64_t>& seeds, const EvalConfig& cfg);

// Un genoma y sus episodios, para medir varios en lote (validación, Φ y test final de las corridas de un pool)
struct GenomeJob {
    const neat::Genome* genome = nullptr;
    const neat::NeatConfig* ncfg = nullptr;
    std::string env_name;
    envs::EnvOptions env_opt;
    const std::vector<uint64_t>* seeds = nullptr;
    const EvalConfig* cfg = nullptr;
    ActividadSNN* actividad = nullptr;   // si está, se suma aquí la actividad de todos sus episodios
};

// Retorno medio de cada trabajo, con los episodios de todos repartidos en una sola región paralela.
// Cada valor es exactamente el de evaluateGenome con los mismos episodios.
std::vector<double> evaluateGenomes(const std::vector<GenomeJob>& jobs);

// Asigna `fitness` a toda la población, en paralelo (OpenMP; un entorno por hilo). Reparte episodios, no genomas:
// mejor balance de carga con poblaciones chicas. El resultado no depende del número de hilos.
void evaluatePopulation(std::vector<neat::Genome>& genomes, const neat::NeatConfig& ncfg,
                        const std::string& env_name, const envs::EnvOptions& env_opt,
                        const std::vector<uint64_t>& seeds, const EvalConfig& cfg);

// Una población de las que evalúa evaluatePopulations
struct PopulationJob {
    std::vector<neat::Genome>* genomes = nullptr;
    const neat::NeatConfig* ncfg = nullptr;
    std::string env_name;
    envs::EnvOptions env_opt;
    const std::vector<uint64_t>* seeds = nullptr;
    const EvalConfig* cfg = nullptr;
};

// Varias poblaciones (p. ej., las corridas en paralelo del entrenamiento de PPO) en una sola región paralela,
// con reparto dinámico sobre todos los episodios de todos los genomas y un entorno por hilo y por tipo de entorno.
// Cada genoma recibe exactamente el fitness que le daría evaluatePopulation (mismo cálculo, mismas semillas).
void evaluatePopulations(const std::vector<PopulationJob>& jobs);

}  // namespace eval

#endif
