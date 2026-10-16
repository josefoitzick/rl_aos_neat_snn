#ifndef EXPERIMENT_OPCIONES_NEAT_HPP
#define EXPERIMENT_OPCIONES_NEAT_HPP

// Opciones de NEAT, del entorno y de la interfaz SNN con los mismos nombres, valores por defecto y semántica que
// neat_control, para que las herramientas del agente PPO (aos_neat_check y el entrenador) acepten las mismas
// líneas de comando que las corridas de NEAT (p. ej., las de las configuraciones ajustadas con Optuna).
// La equivalencia con neat_control se verifica bit a bit (scripts/verificar_neat_run.sh).
#include <CLI/CLI.hpp>
#include <string>
#include "eval/evaluator.hpp"
#include "neat/config.hpp"
#include "neat/operators.hpp"

namespace opciones_neat {

struct Opciones {
    std::string env_name = "pendulum";
    envs::EnvOptions env_opt;
    eval::EvalConfig ecfg;
    neat::NeatConfig ncfg;
    uint64_t seed = 1;
    int generations = 100, episodes = 5, test_episodes = 10, final_episodes = 100;
    bool resample_train = false;
    std::string catalog_txt;
    double weight_replace_prob = 0.0;   // se fija en agregar(), como en neat_control
};

inline void agregar(CLI::App& app, Opciones& o) {
    o.weight_replace_prob = 1.0 - o.ncfg.weight_perturb_prob;
    app.add_option("--env", o.env_name, "Entorno (docs/entornos_rltools.md)")->check(CLI::IsMember(envs::envNames()));
    app.add_option("--torque-scale", o.env_opt.torque_scale, "Escala del torque máximo (acrobot 0.2 → ±1 como gymnasium)");
    app.add_option("--seed", o.seed, "Semilla de la réplica (población inicial y episodios)");
    app.add_option("--generations", o.generations);
    app.add_option("--pop", o.ncfg.population_size);
    app.add_option("--episodes", o.episodes, "Episodios de entrenamiento por evaluación (E)");
    app.add_option("--test-episodes", o.test_episodes, "Episodios de test del campeón por generación (M)");
    app.add_option("--final-episodes", o.final_episodes, "Episodios de test final del mejor genoma (F)");
    app.add_option("--step-ms", o.ecfg.step_ms, "Ventana SNN por paso de control (ms)");
    app.add_option("--input-gain", o.ecfg.input_gain, "Corriente para |obs| = 1");
    app.add_option("--spikes-full", o.ecfg.spikes_full_action, "Diferencia de spikes para |acción| = 1");
    app.add_flag("--keep-state", o.ecfg.keep_state, "Reiniciar la SNN solo al comienzo del episodio (política con memoria)");
    app.add_flag("--resample-train", o.resample_train, "Episodios de entrenamiento nuevos en cada generación");
    app.add_option("--compat-threshold", o.ncfg.compat_threshold, "δt inicial");
    app.add_option("--target-species", o.ncfg.target_species, "Especies objetivo (0 = δt fijo)");
    app.add_option("--compat-weight-scale", o.ncfg.compat_weight_scale);
    app.add_option("--weight-power", o.ncfg.weight_perturb_power, "Perturbación de pesos ~ U(-p, p)");
    app.add_option("--weight-replace-prob", o.weight_replace_prob, "Prob. por peso de valor nuevo (Stanley: 0,1)");
    app.add_option("--add-node-weight", o.ncfg.add_node_in_weight);
    app.add_option("--survival", o.ncfg.survival_fraction);
    app.add_option("--weight-init-range", o.ncfg.weight_init_range, "Peso inicial / nuevo ~ U(-r, r)");
    app.add_option("--weight-max", o.ncfg.weight_max, "Límite |w|");
    app.add_option("--mutation-only", o.ncfg.mutation_only_fraction, "Fracción de hijos sin cruce");
    app.add_option("--species-stagnation", o.ncfg.species_stagnation, "Generaciones sin mejora antes de que una especie deje de reproducirse");
    app.add_option("--catalog", o.catalog_txt, "Operadores del catálogo, separados por coma (por defecto los 3 de Stanley)");
    app.add_option("--c4", o.ncfg.c4, "Peso de la diferencia de parámetros de nodo en δ");
    app.add_option("--fine-sigma", o.ncfg.fine_weight_sigma, "weights_fine: σ de la perturbación");
    app.add_flag("--fine-cauchy", o.ncfg.fine_weight_cauchy, "weights_fine con perturbación de Cauchy");
    app.add_option("--directed-iso", o.ncfg.directed_iso_sigma, "weights_directed: σ isotrópico");
    app.add_option("--directed-line", o.ncfg.directed_line_sigma, "weights_directed: σ sobre la línea entre élites");
    app.add_option("--bias-sigma", o.ncfg.bias_sigma, "neuron_bias: σ");
    app.add_option("--bias-max", o.ncfg.bias_max, "neuron_bias: |sesgo| máximo");
    app.add_option("--gain-log-sigma", o.ncfg.gain_log_sigma, "input_gain: σ en escala logarítmica");
    app.add_option("--multi-k", o.ncfg.add_node_multi_k, "add_node_multi: nodos por aplicación");
}

// Lo que neat_control hace después de leer la línea de comando (mismo orden y mismas expresiones)
inline void finalizar(Opciones& o) {
    o.ncfg.weight_perturb_prob = 1.0 - o.weight_replace_prob;
    if (!o.catalog_txt.empty()) o.ncfg.catalog = neat::parseCatalog(o.catalog_txt);
    neat::activarActividad(o.ncfg);
}

}  // namespace opciones_neat

#endif
