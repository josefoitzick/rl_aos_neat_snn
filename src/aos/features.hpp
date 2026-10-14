#ifndef AOS_FEATURES_HPP
#define AOS_FEATURES_HPP

// Estado del agente PPO (docs/diseno_ppo.md §2.3): 14 rasgos globales + 2 por operador del catálogo, sin
// unidades del entorno. Se alimenta con el GenerationReport de cada generación (con los campos que llena el
// ciclo de NEAT: validaciones, evaluaciones y crédito por rango) y entrega una observación al cerrar cada
// ventana de decisión. Lo comparten el entorno de entrenamiento (NeatRun) y el despliegue en neat_control.
//
//  0  τ                 evaluaciones usadas / presupuesto
//  1  V*                σ de la mejor validación hasta ahora (campeón vigente)
//  2  V_gen             σ media, en la ventana, de la validación del campeón de cada generación
//  3  V* − V_gen        brecha (regresión de la población)
//  4  ΔV*               cambio de V* desde la ventana anterior
//  5  estancamiento     τ − τ de la última mejora de V*
//  6  dispersión        DE del fitness de entrenamiento de la última generación / (J_RL − J_trivial)
//  7  especies          n.º de especies / especies objetivo
//  8  especie mayor     fracción de la población
//  9  umbral δt         log(δt / δt inicial)
// 10  conexiones        log((conexiones medias + 1) / (conexiones iniciales + 1))
// 11  ocultas           log(1 + ocultas medias)
// 12  conex. campeón    log((conexiones del campeón + 1) / (conexiones iniciales + 1))
// 13  ocultas campeón   log(1 + ocultas del campeón)
// 14 … 14+K−1           crédito_k: media móvil (η por ventana) del percentil de los hijos del operador k − 0,5
// 14+K … 14+2K−1        fallo_k: media móvil de la fracción de aplicaciones fallidas del operador k
// σ(J) = (J − J_trivial) / (J_RL − J_trivial) es el score de siempre.
#include <vector>
#include "neat/selector.hpp"

namespace aos {

struct FeatureConfig {
    int K = 3;                      // operadores del catálogo
    long budget = 1;                // presupuesto: generaciones × población
    double ref_trivial = 0.0;       // J_trivial del entorno
    double ref_rl = 1.0;            // J_RL del entorno
    int target_species = 10;        // especies objetivo (si la especiación es dinámica)
    double compat_threshold0 = 3.0; // δt inicial
    int conns0 = 1;                 // conexiones de la topología inicial (entradas × salidas)
    double ema = 0.5;               // η de la media móvil del crédito y de los fallos, por ventana
};

class FeatureExtractor {
public:
    explicit FeatureExtractor(const FeatureConfig& cfg);
    int dim() const { return 14 + 2 * cfg_.K; }
    const FeatureConfig& config() const { return cfg_; }
    double sigma(double J) const { return (J - cfg_.ref_trivial) / (cfg_.ref_rl - cfg_.ref_trivial); }

    // Una generación cerrada (con validaciones, evaluaciones y crédito por rango llenos)
    void addGeneration(const neat::GenerationReport& rep);
    // Cierra la ventana de la decisión: devuelve la observación y empieza una ventana nueva
    std::vector<float> closeWindow();

private:
    FeatureConfig cfg_;
    // ventana en curso
    double vgen_sum_ = 0.0;
    int vgen_n_ = 0;
    std::vector<double> win_rank_, win_n_, win_fail_;
    // última generación vista
    bool have_gen_ = false;
    double tau_ = 0.0, best_val_ = 0.0;
    neat::PopulationStats last_;
    // entre ventanas
    bool first_close_ = true;
    double vbest_prev_close_ = 0.0;
    double best_seen_ = 0.0;
    bool any_best_ = false;
    double tau_improve_ = 0.0;
    std::vector<double> credit_, fail_;
};

}  // namespace aos

#endif
