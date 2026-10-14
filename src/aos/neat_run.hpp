#ifndef AOS_NEAT_RUN_HPP
#define AOS_NEAT_RUN_HPP

// NEAT como entorno del agente PPO (docs/diseno_ppo.md §3.2–3.3).
//
// NeatRun: una corrida de NEAT con rotación de episodios que avanza por ventanas de decisión. Repite el ciclo de
// neat_control con --resample-train paso a paso, con el mismo orden de las llamadas que consumen números
// aleatorios. Con la misma secuencia de P da exactamente las mismas generaciones (verificación V1, aos_neat_check).
// Agrega lo que necesita el agente:
//   - los rasgos de estado (FeatureExtractor);
//   - Φ, el score del campeón vigente en un conjunto de episodios propio (stream 3), que solo existe al entrenar.
//
// NeatPool: varias corridas en ranuras, con la interfaz que espera el lote de RLtools
// (src/experiment/ppo_lote.hpp): start, advance, obs, phi, t, T y reward. Avanza todas las ranuras juntas:
// reproduce y cierra generaciones en paralelo entre corridas, y evalúa los genomas de todas las corridas en una
// sola región paralela (eval::evaluatePopulations).
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "aos/features.hpp"
#include "eval/evaluator.hpp"
#include "neat/population.hpp"

namespace aos {

struct NeatRunConfig {
    std::string env_name;
    envs::EnvOptions env_opt;
    eval::EvalConfig ecfg;
    neat::NeatConfig ncfg;
    int generations = 0;          // G: la corrida termina tras evaluar la generación G − 1
    int episodes = 10;            // E: episodios de entrenamiento por generación (rotan: stream 1000 + generación)
    int test_episodes = 10;       // M: validación del campeón de cada generación (stream 1)
    int final_episodes = 100;     // F: test final (stream 2)
    int reward_episodes = 50;     // conjunto de recompensa (stream 3); 0 = sin Φ
    int T = 50;                   // decisiones por corrida
    double ref_trivial = 0.0;     // σ(J) = (J − J_trivial) / (J_RL − J_trivial)
    double ref_rl = 1.0;
};

class NeatRun {
public:
    NeatRun(const NeatRunConfig& cfg, uint64_t seed);

    // Fases de una generación. Las orquesta quien corre: el uso secuencial de abajo o NeatPool (en paralelo).
    std::vector<neat::Genome>& genomes() { return pop_.genomes(); }
    const std::vector<uint64_t>& trainingSeeds() const { return gen_seeds_; }
    void closeGeneration();                        // tras asignar el fitness de la generación en curso
    void reproduce(const std::vector<double>& P);  // produce la generación siguiente con P
    bool needsGeneration() const;                  // falta llegar a la frontera de la decisión en curso
    void closeWindow();                            // observación y, si cambió el campeón, Φ

    // Uso secuencial (tests y verificación), con eval::evaluatePopulation
    void startSequential();                                // generación 0 y ventana 0
    void advanceSequential(const std::vector<double>& P);  // una decisión completa
    double finalTest();                                    // campeón vigente en F episodios nuevos (stream 2)

    int t() const { return windows_ - 1; }   // decisiones ya tomadas
    int T() const { return cfg_.T; }
    bool done() const { return windows_ - 1 >= cfg_.T; }
    int generation() const { return closed_gen_; }
    uint64_t seed() const { return seed_; }
    int obsDim() const { return features_.dim(); }
    const std::vector<float>& obs() const { return obs_; }
    double phi() const { return phi_; }
    double sigma(double J) const { return features_.sigma(J); }
    double bestValidation() const { return best_val_; }
    const neat::Genome& champion() const { return best_val_genome_; }
    const neat::GenerationReport& lastReport() const { return last_rep_; }
    const NeatRunConfig& config() const { return cfg_; }
    const std::vector<int>& boundaries() const { return bounds_; }

    // generations.csv con el formato de neat_control. `P_inicial` es la P que se anota en la generación 0
    // (neat_control anota ahí la P del selector).
    void enableCsv(const std::string& path, bool credito_rango, const std::vector<double>& P_inicial);

private:
    NeatRunConfig cfg_;
    uint64_t seed_;
    std::unique_ptr<envs::ControlEnv> probe_;   // validación, Φ y test final (escenas de test, si las hay)
    neat::Population pop_;
    FeatureExtractor features_;
    std::vector<int> bounds_;                   // fronteras de decisión g_0 … g_T
    std::vector<uint64_t> gen_seeds_, val_seeds_, reward_seeds_, final_seeds_;
    int gen_ = 0;                // generación en curso (la última producida)
    int closed_gen_ = -1;        // última generación cerrada
    int windows_ = 0;            // ventanas cerradas
    double best_val_;
    neat::Genome best_val_genome_;
    bool champion_changed_ = false;
    double phi_ = 0.0;
    std::vector<float> obs_;
    neat::GenerationReport last_rep_;
    std::vector<double> P_log_;  // P con la que se produjo la generación en curso (para el CSV)
    std::ofstream csv_;
    bool csv_rank_ = false;
    double gen_t0_ = 0.0;

    void writeCsvRow(double champ_test);
};

class NeatPool {
public:
    struct Finished {
        uint64_t seed;
        double final_test;        // test final crudo (stream 2); NaN si no se mide
        double final_score;       // σ(final_test)
        double ret;               // suma de recompensas
        double best_validation;   // σ de la mejor validación
    };

    NeatPool(const NeatRunConfig& cfg, int slots, uint64_t seed_base, double w_area, double reward_scale,
             bool final_test = true);

    int T() const { return cfg_.T; }
    int slots() const { return static_cast<int>(runs_.size()); }
    int obsDim() const;
    void start(const std::vector<int>& slots);
    void advance(const std::vector<std::vector<double>>& P);
    const std::vector<float>& obs(int s) const { return runs_[s]->obs(); }
    double phi(int s) const { return runs_[s]->phi(); }
    int t(int s) const { return runs_[s]->t(); }
    // r_t = escala · [(Φ_{t+1} − Φ_t) + (w/T)·Φ_{t+1}]  (docs/diseno_ppo.md §2.4)
    double reward(double phi_prev, double phi_next) const {
        return scale_ * ((phi_next - phi_prev) + w_area_ / cfg_.T * phi_next);
    }
    NeatRun& run(int s) { return *runs_[s]; }
    // Registro por decisión (CSV): semilla, t, generación, evaluaciones, Φ antes y después, recompensa, P y la
    // observación con que se decidió
    void enableDecisionLog(const std::string& path, bool append = false);
    // generations.csv de cada corrida en dir/seed_XX/ (verificación): se activa al iniciar cada corrida
    void enableCsv(const std::string& dir, bool credito_rango, const std::vector<double>& P_inicial) {
        csv_dir_ = dir;
        csv_rank_ = credito_rango;
        csv_P0_ = P_inicial;
    }

    // Registro: corridas terminadas y P media desde el último vaciado; contabilidad (V2)
    std::vector<Finished> takeFinished();
    std::vector<double> takeMeanP();
    long started() const { return n_started_; }
    long finishedCount() const { return n_finished_; }
    long decisions() const { return n_decisions_; }
    uint64_t nextSeed() const { return next_seed_; }

private:
    NeatRunConfig cfg_;
    std::vector<std::unique_ptr<NeatRun>> runs_;
    std::vector<double> ret_;
    uint64_t next_seed_;
    double w_area_, scale_;
    bool final_test_;
    std::vector<Finished> finished_;
    std::vector<double> p_sum_;
    long p_count_ = 0, n_started_ = 0, n_finished_ = 0, n_decisions_ = 0;
    std::string csv_dir_;
    bool csv_rank_ = false;
    std::vector<double> csv_P0_;
    std::ofstream dec_log_;

    void evaluateCurrent(const std::vector<int>& slots);
};

}  // namespace aos

#endif
