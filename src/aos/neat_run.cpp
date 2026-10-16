#include "neat_run.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include "aos/action_map.hpp"

namespace aos {

namespace {

double ahora() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

envs::EnvOptions opcionesDePrueba(envs::EnvOptions o) {
    o.test_split = true;   // como neat_control: el campeón se mide en escenas de test, si el entorno las tiene
    return o;
}

FeatureConfig configRasgos(const NeatRunConfig& cfg, int n_in, int n_out) {
    FeatureConfig f;
    f.K = static_cast<int>(cfg.ncfg.catalog.size());
    f.budget = static_cast<long>(cfg.generations) * cfg.ncfg.population_size;
    f.ref_trivial = cfg.ref_trivial;
    f.ref_rl = cfg.ref_rl;
    f.target_species = cfg.ncfg.target_species > 0 ? cfg.ncfg.target_species : 10;
    f.compat_threshold0 = cfg.ncfg.compat_threshold;
    f.conns0 = n_in * n_out;
    f.compacto = cfg.estado_compacto;
    f.snn = cfg.estado_snn;
    for (size_t k = 0; k < cfg.ncfg.catalog.size(); k++)
        if (cfg.ncfg.catalog[k] == neat::Operator::MutateWeights) f.op_pesos = static_cast<int>(k);
    f.potencia_base = cfg.ncfg.weight_perturb_power;
    return f;
}

// omp parallel for que no aborta el programa si una iteración lanza: guarda la primera excepción y la relanza
template <typename F>
void paraleloSeguro(int n, F&& f) {
    std::exception_ptr err = nullptr;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        try {
            f(i);
        } catch (...) {
            #pragma omp critical(neat_pool_err)
            if (!err) err = std::current_exception();
        }
    }
    if (err) std::rethrow_exception(err);
}

}  // namespace

NeatRun::NeatRun(const NeatRunConfig& cfg, uint64_t seed)
    : cfg_(cfg),
      seed_(seed),
      probe_(envs::makeEnv(cfg.env_name, opcionesDePrueba(cfg.env_opt))),
      test_opt_(opcionesDePrueba(cfg.env_opt)),
      pop_(cfg.ncfg, eval::numInputs(*probe_), eval::numOutputs(*probe_), seed),
      features_(configRasgos(cfg, eval::numInputs(*probe_), eval::numOutputs(*probe_))),
      bounds_(decisionBoundaries(cfg.generations, cfg.T)),
      best_val_(-std::numeric_limits<double>::infinity()) {
    if (cfg.episodes < 1 || cfg.test_episodes < 1) throw std::invalid_argument("NeatRun: se necesitan episodios de entrenamiento y de validación");
    gen_seeds_ = eval::episodeSeeds(seed, cfg.episodes, 1000);
    val_seeds_ = eval::episodeSeeds(seed, cfg.test_episodes, 1);
    final_seeds_ = eval::episodeSeeds(seed, cfg.final_episodes, 2);
    if (cfg.reward_episodes > 0) reward_seeds_ = eval::episodeSeeds(seed, cfg.reward_episodes, 3);
    gen_t0_ = ahora();
}

size_t NeatRun::championIndex() const {
    const auto& G = pop_.genomes();
    size_t champ = 0;
    for (size_t i = 1; i < G.size(); i++)
        if (G[i].fitness > G[champ].fitness) champ = i;
    return champ;
}

eval::GenomeJob NeatRun::job(const neat::Genome& g, const std::vector<uint64_t>& seeds) const {
    eval::GenomeJob j;
    j.genome = &g;
    j.ncfg = &cfg_.ncfg;
    j.env_name = cfg_.env_name;
    j.env_opt = test_opt_;
    j.seeds = &seeds;
    j.cfg = &cfg_.ecfg;
    return j;
}

eval::GenomeJob NeatRun::validationJob() {
    eval::GenomeJob j = job(pop_.genomes()[championIndex()], val_seeds_);
    if (cfg_.estado_snn) {
        act_val_ = eval::ActividadSNN{};
        j.actividad = &act_val_;
    }
    return j;
}
eval::GenomeJob NeatRun::phiJob() const { return job(best_val_genome_, reward_seeds_); }
eval::GenomeJob NeatRun::finalJob() const { return job(best_val_genome_, final_seeds_); }

void NeatRun::closeGeneration() {
    if (closed_gen_ == gen_) throw std::logic_error("NeatRun: la generación ya está cerrada");
    closeGeneration(eval::evaluateGenomes({validationJob()})[0]);
}

void NeatRun::closeGeneration(double champ_test) {
    if (closed_gen_ == gen_) throw std::logic_error("NeatRun: la generación ya está cerrada");
    auto& G = pop_.genomes();
    // Igual que neat_control con --resample-train: campeón por fitness de entrenamiento, validado en el stream 1
    // (champ_test); el campeón vigente es el de mejor validación
    const size_t champ = championIndex();
    if (cfg_.estado_snn) {   // actividad medida en la validación de este mismo campeón (validationJob)
        const auto red = G[champ].buildNetwork(cfg_.ncfg);
        const eval::ResumenActividad r = eval::resumirActividad(act_val_, *red);
        features_.addActividad({r.ocultas_mudas, r.ocultas_saturadas, r.entradas_mudas, r.salidas_mudas, r.accion_saturada});
    }
    if (champ_test > best_val_) {
        best_val_ = champ_test;
        best_val_genome_ = G[champ];
        champion_changed_ = true;
    }
    std::vector<neat::OpRank> op_rank = neat::rankCredit(G, cfg_.ncfg.catalog.size());
    neat::GenerationReport rep = pop_.endGeneration();
    rep.champion_validation = champ_test;
    rep.best_validation = best_val_;
    rep.evaluations = static_cast<long>(gen_ + 1) * cfg_.ncfg.population_size;
    rep.op_rank = std::move(op_rank);
    features_.addGeneration(rep);
    last_rep_ = std::move(rep);
    closed_gen_ = gen_;
    if (csv_.is_open()) writeCsvRow(champ_test);
}

void NeatRun::reproduce(const std::vector<double>& P, const neat::Variacion* v) {
    if (closed_gen_ != gen_) throw std::logic_error("NeatRun: la generación en curso no está cerrada");
    if (done()) throw std::logic_error("NeatRun: la corrida terminó");
    if (P.size() != cfg_.ncfg.catalog.size()) throw std::invalid_argument("NeatRun: P de tamaño distinto del catálogo");
    if (v) {
        pop_.setVariacion(*v);
        features_.setPotencia(v->weight_perturb_power);
    }
    pop_.reproduce(P);
    gen_++;
    gen_seeds_ = eval::episodeSeeds(seed_, cfg_.episodes, 1000 + static_cast<uint64_t>(gen_));
    P_log_ = P;
    gen_t0_ = ahora();
}

bool NeatRun::needsGeneration() const {
    return windows_ > 0 && !done() && closed_gen_ < bounds_[windows_];
}

void NeatRun::closeWindow() {
    closeWindow(needsPhi() ? eval::evaluateGenomes({phiJob()})[0] : std::numeric_limits<double>::quiet_NaN());
}

void NeatRun::closeWindow(double phi_return) {
    if (closed_gen_ != gen_) throw std::logic_error("NeatRun: la generación en curso no está cerrada");
    if (windows_ > cfg_.T || closed_gen_ != bounds_[windows_]) throw std::logic_error("NeatRun: la ventana no está en su frontera");
    const bool medir = needsPhi();
    if (medir && std::isnan(phi_return)) throw std::logic_error("NeatRun: falta el retorno de Φ");
    obs_ = features_.closeWindow();
    if (medir) phi_ = features_.sigma(phi_return);
    champion_changed_ = false;
    windows_++;
}

void NeatRun::startSequential() {
    if (windows_ != 0) throw std::logic_error("NeatRun: la corrida ya empezó");
    eval::evaluatePopulation(pop_.genomes(), cfg_.ncfg, cfg_.env_name, cfg_.env_opt, gen_seeds_, cfg_.ecfg);
    closeGeneration();
    closeWindow();
}

void NeatRun::advanceSequential(const std::vector<double>& P) {
    if (windows_ == 0) throw std::logic_error("NeatRun: falta startSequential");
    if (done()) throw std::logic_error("NeatRun: la corrida terminó");
    while (needsGeneration()) {
        reproduce(P);
        eval::evaluatePopulation(pop_.genomes(), cfg_.ncfg, cfg_.env_name, cfg_.env_opt, gen_seeds_, cfg_.ecfg);
        closeGeneration();
    }
    closeWindow();
}

double NeatRun::finalTest() {
    return eval::evaluateGenomes({finalJob()})[0];
}

void NeatRun::enableCsv(const std::string& path, bool credito_rango, const std::vector<double>& P_inicial) {
    if (closed_gen_ >= 0) throw std::logic_error("NeatRun::enableCsv: llamar antes de cerrar la generación 0");
    csv_.open(path);
    if (!csv_) throw std::runtime_error("NeatRun: no se pudo abrir " + path);
    csv_rank_ = credito_rango;
    P_log_ = P_inicial;
    csv_ << "gen,evaluations,best_train,mean_train,std_train,champion_test,species,compat_threshold,"
            "champion_hidden,champion_conns,mean_hidden,mean_conns,seconds";
    for (neat::Operator o : cfg_.ncfg.catalog) {
        const char* op = neat::operatorName(o);
        csv_ << ",p_" << op << ",n_" << op << ",improved_" << op;
    }
    if (csv_rank_)
        for (neat::Operator o : cfg_.ncfg.catalog) {
            const char* op = neat::operatorName(o);
            csv_ << ",rank_" << op << ",top_" << op << ",fail_" << op;
        }
    csv_ << "\n";
}

// Misma fila que neat_control (mismas expresiones, para que la comparación sea bit a bit salvo los segundos)
void NeatRun::writeCsvRow(double champ_test) {
    const auto& st = last_rep_.stats;
    const size_t K = cfg_.ncfg.catalog.size();
    std::vector<int> n_op(K, 0), improved(K, 0);
    for (const auto& o : last_rep_.outcomes) {
        n_op[o.slotIndex()]++;
        if (o.child_fitness > o.parent_fitness) improved[o.slotIndex()]++;
    }
    const double secs = ahora() - gen_t0_;
    csv_ << gen_ << "," << static_cast<long>(gen_ + 1) * cfg_.ncfg.population_size << "," << st.best << "," << st.mean << ","
         << st.stdev << "," << champ_test << "," << st.num_species << "," << st.compat_threshold << "," << st.champion_hidden
         << "," << st.champion_conns << "," << st.mean_hidden << "," << st.mean_conns << "," << secs;
    for (size_t i = 0; i < K; i++)
        csv_ << "," << P_log_[i] << "," << n_op[i] << "," << (n_op[i] ? static_cast<double>(improved[i]) / n_op[i] : 0.0);
    if (csv_rank_)
        for (size_t i = 0; i < K; i++) {
            const neat::OpRank& r = last_rep_.op_rank[i];
            if (n_op[i])
                csv_ << "," << r.rank_sum / n_op[i] << "," << static_cast<double>(r.top) / n_op[i] << ","
                     << static_cast<double>(r.fail) / n_op[i];
            else
                csv_ << ",,,";
        }
    csv_ << "\n";
    csv_.flush();
}

NeatPool::NeatPool(const NeatRunConfig& cfg, int slots, uint64_t seed_base, double w_area, double reward_scale, bool final_test)
    : NeatPool(std::vector<NeatRunConfig>(slots < 1 ? 0 : slots, cfg), seed_base, w_area, reward_scale, final_test) {}

NeatPool::NeatPool(const std::vector<NeatRunConfig>& cfgs, uint64_t seed_base, double w_area, double reward_scale, bool final_test)
    : cfg_(cfgs.empty() ? NeatRunConfig{} : cfgs.front()), slot_cfg_(cfgs), runs_(cfgs.size()), ret_(cfgs.size(), 0.0),
      next_seed_(seed_base), w_area_(w_area), scale_(reward_scale), final_test_(final_test),
      p_sum_(cfg_.ncfg.catalog.size(), 0.0) {
    if (cfgs.empty()) throw std::invalid_argument("NeatPool: se necesita al menos una ranura");
    for (const auto& c : cfgs)
        if (c.T != cfg_.T || c.ncfg.catalog.size() != cfg_.ncfg.catalog.size())
            throw std::invalid_argument("NeatPool: todas las ranuras deben tener el mismo T y el mismo tamaño de catálogo");
}

int NeatPool::obsDim() const {
    return cfg_.estado_snn        ? FeatureExtractor::DIM_SNN
           : cfg_.estado_compacto ? FeatureExtractor::DIM_COMPACTO
                                  : 14 + 2 * static_cast<int>(cfg_.ncfg.catalog.size());
}

void NeatPool::evaluateCurrent(const std::vector<int>& slots) {
    std::vector<eval::PopulationJob> jobs;
    for (int s : slots) {
        NeatRun& r = *runs_[s];
        eval::PopulationJob j;
        j.genomes = &r.genomes();
        j.ncfg = &r.config().ncfg;
        j.env_name = r.config().env_name;
        j.env_opt = r.config().env_opt;
        j.seeds = &r.trainingSeeds();
        j.cfg = &r.config().ecfg;
        jobs.push_back(j);
    }
    eval::evaluatePopulations(jobs);
}

void NeatPool::closeGenerations(const std::vector<int>& slots) {
    std::vector<eval::GenomeJob> jobs;
    for (int s : slots) jobs.push_back(runs_[s]->validationJob());
    const std::vector<double> v = eval::evaluateGenomes(jobs);
    paraleloSeguro(static_cast<int>(slots.size()), [&](int i) { runs_[slots[i]]->closeGeneration(v[i]); });
}

void NeatPool::closeWindows(const std::vector<int>& slots) {
    std::vector<eval::GenomeJob> jobs;
    std::vector<int> k(slots.size(), -1);
    for (size_t i = 0; i < slots.size(); i++)
        if (runs_[slots[i]]->needsPhi()) {
            k[i] = static_cast<int>(jobs.size());
            jobs.push_back(runs_[slots[i]]->phiJob());
        }
    const std::vector<double> J = eval::evaluateGenomes(jobs);
    paraleloSeguro(static_cast<int>(slots.size()), [&](int i) {
        runs_[slots[i]]->closeWindow(k[i] >= 0 ? J[k[i]] : std::numeric_limits<double>::quiet_NaN());
    });
}

void NeatPool::start(const std::vector<int>& slots) {
    for (int s : slots) {
        if (s < 0 || s >= this->slots()) throw std::out_of_range("NeatPool::start: ranura inexistente");
        if (runs_[s] && !runs_[s]->done()) throw std::logic_error("NeatPool::start: se reinicia una corrida que no terminó");
        runs_[s] = std::make_unique<NeatRun>(slot_cfg_[s], next_seed_++);
        if (!csv_dir_.empty()) {
            char sub[32];
            std::snprintf(sub, sizeof sub, "seed_%02llu", static_cast<unsigned long long>(runs_[s]->seed()));
            const std::string d = csv_dir_ + "/" + sub;
            std::filesystem::create_directories(d);
            runs_[s]->enableCsv(d + "/generations.csv", csv_rank_, csv_P0_);
        }
        ret_[s] = 0.0;
        n_started_++;
    }
    if (slots.empty()) return;
    evaluateCurrent(slots);
    closeGenerations(slots);
    closeWindows(slots);
}

void NeatPool::enableDecisionLog(const std::string& path, bool append, bool con_intensidades) {
    dec_intensidades_ = con_intensidades;
    const bool header = !append || !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    dec_log_.open(path, append ? std::ios::app : std::ios::trunc);
    if (!dec_log_) throw std::runtime_error("NeatPool: no se pudo abrir " + path);
    if (header) {
        dec_log_ << "semilla,t,generacion,evaluaciones,phi_antes,phi,recompensa";
        for (size_t k = 0; k < cfg_.ncfg.catalog.size(); k++) dec_log_ << ",p" << k;
        if (dec_intensidades_) dec_log_ << ",potencia,reasignar,solo_mutacion";
        for (int d = 0; d < obsDim(); d++) dec_log_ << ",o" << d;
        dec_log_ << "\n";
    }
}

void NeatPool::advance(const std::vector<std::vector<double>>& P, const std::vector<neat::Variacion>* V) {
    const int n = slots();
    if (static_cast<int>(P.size()) != n) throw std::invalid_argument("NeatPool::advance: falta P por ranura");
    if (V && static_cast<int>(V->size()) != n) throw std::invalid_argument("NeatPool::advance: faltan intensidades por ranura");
    std::vector<std::vector<float>> obs0;
    std::vector<int> t0;
    if (dec_log_.is_open())
        for (int s = 0; s < n; s++) {
            obs0.push_back(runs_[s] ? runs_[s]->obs() : std::vector<float>());
            t0.push_back(runs_[s] ? runs_[s]->t() : -1);
        }
    std::vector<double> phi0(n);
    for (int s = 0; s < n; s++) {
        if (!runs_[s]) throw std::logic_error("NeatPool::advance: ranura sin corrida");
        if (runs_[s]->done()) throw std::logic_error("NeatPool::advance: una decisión de más");
        if (P[s].size() != cfg_.ncfg.catalog.size()) throw std::invalid_argument("NeatPool::advance: P de tamaño distinto del catálogo");
        phi0[s] = runs_[s]->phi();
    }
    // Generaciones hasta la frontera de cada corrida (con presupuestos distintos, las ventanas pueden tener distinto largo)
    while (true) {
        std::vector<int> act;
        for (int s = 0; s < n; s++)
            if (runs_[s]->needsGeneration()) act.push_back(s);
        if (act.empty()) break;
        paraleloSeguro(static_cast<int>(act.size()), [&](int i) {
            runs_[act[i]]->reproduce(P[act[i]], V ? &(*V)[act[i]] : nullptr);
        });
        evaluateCurrent(act);
        closeGenerations(act);
    }
    std::vector<int> todas(n);
    for (int s = 0; s < n; s++) todas[s] = s;
    closeWindows(todas);
    for (int s = 0; s < n; s++) {
        const double r = reward(phi0[s], runs_[s]->phi());
        ret_[s] += r;
        for (size_t k = 0; k < p_sum_.size(); k++) p_sum_[k] += P[s][k];
        p_count_++;
        if (V) {
            const neat::Variacion& v = (*V)[s];
            v_sum_[0] += v.weight_perturb_power;
            v_sum_[1] += 1.0 - v.weight_perturb_prob;
            v_sum_[2] += v.mutation_only_fraction;
            v_count_++;
        }
        n_decisions_++;
        if (dec_log_.is_open()) {
            const NeatRun& run = *runs_[s];
            dec_log_ << run.seed() << "," << t0[s] << "," << run.generation() << "," << run.lastReport().evaluations << "," << phi0[s]
                     << "," << run.phi() << "," << r;
            for (double p : P[s]) dec_log_ << "," << p;
            if (dec_intensidades_) {
                const neat::Variacion v = V ? (*V)[s] : run.variacion();
                dec_log_ << "," << v.weight_perturb_power << "," << 1.0 - v.weight_perturb_prob << "," << v.mutation_only_fraction;
            }
            for (float o : obs0[s]) dec_log_ << "," << o;
            dec_log_ << "\n";
        }
    }
    if (dec_log_.is_open()) dec_log_.flush();
    std::vector<int> fin;
    for (int s = 0; s < n; s++)
        if (runs_[s]->done()) fin.push_back(s);
    std::vector<double> ft(fin.size(), std::numeric_limits<double>::quiet_NaN());
    if (final_test_) {
        std::vector<eval::GenomeJob> jobs;
        for (int s : fin) jobs.push_back(runs_[s]->finalJob());
        ft = eval::evaluateGenomes(jobs);
    }
    for (size_t i = 0; i < fin.size(); i++) {
        NeatRun& r = *runs_[fin[i]];
        finished_.push_back({r.seed(), ft[i], final_test_ ? r.sigma(ft[i]) : std::numeric_limits<double>::quiet_NaN(), ret_[fin[i]],
                             r.sigma(r.bestValidation()), r.config().env_name});
        n_finished_++;
    }
}

std::vector<NeatPool::Finished> NeatPool::takeFinished() {
    auto f = std::move(finished_);
    finished_.clear();
    return f;
}

std::vector<double> NeatPool::takeMeanVariacion() {
    std::vector<double> m;
    if (v_count_ > 0)
        for (double x : v_sum_) m.push_back(x / v_count_);
    std::fill(v_sum_.begin(), v_sum_.end(), 0.0);
    v_count_ = 0;
    return m;
}

std::vector<double> NeatPool::takeMeanP() {
    std::vector<double> m(p_sum_.size(), 0.0);
    for (size_t k = 0; k < m.size(); k++) m[k] = p_count_ ? p_sum_[k] / p_count_ : 0.0;
    std::fill(p_sum_.begin(), p_sum_.end(), 0.0);
    p_count_ = 0;
    return m;
}

}  // namespace aos
