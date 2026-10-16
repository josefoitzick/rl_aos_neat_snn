#include "evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace eval {

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

std::vector<uint64_t> episodeSeeds(uint64_t run_seed, int n, uint64_t stream) {
    std::vector<uint64_t> s(n);
    for (int k = 0; k < n; k++)
        s[k] = splitmix64(splitmix64(run_seed) ^ splitmix64(stream * 1000003ull + static_cast<uint64_t>(k)))
               & 0x7FFFFFFFull;   // RLtools siembra con index_t; se mantiene en 31 bits
    return s;
}

void ActividadSNN::sumar(const ActividadSNN& o) {
    if (espigas.size() < o.espigas.size()) espigas.resize(o.espigas.size(), 0);
    for (size_t i = 0; i < o.espigas.size(); i++) espigas[i] += o.espigas[i];
    pasos_snn += o.pasos_snn;
    acciones += o.acciones;
    acciones_nulas += o.acciones_nulas;
    acciones_saturadas += o.acciones_saturadas;
}

ResumenActividad resumirActividad(const ActividadSNN& a, const Network& red, double umbral_saturacion) {
    ResumenActividad r;
    auto espigas = [&](int id) { return id < static_cast<int>(a.espigas.size()) ? a.espigas[id] : 0L; };
    int n = 0, mudas = 0, saturadas = 0;
    for (int id : red.getHiddenNeuronIds()) {
        if (!red.getNeuron(id).isEnabled()) continue;
        n++;
        if (espigas(id) == 0) mudas++;
        if (a.pasos_snn > 0 && static_cast<double>(espigas(id)) / a.pasos_snn >= umbral_saturacion) saturadas++;
    }
    if (n > 0) {
        r.ocultas_mudas = static_cast<double>(mudas) / n;
        r.ocultas_saturadas = static_cast<double>(saturadas) / n;
    }
    int ne = 0, emudas = 0;
    for (int id : red.getInputNeuronIds()) {
        ne++;
        if (espigas(id) == 0) emudas++;
    }
    if (ne > 0) r.entradas_mudas = static_cast<double>(emudas) / ne;
    if (a.acciones > 0) {
        r.salidas_mudas = static_cast<double>(a.acciones_nulas) / a.acciones;
        r.accion_saturada = static_cast<double>(a.acciones_saturadas) / a.acciones;
    }
    return r;
}

double runEpisode(std::vector<std::unique_ptr<Network>>& nets, envs::ControlEnv& env, uint64_t seed,
                  const EvalConfig& cfg, const std::vector<double>& gains, ActividadSNN* actividad) {
    const int agents = env.numAgents(), obs_dim = env.obsDim(), act_dim = env.actionDim();
    const std::vector<int> outputs = nets.at(0)->getOutputNeuronIds();
    const int steps = static_cast<int>(cfg.step_ms / nets[0]->getTimestep());
    const bool non_neg = env.nonNegativeObs();
    std::vector<double> currents(numInputs(env)), action(agents * act_dim);
    std::vector<int> count(outputs.size());

    if (actividad && actividad->espigas.size() < nets[0]->getNeuronCount()) actividad->espigas.resize(nets[0]->getNeuronCount(), 0);
    env.reset(seed);
    for (auto& net : nets) net->fastReset();
    double ret = 0.0;
    for (int t = 0; t < env.maxSteps(); t++) {
        const std::vector<double> obs = env.observe();
        for (int ag = 0; ag < agents; ag++) {
            Network& net = *nets[ag];
            const double* o = obs.data() + ag * obs_dim;
            for (int i = 0; i < obs_dim; i++) {
                if (non_neg) {
                    currents[i] = std::max(0.0, o[i]) * cfg.input_gain;
                } else {
                    currents[2 * i]     = o[i] < 0.0 ? -o[i] * cfg.input_gain : 0.0;
                    currents[2 * i + 1] = o[i] >= 0.0 ? o[i] * cfg.input_gain : 0.0;
                }
            }
            for (size_t i = 0; i < gains.size(); i++) currents[i] *= gains[i];   // E1 (vacío: sin cambio)
            if (!cfg.keep_state) net.fastReset();
            net.setInputOffsets(currents);
            std::fill(count.begin(), count.end(), 0);
            for (int k = 0; k < steps; k++) {
                net.step();
                for (size_t q = 0; q < outputs.size(); q++)
                    if (net.getNeuron(outputs[q]).hasSpiked()) count[q]++;
                if (actividad) {   // solo observa
                    const int nn = static_cast<int>(net.getNeuronCount());
                    for (int i = 0; i < nn; i++)
                        if (net.getNeuron(i).hasSpiked()) actividad->espigas[i]++;
                    actividad->pasos_snn++;
                }
            }
            for (int j = 0; j < act_dim; j++)
                action[ag * act_dim + j] =
                    std::clamp((count[2 * j] - count[2 * j + 1]) / cfg.spikes_full_action, -1.0, 1.0);
            if (actividad)
                for (int j = 0; j < act_dim; j++) {
                    actividad->acciones++;
                    if (count[2 * j] == 0 && count[2 * j + 1] == 0) actividad->acciones_nulas++;
                    if (std::abs(action[ag * act_dim + j]) >= 1.0) actividad->acciones_saturadas++;
                }
        }
        envs::StepResult r = env.step(action);
        ret += r.reward;
        if (r.terminated) break;
    }
    return ret;
}

double evaluateGenome(const neat::Genome& g, const neat::NeatConfig& ncfg, envs::ControlEnv& env,
                      const std::vector<uint64_t>& seeds, const EvalConfig& cfg) {
    std::vector<std::unique_ptr<Network>> nets;
    for (int ag = 0; ag < env.numAgents(); ag++) nets.push_back(g.buildNetwork(ncfg));
    const std::vector<double> gains = g.inputGains();
    double sum = 0.0;
    for (uint64_t s : seeds) sum += runEpisode(nets, env, s, cfg, gains);
    return seeds.empty() ? 0.0 : sum / seeds.size();
}

// Un episodio de un genoma con redes propias. Los episodios son independientes (la red se reinicia al empezar
// cada episodio y las sinapsis no tienen estado), así que repartirlos entre hilos no cambia ningún retorno.
static double genomeEpisode(const neat::Genome& g, const neat::NeatConfig& ncfg, envs::ControlEnv& env,
                            uint64_t seed, const EvalConfig& cfg, ActividadSNN* actividad = nullptr) {
    std::vector<std::unique_ptr<Network>> nets;
    for (int ag = 0; ag < env.numAgents(); ag++) nets.push_back(g.buildNetwork(ncfg));
    return runEpisode(nets, env, seed, cfg, g.inputGains(), actividad);
}

// Media en el mismo orden que evaluateGenome: ((0 + r0) + r1) + … y luego / n
static double orderedMean(const double* r, size_t n) {
    double sum = 0.0;
    for (size_t k = 0; k < n; k++) sum += r[k];
    return n == 0 ? 0.0 : sum / n;
}

// Clave del entorno por hilo: dos trabajos con la misma clave pueden compartir la instancia
static std::string envKey(const std::string& name, const envs::EnvOptions& opt) {
    std::ostringstream key;
    key << name << '|' << std::hexfloat << opt.torque_scale << '|' << opt.test_split;
    return key.str();
}

std::vector<double> evaluateGenomes(const std::vector<GenomeJob>& jobs) {
    if (jobs.empty()) return {};
    struct Tarea { int job, episode; };
    std::vector<Tarea> tasks;
    std::vector<size_t> base(jobs.size());
    std::vector<std::string> keys(jobs.size());
    size_t total = 0;
    for (size_t j = 0; j < jobs.size(); j++) {
        const GenomeJob& job = jobs[j];
        if (!job.genome || !job.ncfg || !job.seeds || !job.cfg) throw std::invalid_argument("evaluateGenomes: trabajo incompleto");
        keys[j] = envKey(job.env_name, job.env_opt);
        base[j] = total;
        for (int k = 0; k < static_cast<int>(job.seeds->size()); k++) tasks.push_back({static_cast<int>(j), k});
        total += job.seeds->size();
    }
    std::vector<double> ret(total);
    std::vector<ActividadSNN> parcial(total);   // actividad por episodio; se suma después, en orden
    #pragma omp parallel
    {
        std::map<std::string, std::unique_ptr<envs::ControlEnv>> envs_hilo;
        #pragma omp for schedule(dynamic)
        for (int t = 0; t < static_cast<int>(tasks.size()); t++) {
            const GenomeJob& job = jobs[tasks[t].job];
            auto& env = envs_hilo[keys[tasks[t].job]];
            if (!env) env = envs::makeEnv(job.env_name, job.env_opt);
            const size_t i = base[tasks[t].job] + tasks[t].episode;
            ret[i] = genomeEpisode(*job.genome, *job.ncfg, *env, (*job.seeds)[tasks[t].episode], *job.cfg,
                                   job.actividad ? &parcial[i] : nullptr);
        }
    }
    for (size_t j = 0; j < jobs.size(); j++)
        if (jobs[j].actividad)
            for (size_t k = 0; k < jobs[j].seeds->size(); k++) jobs[j].actividad->sumar(parcial[base[j] + k]);
    std::vector<double> mean(jobs.size());
    for (size_t j = 0; j < jobs.size(); j++) mean[j] = orderedMean(ret.data() + base[j], jobs[j].seeds->size());
    return mean;
}

double evaluateGenomeParallel(const neat::Genome& g, const neat::NeatConfig& ncfg, const std::string& env_name,
                              const envs::EnvOptions& env_opt, const std::vector<uint64_t>& seeds, const EvalConfig& cfg) {
    GenomeJob job;
    job.genome = &g;
    job.ncfg = &ncfg;
    job.env_name = env_name;
    job.env_opt = env_opt;
    job.seeds = &seeds;
    job.cfg = &cfg;
    return evaluateGenomes({job})[0];
}

// F1–F4: suma la actividad de los E episodios de cada genoma (enteros: el orden no importa) y escribe la tasa de cada neurona
static void asignarActividad(std::vector<neat::Genome>& genomes, const ActividadSNN* act, int E) {
    for (size_t i = 0; i < genomes.size(); i++) {
        ActividadSNN a;
        for (int k = 0; k < E; k++) a.sumar(act[i * E + k]);
        genomes[i].asignarTasas(a.espigas, a.pasos_snn);
    }
}

void evaluatePopulation(std::vector<neat::Genome>& genomes, const neat::NeatConfig& ncfg,
                        const std::string& env_name, const envs::EnvOptions& env_opt,
                        const std::vector<uint64_t>& seeds, const EvalConfig& cfg) {
    const int G = static_cast<int>(genomes.size()), E = static_cast<int>(seeds.size());
    std::vector<double> ret(static_cast<size_t>(G) * E);
    std::vector<ActividadSNN> act(ncfg.medir_actividad ? ret.size() : 0);   // F1–F4: actividad por episodio
    #pragma omp parallel
    {
        auto env = envs::makeEnv(env_name, env_opt);
        #pragma omp for schedule(dynamic)
        for (int t = 0; t < G * E; t++)
            ret[t] = genomeEpisode(genomes[t / E], ncfg, *env, seeds[t % E], cfg, act.empty() ? nullptr : &act[t]);
    }
    for (int i = 0; i < G; i++) genomes[i].fitness = orderedMean(ret.data() + static_cast<size_t>(i) * E, E);
    if (!act.empty()) asignarActividad(genomes, act.data(), E);
}

void evaluatePopulations(const std::vector<PopulationJob>& jobs) {
    struct Tarea { int job, genome, episode; };
    std::vector<Tarea> tasks;
    std::vector<size_t> base(jobs.size());
    std::vector<std::string> keys(jobs.size());
    size_t total = 0;
    for (size_t j = 0; j < jobs.size(); j++) {
        const PopulationJob& job = jobs[j];
        if (!job.genomes || !job.ncfg || !job.seeds || !job.cfg) throw std::invalid_argument("evaluatePopulations: trabajo incompleto");
        keys[j] = envKey(job.env_name, job.env_opt);
        base[j] = total;
        const int E = static_cast<int>(job.seeds->size());
        for (int i = 0; i < static_cast<int>(job.genomes->size()); i++)
            for (int k = 0; k < E; k++) tasks.push_back({static_cast<int>(j), i, k});
        total += job.genomes->size() * job.seeds->size();
    }
    std::vector<double> ret(total);
    bool medir = false;
    for (const PopulationJob& job : jobs) medir = medir || job.ncfg->medir_actividad;
    std::vector<ActividadSNN> act(medir ? total : 0);   // F1–F4: actividad por episodio
    #pragma omp parallel
    {
        std::map<std::string, std::unique_ptr<envs::ControlEnv>> envs_hilo;
        #pragma omp for schedule(dynamic)
        for (int t = 0; t < static_cast<int>(tasks.size()); t++) {
            const Tarea& ta = tasks[t];
            const PopulationJob& job = jobs[ta.job];
            auto& env = envs_hilo[keys[ta.job]];
            if (!env) env = envs::makeEnv(job.env_name, job.env_opt);
            const size_t E = job.seeds->size();
            const size_t r = base[ta.job] + static_cast<size_t>(ta.genome) * E + ta.episode;
            ret[r] = genomeEpisode((*job.genomes)[ta.genome], *job.ncfg, *env, (*job.seeds)[ta.episode], *job.cfg,
                                   job.ncfg->medir_actividad ? &act[r] : nullptr);
        }
    }
    for (size_t j = 0; j < jobs.size(); j++) {
        const size_t E = jobs[j].seeds->size();
        for (size_t i = 0; i < jobs[j].genomes->size(); i++)
            (*jobs[j].genomes)[i].fitness = orderedMean(ret.data() + base[j] + i * E, E);
        if (jobs[j].ncfg->medir_actividad) asignarActividad(*jobs[j].genomes, act.data() + base[j], static_cast<int>(E));
    }
}

}  // namespace eval
