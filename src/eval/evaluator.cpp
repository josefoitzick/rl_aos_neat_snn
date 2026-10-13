#include "evaluator.hpp"

#include <algorithm>
#include <omp.h>

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

double runEpisode(std::vector<std::unique_ptr<Network>>& nets, envs::ControlEnv& env, uint64_t seed,
                  const EvalConfig& cfg, const std::vector<double>& gains) {
    const int agents = env.numAgents(), obs_dim = env.obsDim(), act_dim = env.actionDim();
    const std::vector<int> outputs = nets.at(0)->getOutputNeuronIds();
    const int steps = static_cast<int>(cfg.step_ms / nets[0]->getTimestep());
    const bool non_neg = env.nonNegativeObs();
    std::vector<double> currents(numInputs(env)), action(agents * act_dim);
    std::vector<int> count(outputs.size());

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
            }
            for (int j = 0; j < act_dim; j++)
                action[ag * act_dim + j] =
                    std::clamp((count[2 * j] - count[2 * j + 1]) / cfg.spikes_full_action, -1.0, 1.0);
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

void evaluatePopulation(std::vector<neat::Genome>& genomes, const neat::NeatConfig& ncfg,
                        const std::string& env_name, const envs::EnvOptions& env_opt,
                        const std::vector<uint64_t>& seeds, const EvalConfig& cfg) {
    #pragma omp parallel
    {
        auto env = envs::makeEnv(env_name, env_opt);
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(genomes.size()); i++)
            genomes[i].fitness = evaluateGenome(genomes[i], ncfg, *env, seeds, cfg);
    }
}

}  // namespace eval
