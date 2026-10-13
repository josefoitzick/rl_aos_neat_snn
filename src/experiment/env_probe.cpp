// Sonda de entornos (etapa A de la selección de entornos, docs/entornos_rltools.md):
// por entorno, dimensiones, horizonte, retornos de referencia (acción aleatoria y acción nula),
// estadísticas de la observación cruda y costo de evaluar una SNN mínima.
//
// Salida: una línea por entorno en la terminal y, con --out, <out>/probe.csv y <out>/obs_stats.csv.
#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "eval/evaluator.hpp"
#include "neat/innovation.hpp"

using Clock = std::chrono::steady_clock;

struct Returns { double mean = 0, sd = 0, mean_len = 0; };

// Política de acciones: "random" ~ U(−1, 1) por dimensión con RNG fijo; "zero" = 0.
static Returns rollouts(envs::ControlEnv& env, const std::vector<uint64_t>& seeds, bool random,
                        std::vector<std::vector<double>>* raw_samples) {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> rets;
    double steps = 0;
    const int A = env.numAgents() * env.actionDim();
    for (uint64_t s : seeds) {
        env.reset(s);
        double ret = 0;
        for (int t = 0; t < env.maxSteps(); t++) {
            if (raw_samples) raw_samples->push_back(env.rawObserve());
            std::vector<double> a(A, 0.0);
            if (random) for (double& x : a) x = U(rng);
            auto r = env.step(a);
            ret += r.reward;
            steps++;
            if (r.terminated) break;
        }
        rets.push_back(ret);
    }
    Returns R;
    for (double r : rets) R.mean += r;
    R.mean /= rets.size();
    for (double r : rets) R.sd += (r - R.mean) * (r - R.mean);
    R.sd = rets.size() > 1 ? std::sqrt(R.sd / (rets.size() - 1)) : 0.0;
    R.mean_len = steps / seeds.size();
    return R;
}

static double quantile(std::vector<double> v, double q) {
    if (v.empty()) return NAN;
    size_t k = static_cast<size_t>(q * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

int main(int argc, char** argv) {
    CLI::App app{"Sonda de entornos: referencias, cotas de observación y costo de la SNN"};
    std::vector<std::string> names;
    std::string out_dir;
    int episodes = 20, snn_episodes = 2;
    envs::EnvOptions opt;
    eval::EvalConfig ecfg;
    app.add_option("--envs", names, "Entornos (por defecto, todos)");
    app.add_option("--episodes", episodes, "Episodios para las referencias y las estadísticas");
    app.add_option("--snn-episodes", snn_episodes, "Episodios para medir el costo de la SNN");
    app.add_option("--torque-scale", opt.torque_scale);
    app.add_flag("--keep-state", ecfg.keep_state);
    app.add_option("--out", out_dir, "Directorio para probe.csv y obs_stats.csv");
    CLI11_PARSE(app, argc, argv);
    if (names.empty()) names = envs::envNames();

    std::ofstream probe_csv, obs_csv;
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
        probe_csv.open(out_dir + "/probe.csv");
        probe_csv << "env,agents,obs_dim,act_dim,snn_inputs,snn_outputs,max_steps,random_mean,random_sd,random_len,"
                     "zero_mean,zero_sd,zero_len,env_us_per_step,snn_ms_per_episode,snn_len\n";
        obs_csv.open(out_dir + "/obs_stats.csv");
        obs_csv << "env,dim,min,max,p01,p99,abs_p99,norm_saturated_frac\n";
    }

    const auto seeds = eval::episodeSeeds(2026, episodes, 9);
    std::printf("%-20s %3s %4s %4s %6s %6s  %12s %12s  %9s %10s\n", "entorno", "ag", "obs", "act", "pasos",
                "ent/sal", "aleatoria", "nula", "us/paso", "SNN ms/ep");
    for (const auto& name : names) {
        auto env = envs::makeEnv(name, opt);
        std::vector<std::vector<double>> raw;

        auto t0 = Clock::now();
        Returns rnd = rollouts(*env, seeds, true, &raw);
        const double env_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() /
                              std::max(1.0, rnd.mean_len * seeds.size());
        Returns zero = rollouts(*env, seeds, false, nullptr);

        // Saturación de la normalización: fracción de |observe()| = 1 bajo acciones aleatorias
        std::vector<double> sat(raw.empty() ? 0 : raw[0].size(), 0.0);
        {
            std::mt19937_64 rng(777);
            std::uniform_real_distribution<double> U(-1.0, 1.0);
            double n = 0;
            for (uint64_t s : seeds) {
                env->reset(s);
                for (int t = 0; t < env->maxSteps(); t++) {
                    auto o = env->observe();
                    for (size_t i = 0; i < o.size(); i++) sat[i] += std::abs(o[i]) >= 1.0 - 1e-12;
                    n++;
                    std::vector<double> a(env->numAgents() * env->actionDim());
                    for (double& x : a) x = U(rng);
                    if (env->step(a).terminated) break;
                }
            }
            for (double& x : sat) x /= n;
        }

        // Costo de la SNN: genoma mínimo (topología inicial de NEAT, totalmente conectada)
        neat::NeatConfig ncfg;
        const int n_in = eval::numInputs(*env), n_out = eval::numOutputs(*env);
        neat::InnovationTracker tr(n_in + n_out);
        neat::Rng rng(1);
        neat::Genome g = neat::Genome::minimal(0, n_in, n_out, ncfg, tr, rng);
        double snn_len = 0;
        auto t1 = Clock::now();
        for (int k = 0; k < snn_episodes; k++)
            (void)eval::evaluateGenome(g, ncfg, *env, {seeds[k % seeds.size()]}, ecfg);
        const double snn_ms = std::chrono::duration<double, std::milli>(Clock::now() - t1).count() / snn_episodes;
        snn_len = rnd.mean_len;   // referencia de duración (el largo real depende de la política)

        std::printf("%-20s %3d %4d %4d %6d %3d/%-3d %12.2f %12.2f  %9.1f %10.1f\n", name.c_str(), env->numAgents(),
                    env->obsDim(), env->actionDim(), env->maxSteps(), n_in, n_out, rnd.mean, zero.mean, env_us, snn_ms);
        if (probe_csv.is_open()) {
            probe_csv << name << "," << env->numAgents() << "," << env->obsDim() << "," << env->actionDim() << ","
                      << n_in << "," << n_out << "," << env->maxSteps() << "," << rnd.mean << "," << rnd.sd << ","
                      << rnd.mean_len << "," << zero.mean << "," << zero.sd << "," << zero.mean_len << "," << env_us
                      << "," << snn_ms << "," << snn_len << "\n";
            for (size_t i = 0; !raw.empty() && i < raw[0].size(); i++) {
                std::vector<double> col, ab;
                for (const auto& r : raw) { col.push_back(r[i]); ab.push_back(std::abs(r[i])); }
                obs_csv << name << "," << i << "," << *std::min_element(col.begin(), col.end()) << ","
                        << *std::max_element(col.begin(), col.end()) << "," << quantile(col, 0.01) << ","
                        << quantile(col, 0.99) << "," << quantile(ab, 0.99) << "," << sat[i] << "\n";
            }
        }
    }
    return 0;
}
