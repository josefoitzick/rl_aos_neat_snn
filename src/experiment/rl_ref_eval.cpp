// Referencia RL (docs/diseno_neat_ap.md §15.4, paso 3): evalúa una política exportada por el zoo de RLtools
// en NUESTRO entorno (envs::makeEnv) y con los MISMOS episodios finales que NEAT: para cada semilla de NEAT s,
// los episodios eval::episodeSeeds(s, 100, 2) con test_split, como hace neat_control al final de la corrida.
//
// La política se carga con dlopen desde una biblioteca compilada con scripts/compilar_politica_rl.sh
// (src/experiment/rl_policy_shim.cpp). Recibe la observación CRUDA (rawObserve, la que vio en el
// entrenamiento) y devuelve acciones en [-1, 1].
//
// Controles:
//   --policy zero | random  → política nula o aleatoria (control negativo: reproduce el trivial del probe);
//   al cargar una política se exige que reproduzca los ejemplos del checkpoint (control positivo).
//
// Salida: una línea por (política, semilla NEAT) en la terminal y, con --out, un CSV.
#include <CLI/CLI.hpp>
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "envs/control_env.hpp"
#include "eval/evaluator.hpp"

namespace {

struct Loaded {
    std::string path;
    void* handle = nullptr;
    int (*obs_dim)() = nullptr;
    int (*act_dim)() = nullptr;
    void (*reset)() = nullptr;
    void (*act)(const double*, double*) = nullptr;
    double (*selftest)() = nullptr;
};

Loaded load(const std::string& path) {
    Loaded p;
    p.path = path;
    p.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!p.handle) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    auto sym = [&](const char* n) {
        void* s = dlsym(p.handle, n);
        if (!s) throw std::runtime_error(path + ": falta el símbolo " + n);
        return s;
    };
    p.obs_dim = reinterpret_cast<int (*)()>(sym("rlp_obs_dim"));
    p.act_dim = reinterpret_cast<int (*)()>(sym("rlp_act_dim"));
    p.reset = reinterpret_cast<void (*)()>(sym("rlp_reset"));
    p.act = reinterpret_cast<void (*)(const double*, double*)>(sym("rlp_act"));
    p.selftest = reinterpret_cast<double (*)()>(sym("rlp_selftest"));
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Evalúa una política RL del zoo de RLtools con los episodios finales de NEAT"};
    std::string env_name, out_csv;
    std::vector<std::string> policies;
    int first_seed = 1, last_seed = 5, episodes = 100;
    double torque_scale = 1.0;
    app.add_option("--env", env_name, "Entorno (envs::envNames)")->required();
    app.add_option("--policy", policies, "Bibliotecas .so de política, o 'zero' / 'random'")->required();
    app.add_option("--first-seed", first_seed, "Primera semilla de NEAT cuyos episodios finales se usan");
    app.add_option("--last-seed", last_seed, "Última semilla de NEAT");
    app.add_option("--episodes", episodes, "Episodios por semilla (NEAT usa 100)");
    app.add_option("--torque-scale", torque_scale, "Igual que en neat_control (pendulum*, acrobot*)");
    app.add_option("--out", out_csv, "CSV de salida (se agrega al final si existe)");
    CLI11_PARSE(app, argc, argv);

    envs::EnvOptions opt;
    opt.torque_scale = torque_scale;
    opt.test_split = true;  // como la evaluación final de neat_control
    auto env = envs::makeEnv(env_name, opt);
    const int agents = env->numAgents(), od = env->obsDim(), ad = env->actionDim();

    std::ofstream csv;
    if (!out_csv.empty()) {
        const bool exists = std::ifstream(out_csv).good();
        csv.open(out_csv, std::ios::app);
        if (!exists) csv << "env,torque_scale,policy,neat_seed,episodes,mean,sd,mean_len\n";
    }

    int failures = 0;
    for (const std::string& pol : policies) {
        const bool zero = pol == "zero", random = pol == "random";
        Loaded p;
        if (!zero && !random) {
            p = load(pol);
            if (p.obs_dim() != od || p.act_dim() != ad) {
                std::fprintf(stderr, "ERROR %s: dimensiones %d→%d, el entorno %s tiene %d→%d por agente\n",
                             pol.c_str(), p.obs_dim(), p.act_dim(), env_name.c_str(), od, ad);
                failures++;
                continue;
            }
            const double d = p.selftest();
            if (!(d < 1e-5)) {
                std::fprintf(stderr, "ERROR %s: no reproduce los ejemplos del checkpoint (error %.3g)\n", pol.c_str(), d);
                failures++;
                continue;
            }
        }
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        for (int s = first_seed; s <= last_seed; s++) {
            const auto seeds = eval::episodeSeeds(static_cast<uint64_t>(s), episodes, 2);
            double sum = 0, sum2 = 0, len = 0;
            for (uint64_t es : seeds) {
                env->reset(es);
                if (p.reset) p.reset();
                double ret = 0;
                int t = 0;
                std::vector<double> a(static_cast<size_t>(agents * ad), 0.0);
                for (; t < env->maxSteps(); t++) {
                    if (random) {
                        for (double& x : a) x = U(rng);
                    } else if (!zero) {
                        const std::vector<double> o = env->rawObserve();
                        for (int k = 0; k < agents; k++) p.act(o.data() + k * od, a.data() + k * ad);
                    }
                    const auto r = env->step(a);
                    ret += r.reward;
                    if (r.terminated) { t++; break; }
                }
                sum += ret;
                sum2 += ret * ret;
                len += t;
            }
            const double n = episodes, mean = sum / n, sd = std::sqrt(std::max(0.0, sum2 / n - mean * mean));
            std::printf("%-20s %-60s semilla %d: %10.4f ± %-9.4f largo %.1f\n", env_name.c_str(), pol.c_str(), s,
                        mean, sd, len / n);
            if (csv) csv << env_name << ',' << torque_scale << ',' << pol << ',' << s << ',' << episodes << ',' << mean << ',' << sd << ','
                         << len / n << '\n';
        }
    }
    return failures ? 1 : 0;
}
