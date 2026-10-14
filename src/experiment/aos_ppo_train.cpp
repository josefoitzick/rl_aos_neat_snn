// Paso 3 del agente PPO (docs/diseno_ppo.md §3–4): entrenamiento de PPO (RLtools) sobre NEAT real.
//
// El lote propio (src/experiment/ppo_lote.hpp) conecta el runner de RLtools con NeatPool (src/aos/neat_run.hpp):
// 16 corridas de NEAT avanzan juntas, una decisión de P cada k generaciones. Configuración de PPO del paso 1
// (§5.1): γ = 0,99, λ = 1, 20 épocas, α 3e-4 (actor) y 1e-3 (crítico), última capa del actor × 0,01 (arranque en P₀).
//
// Opciones de NEAT: las mismas de neat_control (src/experiment/opciones_neat.hpp); --p es P₀ (la P estática).
// K (operadores del catálogo) se fija al compilar: 3–10 y 15.
//
// Salida en --out:
//   config.txt             línea de comando, binario y parámetros
//   ppo.csv                una fila por iteración
//   corridas.csv           una fila por corrida terminada (test final, score, retorno, mejor validación)
//   decisiones.csv         una fila por decisión (Φ, recompensa, P y observación)
//   estado.tar, estado.txt checkpoint para --reanudar (se guarda cuando todas las corridas recién empiezan)
//   actor_final/checkpoint.h (y actor_iterNNNN/ con --exportar-cada): el actor como código, para el despliegue
//   validacion.txt         con --validar N: la política determinista frente a P₀ en N semillas nuevas (pareado)
//
// RLtools va envuelto en el espacio de nombres ppo_rlt: libenvs también lo incluye (entornos de control) y algunas
// de sus funciones no son inline, así que sin el envoltorio el enlazado choca (definición múltiple).
#define RL_TOOLS_NAMESPACE_WRAPPER ppo_rlt
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include "experiment/ppo_lote.hpp"
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/random/operations_generic_array.h>
// Persistencia: estado de entrenamiento en tar y actor como código (despliegue)
#include <rl_tools/persist/backends/tar/operations_cpu.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn/optimizers/adam/persist.h>
#include <rl_tools/nn/optimizers/adam/instance/persist.h>
#include <rl_tools/rl/algorithms/ppo/persist.h>
#include <rl_tools/random/persist.h>
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

#include <CLI/CLI.hpp>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "aos/action_map.hpp"
#include "aos/factory.hpp"
#include "aos/neat_run.hpp"
#include "experiment/opciones_neat.hpp"
#include "neat/selector.hpp"

namespace rlt = ppo_rlt::rl_tools;

namespace {

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
using T = float;
using TI = typename DEVICE::index_t;
using RNG = rlt::devices::generic::random::ArrayENGINE<rlt::devices::generic::random::ArraySpecification<TI, 1024, true>>;
constexpr TI N_CORRIDAS = 16;      // corridas de NEAT en paralelo
constexpr TI PASOS_POR_ITER = 25;  // decisiones por corrida en cada iteración de PPO

// J_trivial y J_RL de ~/tesis/runs/ref_rl/score_v2.json (los del score de la tesis)
const std::map<std::string, std::pair<double, double>> kReferencias = {
    {"pendulum", {-1251.526, -152.99458}},
    {"acrobot-swingup", {-18.03048, -6.65627}},
    {"bottleneck", {0.0, 61.743476}},
    {"l2f", {4.27954, 30.54559}},
};

template <TI K>
struct Tipos {
    static constexpr TI N = N_CORRIDAS;
    static constexpr TI OBS = 14 + 2 * K;
    using ENV = ppo_rlt::ppo_lote::Entorno<T, TI, OBS, K>;
    struct PARAMS : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENV> {
        static constexpr TI N_ENVIRONMENTS = N;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = PASOS_POR_ITER;
        static constexpr TI BATCH_SIZE = 100;
        static constexpr TI ACTOR_HIDDEN_DIM = 64;
        static constexpr TI ACTOR_NUM_LAYERS = 3;
        static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
        static constexpr TI CRITIC_HIDDEN_DIM = 128;
        static constexpr TI CRITIC_NUM_LAYERS = 3;
        static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
        static constexpr bool NORMALIZE_OBSERVATIONS = false;   // estandarización propia (primeras iteraciones)
        using ACTOR_OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;
        using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;
        struct PPO_PARAMETERS : rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE> {
            static constexpr T GAMMA = 0.99f;
            static constexpr T LAMBDA = 1.0f;
            static constexpr TI N_EPOCHS = 20;
        };
    };
    using BASE = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENV, PARAMS,
                                                               rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;
    using LOTE = ppo_rlt::ppo_lote::Lote<ppo_rlt::ppo_lote::LoteSpec<ENV, N, aos::NeatPool>>;
    using CONFIG = ppo_rlt::ppo_lote::ConfigConLote<BASE, LOTE>;
    using STATE = rlt::rl::algorithms::ppo::loop::core::State<CONFIG>;
};

struct Opciones {
    std::string out;
    int T = 50, reward_episodes = 50;
    double ref_trivial = std::numeric_limits<double>::quiet_NaN(), ref_rl = std::numeric_limits<double>::quiet_NaN();
    std::string p_txt;              // P₀ (como --p de neat_control)
    int corridas = 400;             // corridas de entrenamiento (completas)
    uint64_t semilla = 1;           // PPO: inicialización y muestreo de acciones
    uint64_t semilla_corridas = 1001;
    double c = 3.0, w = 0.5, escala = 10.0, std0 = 0.5, lr = 3e-4, lr_critico = 1e-3, ultima_capa = 0.01;
    int iter_normalizar = 2;        // iteraciones iniciales que acumulan la estandarización de observaciones
    int validar = 0;                // corridas de validación al final (múltiplo de 16), semillas 2001+
    uint64_t semilla_validacion = 2001;
    int exportar_cada = 0;
    bool reanudar = false;
    int max_iter = 0;               // pruebas: detenerse tras esta iteración (simula una interrupción)
    bool sin_test_final = false;    // no medir el test final de cada corrida de entrenamiento (más rápido)
};

std::vector<double> parseP(const std::string& txt, size_t K) {
    std::vector<double> w;
    std::stringstream ss(txt);
    std::string tok;
    while (std::getline(ss, tok, ',')) w.push_back(std::stod(tok));
    if (w.size() != K) throw std::invalid_argument("--p: se esperan " + std::to_string(K) + " valores separados por coma");
    for (double x : w)
        if (x < 0.0) throw std::invalid_argument("--p: valores negativos");
    return neat::StaticSelector(w).probabilities();
}

std::string join(const std::vector<double>& v, const char* fmt = "%.6g") {
    std::string s;
    char buf[48];
    for (size_t i = 0; i < v.size(); i++) {
        std::snprintf(buf, sizeof buf, fmt, v[i]);
        s += (i ? "," : "") + std::string(buf);
    }
    return s;
}

double media(const std::vector<double>& x) { return x.empty() ? NAN : std::accumulate(x.begin(), x.end(), 0.0) / x.size(); }
double desv(const std::vector<double>& x) {
    if (x.size() < 2) return NAN;
    const double m = media(x);
    double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return std::sqrt(s / (x.size() - 1));
}

std::string sha16(const std::string& path) {
    std::string cmd = "sha256sum '" + path + "' 2>/dev/null | cut -c1-16";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return "?";
    char buf[64] = {0};
    if (!fgets(buf, sizeof buf, p)) buf[0] = 0;
    pclose(p);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s.empty() ? "?" : s;
}

// --- estado de entrenamiento (tar) ---
template <typename STATE>
void guardarEstado(DEVICE& device, STATE& ts, const std::string& path) {
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> root;
    root.writer = &writer;
    auto g_ao = rlt::create_group(device, root, "actor_optimizer");
    rlt::save(device, ts.actor_optimizer, g_ao);
    auto g_co = rlt::create_group(device, root, "critic_optimizer");
    rlt::save(device, ts.critic_optimizer, g_co);
    auto g_ppo = rlt::create_group(device, root, "ppo");
    rlt::save(device, ts.ppo, g_ppo);
    auto g_rng = rlt::create_group(device, root, "rng");
    rlt::save(device, ts.rng, g_rng);
    rlt::persist::backends::tar::finalize(device, writer);
    if (!root.success || !g_ao.success || !g_co.success || !g_ppo.success || !g_rng.success)
        throw std::runtime_error("no se pudo serializar el estado de entrenamiento");
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(writer.buffer.data(), writer.buffer.size());
        if (!f) throw std::runtime_error("no se pudo escribir " + tmp);
    }
    std::filesystem::rename(tmp, path);   // atómico: un checkpoint a medias nunca reemplaza al anterior
}

template <typename STATE>
void cargarEstado(DEVICE& device, STATE& ts, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("no se pudo leer " + path);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> root;
    root.data = {buf.data(), static_cast<TI>(buf.size())};
    bool ok = true;
    auto g_ao = rlt::get_group(device, root, "actor_optimizer");
    ok &= rlt::load(device, ts.actor_optimizer, g_ao);
    auto g_co = rlt::get_group(device, root, "critic_optimizer");
    ok &= rlt::load(device, ts.critic_optimizer, g_co);
    auto g_ppo = rlt::get_group(device, root, "ppo");
    ok &= rlt::load(device, ts.ppo, g_ppo);
    auto g_rng = rlt::get_group(device, root, "rng");
    ok &= rlt::load(device, ts.rng, g_rng);
    if (!ok) throw std::runtime_error("no se pudo cargar el estado de entrenamiento de " + path);
}

// --- exportación del actor como código (despliegue, paso 4) ---
template <TI K>
void exportarActor(DEVICE& device, typename Tipos<K>::STATE& ts, const std::string& dir, const aos::NeatRunConfig& cfg,
                   const std::vector<double>& P0, const Opciones& o, int iter) {
    using ACTOR = typename Tipos<K>::CONFIG::NN::ACTOR_TYPE;
    using FWD = typename ACTOR::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    FWD actor;
    typename FWD::template Buffer<true> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename FWD::INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename FWD::OUTPUT_SHAPE>> output;
    RNG rng;
    rlt::malloc(device, actor);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 12345);
    rlt::copy(device, device, ts.ppo.actor, actor);
    rlt::randn(device, input, rng);
    rlt::evaluate(device, actor, input, output, buffer, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
    std::stringstream ss;
    ss << rlt::save_code(device, actor, std::string("rl_tools::checkpoint::actor"), true);
    ss << "\nnamespace rl_tools::checkpoint::example::inputs{\n" << rlt::save_code(device, input, std::string("_0"), true) << "\n}";
    ss << "\nnamespace rl_tools::checkpoint::example::outputs{\n" << rlt::save_code(device, output, std::string("_0"), true) << "\n}";
    // Metadatos que el despliegue verifica contra la corrida (docs/diseno_ppo.md §3.4)
    ss << "\nnamespace aos_ppo::meta{";
    ss << "\n    constexpr int K = " << K << ";";
    ss << "\n    constexpr int OBS = " << Tipos<K>::OBS << ";";
    ss << "\n    constexpr int T = " << cfg.T << ";";
    ss << "\n    constexpr int version_rasgos = 1;";
    ss << "\n    constexpr double c = " << std::hexfloat << o.c << ";";
    ss << "\n    constexpr double P0[] = {";
    for (int k = 0; k < static_cast<int>(K); k++) ss << (k ? ", " : "") << std::hexfloat << P0[k];
    ss << "};";
    ss << "\n    constexpr double ref_trivial = " << std::hexfloat << cfg.ref_trivial << ";";
    ss << "\n    constexpr double ref_rl = " << std::hexfloat << cfg.ref_rl << ";" << std::defaultfloat;
    ss << "\n    constexpr const char* catalogo = \"";
    for (size_t k = 0; k < cfg.ncfg.catalog.size(); k++) ss << (k ? "," : "") << neat::operatorName(cfg.ncfg.catalog[k]);
    ss << "\";";
    ss << "\n    constexpr const char* entorno = \"" << cfg.env_name << "\";";
    ss << "\n    constexpr int iteracion = " << iter << ";";
    ss << "\n}\n";
    std::filesystem::create_directories(dir);
    std::ofstream f(dir + "/checkpoint.h");
    f << ss.str();
    rlt::free(device, actor);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, rng);
}

// --- política determinista sobre NeatPool (validación) ---
struct Eval {
    std::vector<double> final_test, score;
};

template <TI K>
Eval evaluarActor(DEVICE& device, typename Tipos<K>::STATE& ts, const aos::NeatRunConfig& cfg, uint64_t seed_base, int n, double w,
                  double escala) {
    using TP = Tipos<K>;
    constexpr TI N = TP::N, D = TP::OBS;
    aos::NeatPool pool(cfg, N, seed_base, w, escala, true);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, D>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, K>>> output;
    typename TP::CONFIG::NN::ACTOR_TYPE::template State<true> pstate;
    RNG rng;
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, pstate);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    rlt::reset(device, ts.ppo.actor, pstate, rng);
    std::vector<int> all(N);
    std::iota(all.begin(), all.end(), 0);
    Eval e;
    std::vector<std::vector<double>> P(N);
    while (static_cast<int>(e.final_test.size()) < n) {
        pool.start(all);
        for (int t = 0; t < cfg.T; t++) {
            for (TI i = 0; i < N; i++)
                for (TI j = 0; j < D; j++) rlt::set(device, input, pool.obs(i)[j], i, j);
            rlt::evaluate_step(device, ts.ppo.actor, input, pstate, output, ts.actor_eval_buffers, rng, rlt::Mode<rlt::mode::Rollout<>>{});
            for (TI i = 0; i < N; i++) {
                double a[K];
                for (TI j = 0; j < K; j++) a[j] = rlt::get(device, output, i, j);
                P[i] = aos::actionToP(a, ts.environment.logp0, ts.environment.c);
            }
            pool.advance(P);
        }
        for (const auto& f : pool.takeFinished()) {
            e.final_test.push_back(f.final_test);
            e.score.push_back(f.final_score);
        }
    }
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, pstate);
    rlt::free(device, rng);
    return e;
}

Eval evaluarFija(const std::vector<double>& P, const aos::NeatRunConfig& cfg, uint64_t seed_base, int n, int N, double w, double escala) {
    aos::NeatPool pool(cfg, N, seed_base, w, escala, true);
    std::vector<int> all(N);
    std::iota(all.begin(), all.end(), 0);
    Eval e;
    while (static_cast<int>(e.final_test.size()) < n) {
        pool.start(all);
        for (int t = 0; t < cfg.T; t++) pool.advance(std::vector<std::vector<double>>(N, P));
        for (const auto& f : pool.takeFinished()) {
            e.final_test.push_back(f.final_test);
            e.score.push_back(f.final_score);
        }
    }
    return e;
}

template <TI K>
double klAprox(DEVICE& device, typename Tipos<K>::STATE& ts) {
    using TP = Tipos<K>;
    constexpr TI N = TP::N, D = TP::OBS;
    auto& ds = ts.on_policy_runner_dataset;
    constexpr TI STEPS_TOTAL = TP::CONFIG::ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, D>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, K>>> output;
    typename TP::CONFIG::NN::ACTOR_TYPE::template State<true> pstate;
    RNG rng;
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, pstate);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    rlt::reset(device, ts.ppo.actor, pstate, rng);
    auto& last = rlt::get_last_layer(ts.ppo.actor);
    double kl = 0.0;
    for (TI b = 0; b < STEPS_TOTAL; b += N) {
        for (TI r = 0; r < N; r++)
            for (TI j = 0; j < D; j++) rlt::set(device, input, rlt::get(device, ds.all_observations, b + r, j), r, j);
        rlt::evaluate_step(device, ts.ppo.actor, input, pstate, output, ts.actor_eval_buffers, rng, rlt::Mode<rlt::mode::Rollout<>>{});
        for (TI r = 0; r < N; r++) {
            double lp = 0.0;
            for (TI j = 0; j < K; j++) {
                const double ls = rlt::get(device, last.log_std.parameters, j);
                const double z = (rlt::get(ds.actions, b + r, j) - rlt::get(device, output, r, j)) / std::exp(ls);
                lp += -0.5 * z * z - ls - 0.5 * std::log(2 * M_PI);
            }
            kl += rlt::get(ds.action_log_probs, b + r, 0) - lp;
        }
    }
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, pstate);
    rlt::free(device, rng);
    return kl / STEPS_TOTAL;
}

// Estandarización de observaciones (actor y crítico) con el lote recién recolectado, como el modo de acumulación
// del núcleo de RLtools pero sin su colecta extra (docs/diseno_ppo.md §2.3)
template <TI K>
void acumularEstandarizacion(DEVICE& device, typename Tipos<K>::STATE& ts) {
    using TP = Tipos<K>;
    using PARAMS = typename TP::PARAMS;
    constexpr TI BATCH = PARAMS::BATCH_SIZE;
    constexpr TI STEPS_TOTAL = TP::CONFIG::ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL;
    constexpr TI N_BATCHES = STEPS_TOTAL / BATCH;
    using OBS_SHAPE = typename TP::ENV::Observation::SHAPE;
    using PRIV_SHAPE = typename TP::ENV::ObservationPrivileged::SHAPE;
    using ACTOR_INPUT = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, BATCH>, 1>;
    using CRITIC_INPUT = rlt::tensor::Prepend<rlt::tensor::Prepend<PRIV_SHAPE, BATCH>, 1>;
    rlt::Mode<rlt::nn::layers::standardize::AccumulateMode<>> modo;
    for (TI b = 0; b < N_BATCHES; b++) {
        auto obs = rlt::view_range(device, ts.on_policy_runner_dataset.all_observations, b * BATCH, rlt::tensor::ViewSpec<0, BATCH>{});
        auto obs_r = rlt::reshape_row_major(device, obs, ACTOR_INPUT{});
        rlt::forward(device, ts.ppo.actor, obs_r, ts.actor_buffers, ts.rng, modo);
        auto priv = rlt::view_range(device, ts.on_policy_runner_dataset.all_observations_privileged, b * BATCH, rlt::tensor::ViewSpec<0, BATCH>{});
        auto priv_r = rlt::reshape_row_major(device, priv, CRITIC_INPUT{});
        rlt::forward(device, ts.ppo.critic, priv_r, ts.critic_buffers, ts.rng, modo);
    }
}

double ahora() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <TI K>
int entrenar(const Opciones& o, const aos::NeatRunConfig& cfg, const std::vector<double>& P0, const std::string& linea, const std::string& sha) {
    using TP = Tipos<K>;
    using CONFIG = typename TP::CONFIG;
    constexpr TI N = TP::N;
    DEVICE device;
    std::filesystem::create_directories(o.out);
    const std::string f_estado_txt = o.out + "/estado.txt", f_estado_tar = o.out + "/estado.tar";

    // Reanudar: iteración, corridas y la semilla con que empezaron las corridas en curso
    int iter = 0;
    long corridas = 0;
    uint64_t seed_base = o.semilla_corridas;
    if (o.reanudar) {
        std::ifstream f(f_estado_txt);
        if (!f) throw std::runtime_error("--reanudar: falta " + f_estado_txt);
        std::map<std::string, std::string> kv;
        std::string l;
        while (std::getline(f, l)) {
            const auto p = l.find('=');
            if (p != std::string::npos) kv[l.substr(0, p)] = l.substr(p + 1);
        }
        if (std::stoi(kv.at("K")) != static_cast<int>(K) || std::stoi(kv.at("OBS")) != static_cast<int>(TP::OBS))
            throw std::runtime_error("--reanudar: el checkpoint es de otro catálogo");
        if (kv.at("binario") != sha) std::printf("aviso: el checkpoint se guardó con otro binario (%s)\n", kv.at("binario").c_str());
        iter = std::stoi(kv.at("iteracion"));
        corridas = std::stol(kv.at("corridas"));
        seed_base = std::stoull(kv.at("semilla_en_curso"));
        std::printf("reanudando: iteración %d, %ld corridas, corridas en curso desde la semilla %llu\n", iter, corridas,
                    static_cast<unsigned long long>(seed_base));
    }

    aos::NeatPool pool(cfg, N, seed_base, o.w, o.escala, !o.sin_test_final);
    pool.enableDecisionLog(o.out + "/decisiones.csv", o.reanudar);
    typename TP::STATE ts;
    rlt::malloc(device, ts);
    ts.environment.pool = &pool;
    ts.environment.logp0 = aos::logP0(P0);
    ts.environment.c = o.c;
    const double t0 = ahora();
    rlt::init(device, ts, o.semilla);   // inicia las N corridas (generación 0 evaluada)
    if (o.reanudar) {
        cargarEstado(device, ts, f_estado_tar);
    } else {
        // Arranque en P₀: última capa del actor 100× menor (Andrychowicz et al. 2021) y σ inicial
        auto& last = rlt::get_last_layer(ts.ppo.actor);
        rlt::scale(device, last.output_layer.weights.parameters, static_cast<T>(o.ultima_capa));
        rlt::scale(device, last.output_layer.biases.parameters, static_cast<T>(o.ultima_capa));
        rlt::set_all(device, last.log_std.parameters, static_cast<T>(std::log(o.std0)));
    }
    rlt::get_ref(device, ts.actor_optimizer.parameters, 0).alpha = static_cast<T>(o.lr);
    rlt::get_ref(device, ts.critic_optimizer.parameters, 0).alpha = static_cast<T>(o.lr_critico);

    std::ofstream ppo_csv(o.out + "/ppo.csv", o.reanudar ? std::ios::app : std::ios::trunc);
    std::ofstream corr_csv(o.out + "/corridas.csv", o.reanudar ? std::ios::app : std::ios::trunc);
    if (!o.reanudar) {
        ppo_csv << "iteracion,corridas,segundos,score_medio,retorno_medio,mejor_validacion_media,log_std,kl";
        for (TI k = 0; k < K; k++) ppo_csv << ",p" << k;
        ppo_csv << "\n";
        corr_csv << "iteracion,semilla,test_final,score,retorno,mejor_validacion\n";
        std::ofstream c(o.out + "/config.txt");
        c << "linea=" << linea << "\nbinario=" << sha << "\nK=" << K << "\nOBS=" << TP::OBS << "\nN=" << N
          << "\npasos_por_iteracion=" << PASOS_POR_ITER << "\nT=" << cfg.T << "\ngeneraciones=" << cfg.generations
          << "\npoblacion=" << cfg.ncfg.population_size << "\nP0=" << join(P0) << "\nref_trivial=" << cfg.ref_trivial
          << "\nref_rl=" << cfg.ref_rl << "\nc=" << o.c << "\nw=" << o.w << "\nescala=" << o.escala << "\nstd0=" << o.std0
          << "\nlr=" << o.lr << "\nlr_critico=" << o.lr_critico << "\ngamma=0.99\nlambda=1\nepocas=20\nhilos=" << omp_get_max_threads() << "\n";
    }
    std::printf("PPO sobre NEAT: entorno %s, K=%d, OBS=%d, %d corridas en paralelo, T=%d, G=%d, presupuesto %d corridas, hilos %d\n",
                cfg.env_name.c_str(), static_cast<int>(K), static_cast<int>(TP::OBS), static_cast<int>(N), cfg.T, cfg.generations,
                o.corridas, omp_get_max_threads());
    std::printf("P0 = (%s)  refs %.4g → %.4g\n", join(P0, "%.4f").c_str(), cfg.ref_trivial, cfg.ref_rl);
    std::fflush(stdout);

    while (corridas < o.corridas) {
        const double ti = ahora();
        rlt::collect(device, ts.on_policy_runner_dataset, ts.on_policy_runner, ts.on_policy_runner_buffer, ts.environment, ts.ppo.actor,
                     ts.actor_eval_buffers, ts.ppo.critic, ts.critic_collection_state, ts.critic_collection_buffer, ts.rng,
                     typename decltype(ts.ppo)::SPEC::COLLECTION_MODE{});
        if (iter < o.iter_normalizar) acumularEstandarizacion<K>(device, ts);
        rlt::estimate_generalized_advantages(device, ts.on_policy_runner_dataset, ts.on_policy_runner_dataset.bootstrap_values,
                                             typename CONFIG::PPO_TYPE::SPEC::PARAMETERS{});
        rlt::train(device, ts.ppo, ts.on_policy_runner_dataset, ts.actor_optimizer, ts.critic_optimizer, ts.ppo_buffers, ts.actor_buffers,
                   ts.critic_buffers, ts.rng);
        iter++;
        const double kl = klAprox<K>(device, ts);
        auto& last = rlt::get_last_layer(ts.ppo.actor);
        double ls = 0.0;
        for (TI j = 0; j < K; j++) ls += rlt::get(device, last.log_std.parameters, j);
        ls /= K;
        const auto fin = pool.takeFinished();
        const auto pm = pool.takeMeanP();
        corridas += static_cast<long>(fin.size());
        std::vector<double> sc, re, bv;
        for (const auto& f : fin) {
            sc.push_back(f.final_score);
            re.push_back(f.ret);
            bv.push_back(f.best_validation);
            corr_csv << iter << "," << f.seed << "," << f.final_test << "," << f.final_score << "," << f.ret << "," << f.best_validation << "\n";
        }
        corr_csv.flush();
        const double secs = ahora() - ti;
        ppo_csv << iter << "," << corridas << "," << secs << "," << media(sc) << "," << media(re) << "," << media(bv) << "," << ls << ","
                << kl;
        for (double p : pm) ppo_csv << "," << p;
        ppo_csv << "\n";
        ppo_csv.flush();
        std::printf("iter %3d  corridas %4ld  %6.0f s  score %s  retorno %s  log σ %.3f  KL %.5f  P=(%s)\n", iter, corridas, secs,
                    fin.empty() ? "   —  " : (std::to_string(media(sc)).substr(0, 6)).c_str(),
                    fin.empty() ? "   —  " : (std::to_string(media(re)).substr(0, 6)).c_str(), ls, kl, join(pm, "%.3f").c_str());
        std::fflush(stdout);

        // Checkpoint cuando todas las corridas recién empiezan (no hay que guardar NEAT a medias)
        bool todas_inician = iter >= o.iter_normalizar;
        uint64_t semilla_min = std::numeric_limits<uint64_t>::max();
        for (TI s = 0; s < N; s++) {
            todas_inician &= pool.t(static_cast<int>(s)) == 0;
            semilla_min = std::min(semilla_min, pool.run(static_cast<int>(s)).seed());
        }
        if (todas_inician) {
            guardarEstado(device, ts, f_estado_tar);
            std::ofstream e(f_estado_txt + ".tmp");
            e << "iteracion=" << iter << "\ncorridas=" << corridas << "\nsemilla_en_curso=" << semilla_min << "\nK=" << K
              << "\nOBS=" << TP::OBS << "\nbinario=" << sha << "\n";
            e.close();
            std::filesystem::rename(f_estado_txt + ".tmp", f_estado_txt);
        }
        if (o.exportar_cada > 0 && iter % o.exportar_cada == 0) {
            char d[32];
            std::snprintf(d, sizeof d, "/actor_iter%04d", iter);
            exportarActor<K>(device, ts, o.out + d, cfg, P0, o, iter);
        }
        if (o.max_iter > 0 && iter >= o.max_iter) {
            std::printf("detenido en la iteración %d (--max-iter)\n", iter);
            rlt::free(device, ts);
            return 0;
        }
    }
    exportarActor<K>(device, ts, o.out + "/actor_final", cfg, P0, o, iter);
    std::printf("entrenamiento terminado: %d iteraciones, %ld corridas, %.1f h\n", iter, corridas, (ahora() - t0) / 3600.0);

    if (o.validar > 0) {
        if (o.validar % static_cast<int>(N) != 0) throw std::invalid_argument("--validar debe ser múltiplo de 16");
        const Eval ppo = evaluarActor<K>(device, ts, cfg, o.semilla_validacion, o.validar, o.w, o.escala);
        const Eval base = evaluarFija(std::vector<double>(P0), cfg, o.semilla_validacion, o.validar, static_cast<int>(N), o.w, o.escala);
        std::vector<double> d(ppo.score.size());
        int gana = 0;
        for (size_t i = 0; i < d.size(); i++) {
            d[i] = ppo.score[i] - base.score[i];
            gana += d[i] > 0;
        }
        std::ofstream v(o.out + "/validacion.txt");
        v << "semillas=" << o.semilla_validacion << "-" << o.semilla_validacion + o.validar - 1 << "\nscore_ppo=" << media(ppo.score)
          << "\nscore_p0=" << media(base.score) << "\ndelta=" << media(d) << "\nee_delta=" << desv(d) / std::sqrt(d.size())
          << "\nppo_gana=" << gana << "/" << d.size() << "\ntest_final_ppo=" << join(ppo.final_test) << "\ntest_final_p0="
          << join(base.final_test) << "\n";
        std::printf("validación (%d semillas): PPO %.4f, P0 %.4f, Δ %+.4f (ee %.4f), PPO gana en %d\n", o.validar, media(ppo.score),
                    media(base.score), media(d), desv(d) / std::sqrt(d.size()), gana);
    }
    rlt::free(device, ts);
    return 0;
}

template <TI K>
int lanzar(const Opciones& o, const aos::NeatRunConfig& cfg, const std::vector<double>& P0, const std::string& l, const std::string& sha) {
    return entrenar<K>(o, cfg, P0, l, sha);
}

int despachar(size_t K, const Opciones& o, const aos::NeatRunConfig& cfg, const std::vector<double>& P0, const std::string& l,
              const std::string& sha) {
    switch (K) {
        case 3: return lanzar<3>(o, cfg, P0, l, sha);
        case 4: return lanzar<4>(o, cfg, P0, l, sha);
        case 5: return lanzar<5>(o, cfg, P0, l, sha);
        case 6: return lanzar<6>(o, cfg, P0, l, sha);
        case 7: return lanzar<7>(o, cfg, P0, l, sha);
        case 8: return lanzar<8>(o, cfg, P0, l, sha);
        case 9: return lanzar<9>(o, cfg, P0, l, sha);
        case 10: return lanzar<10>(o, cfg, P0, l, sha);
        case 15: return lanzar<15>(o, cfg, P0, l, sha);
        default: throw std::invalid_argument("catálogo de " + std::to_string(K) + " operadores: agregar ese K en despachar()");
    }
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Entrenamiento de PPO (RLtools) sobre NEAT (paso 3 del agente, docs/diseno_ppo.md)"};
    opciones_neat::Opciones on;
    opciones_neat::agregar(app, on);
    Opciones o;
    std::string selector = "static";
    int print_every = 0;
    app.add_option("--out", o.out, "Directorio de salida")->required();
    app.add_option("--p", o.p_txt, "P₀: P estática del catálogo (como --p de neat_control)");
    app.add_option("--T", o.T, "Decisiones por corrida");
    app.add_option("--corridas", o.corridas, "Corridas de entrenamiento");
    app.add_option("--semilla", o.semilla, "Semilla de PPO");
    app.add_option("--semilla-corridas", o.semilla_corridas, "Primera semilla de las corridas de entrenamiento");
    app.add_option("--episodios-recompensa", o.reward_episodes, "Episodios del conjunto de recompensa (stream 3)");
    app.add_option("--ref-trivial", o.ref_trivial, "J_trivial (por defecto, el de score_v2 para el entorno)");
    app.add_option("--ref-rl", o.ref_rl, "J_RL (por defecto, el de score_v2 para el entorno)");
    app.add_option("--c", o.c, "Cota de la desviación de cada logit");
    app.add_option("--w", o.w, "Peso del área bajo la curva en la recompensa");
    app.add_option("--escala", o.escala, "Escala de la recompensa");
    app.add_option("--std0", o.std0, "Desviación estándar inicial de la acción");
    app.add_option("--lr", o.lr, "Tasa de aprendizaje del actor");
    app.add_option("--lr-critico", o.lr_critico, "Tasa de aprendizaje del crítico");
    app.add_option("--iter-normalizar", o.iter_normalizar, "Iteraciones iniciales que estiman la estandarización de observaciones");
    app.add_option("--validar", o.validar, "Corridas de validación al final (múltiplo de 16)");
    app.add_option("--semilla-validacion", o.semilla_validacion, "Primera semilla de validación");
    app.add_option("--exportar-cada", o.exportar_cada, "Iteraciones entre exportaciones del actor (0: solo al final)");
    app.add_flag("--reanudar", o.reanudar, "Continuar desde estado.tar / estado.txt de --out");
    app.add_option("--max-iter", o.max_iter, "Pruebas: detenerse tras esta iteración");
    app.add_flag("--sin-test-final", o.sin_test_final, "No medir el test final de las corridas de entrenamiento");
    app.add_option("--selector", selector)->check(CLI::IsMember({"static"}));   // aceptadas para reusar líneas de neat_control
    app.add_option("--print-every", print_every);
    CLI11_PARSE(app, argc, argv);
    std::string linea;
    for (int i = 0; i < argc; i++) linea += (i ? " " : "") + std::string(argv[i]);
    try {
        opciones_neat::finalizar(on);
        if (!on.resample_train) throw std::invalid_argument("el entrenamiento usa rotación de episodios: falta --resample-train");
        const size_t K = on.ncfg.catalog.size();
        std::vector<double> P0;
        if (o.p_txt.empty()) {
            if (on.ncfg.catalog != neat::defaultCatalog()) throw std::invalid_argument("--catalog distinto del de Stanley: dar --p (P₀)");
            aos::AdaptivePursuitConfig apc;
            P0 = aos::makeSelector("static", apc, false, on.ncfg.catalog)->probabilities();
        } else {
            P0 = neat::StaticSelector(parseP(o.p_txt, K)).probabilities();
        }
        aos::NeatRunConfig cfg;
        cfg.env_name = on.env_name;
        cfg.env_opt = on.env_opt;
        cfg.ecfg = on.ecfg;
        cfg.ncfg = on.ncfg;
        cfg.generations = on.generations;
        cfg.episodes = on.episodes;
        cfg.test_episodes = on.test_episodes;
        cfg.final_episodes = on.final_episodes;
        cfg.reward_episodes = o.reward_episodes;
        cfg.T = o.T;
        if (std::isnan(o.ref_trivial) || std::isnan(o.ref_rl)) {
            const auto it = kReferencias.find(on.env_name);
            if (it == kReferencias.end()) throw std::invalid_argument("entorno sin referencias conocidas: dar --ref-trivial y --ref-rl");
            cfg.ref_trivial = std::isnan(o.ref_trivial) ? it->second.first : o.ref_trivial;
            cfg.ref_rl = std::isnan(o.ref_rl) ? it->second.second : o.ref_rl;
        } else {
            cfg.ref_trivial = o.ref_trivial;
            cfg.ref_rl = o.ref_rl;
        }
        if (cfg.T % PASOS_POR_ITER != 0 && PASOS_POR_ITER % cfg.T != 0)
            std::printf("aviso: T no es múltiplo ni divisor de %d: los checkpoints quedarán espaciados\n", static_cast<int>(PASOS_POR_ITER));
        return despachar(K, o, cfg, P0, linea, sha16(std::filesystem::read_symlink("/proc/self/exe").string()));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
