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

// J_trivial y J_RL de ~/tesis/runs/ref_rl/score_v2_s1-10.json (los del score de la tesis y de Optuna)
const std::map<std::string, std::pair<double, double>> kReferencias = {
    {"pendulum", {-1231.444, -148.79341}},
    {"pendulum-meta", {-1231.444, -323.56979}},   // variante reactiva (sin --keep-state)
    {"acrobot-swingup", {-18.04437, -6.8019405}},
    {"bottleneck", {0.0, 61.8567}},
    {"l2f", {4.17605, 30.610236}},
};

// Tipos por dimensiones de la observación y de la acción (estado completo: 14 + 2K y K; plan B: estado compacto de 9 y
// acción con intensidades)
template <TI OBS_DIM, TI ACT_DIM>
struct Tipos {
    static constexpr TI N = N_CORRIDAS;
    static constexpr TI OBS = OBS_DIM;
    static constexpr TI ACT = ACT_DIM;
    using ENV = ppo_rlt::ppo_lote::Entorno<T, TI, OBS, ACT>;
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
    std::string mezcla;             // opciones de NEAT de un segundo entorno (mitad de las ranuras), para transferencia
    std::string evaluar;            // directorio con estado_final.tar (o estado.tar): evaluar la política congelada
    std::string semillas = "1-11";  // semillas de la evaluación (a-b)
    std::string politica_fija;      // evaluar con una P fija (verificación del modo de evaluación)
    std::string estado = "completo";  // completo (14 + 2K rasgos), compacto (9, plan B) o snn (10, con actividad de la SNN)
    std::string accion = "p";         // p, intensidades (plan B) o ambas
    std::string regla;                // archivo con una política escrita a mano: condiciones sobre el estado → empujones
    std::string regla_base;           // PPO residual: el agente aprende correcciones sobre esta regla (acción 0 = la regla)
};

// Política escrita a mano (docs/diseno_ppo.md §11): líneas "si <rasgo> <op> <valor> [y …] : <operador>=<empuje> …" o
// "siempre : …"; se aplica la primera que se cumple y el vector de empujones pasa por el mismo MapaAccion que la acción
// del agente (empuje 0 = P₀). Rasgos por índice de la observación; operadores por su nombre en el catálogo.
struct Regla {
    struct Cond { int rasgo; bool menor_igual; double valor; };
    std::vector<Cond> cond;
    std::vector<double> empuje;
};

std::vector<Regla> leerReglas(const std::string& path, const std::vector<neat::Operator>& catalogo, int dim_obs) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("--regla: no se pudo leer " + path);
    std::vector<Regla> reglas;
    std::string linea;
    while (std::getline(f, linea)) {
        const auto h = linea.find('#');
        if (h != std::string::npos) linea = linea.substr(0, h);
        const auto dos = linea.find(':');
        if (linea.find_first_not_of(" \t") == std::string::npos) continue;
        if (dos == std::string::npos) throw std::invalid_argument("--regla: falta ':' en «" + linea + "»");
        std::istringstream izq(linea.substr(0, dos)), der(linea.substr(dos + 1));
        Regla r;
        r.empuje.assign(catalogo.size(), 0.0);
        std::string tok;
        izq >> tok;
        if (tok == "si") {
            while (true) {
                Regla::Cond c;
                std::string op;
                if (!(izq >> c.rasgo >> op >> c.valor)) throw std::invalid_argument("--regla: condición mal formada en «" + linea + "»");
                if (c.rasgo < 0 || c.rasgo >= dim_obs) throw std::invalid_argument("--regla: rasgo fuera de la observación");
                if (op != "<=" && op != ">") throw std::invalid_argument("--regla: operador " + op + " (se acepta <= o >)");
                c.menor_igual = op == "<=";
                r.cond.push_back(c);
                if (!(izq >> tok)) break;
                if (tok != "y") throw std::invalid_argument("--regla: se esperaba 'y' entre condiciones");
            }
        } else if (tok != "siempre") {
            throw std::invalid_argument("--regla: cada línea empieza con 'si' o 'siempre'");
        }
        while (der >> tok) {
            const auto eq = tok.find('=');
            if (eq == std::string::npos) throw std::invalid_argument("--regla: empuje mal formado «" + tok + "»");
            const std::string nombre = tok.substr(0, eq);
            size_t k = 0;
            while (k < catalogo.size() && nombre != neat::operatorName(catalogo[k])) k++;
            if (k == catalogo.size()) throw std::invalid_argument("--regla: operador desconocido «" + nombre + "»");
            r.empuje[k] = std::stod(tok.substr(eq + 1));
        }
        reglas.push_back(r);
    }
    if (reglas.empty()) throw std::invalid_argument("--regla: archivo sin reglas");
    return reglas;
}

const std::vector<double>* reglaQueAplica(const std::vector<Regla>& reglas, const std::vector<float>& obs) {
    for (const auto& r : reglas) {
        bool ok = true;
        for (const auto& c : r.cond) ok = ok && (c.menor_igual ? obs[c.rasgo] <= c.valor : obs[c.rasgo] > c.valor);
        if (ok) return &r.empuje;
    }
    return nullptr;
}

// Suma a la acción `a` (sus primeros K valores: P) el empuje de la primera regla que se cumple con la observación
void sumarRegla(const std::vector<Regla>& reglas, const std::vector<float>& obs, double* a) {
    if (const std::vector<double>* e = reglaQueAplica(reglas, obs))
        for (size_t k = 0; k < e->size(); k++) a[k] += (*e)[k];
}

const char* nombreModo(aos::ModoAccion m) {
    return m == aos::ModoAccion::P ? "p" : m == aos::ModoAccion::Intensidades ? "intensidades" : "ambas";
}

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
template <TI OBS_DIM, TI ACT_DIM>
void exportarActor(DEVICE& device, typename Tipos<OBS_DIM, ACT_DIM>::STATE& ts, const std::string& dir, const aos::NeatRunConfig& cfg,
                   const std::vector<double>& P0, const aos::MapaAccion& mapa, const Opciones& o, int iter) {
    using ACTOR = typename Tipos<OBS_DIM, ACT_DIM>::CONFIG::NN::ACTOR_TYPE;
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
    const int K = mapa.K();
    ss << "\n    constexpr int K = " << K << ";";
    ss << "\n    constexpr int OBS = " << OBS_DIM << ";";
    ss << "\n    constexpr int ACT = " << ACT_DIM << ";";
    ss << "\n    constexpr int T = " << cfg.T << ";";
    ss << "\n    constexpr int version_rasgos = " << (cfg.estado_snn ? 3 : cfg.estado_compacto ? 2 : 1)
       << ";   // 1: completo (14 + 2K), 2: compacto (9), 3: SNN (10)";
    ss << "\n    constexpr const char* accion = \"" << nombreModo(mapa.modo) << "\";";
    ss << "\n    constexpr double c = " << std::hexfloat << o.c << ";";
    ss << "\n    constexpr double P0[] = {";
    for (int k = 0; k < K; k++) ss << (k ? ", " : "") << std::hexfloat << P0[k];
    ss << "};";
    ss << "\n    constexpr double base_potencia = " << mapa.base.weight_perturb_power << ", base_perturbar = "
       << mapa.base.weight_perturb_prob << ", base_sin_cruce = " << mapa.base.mutation_only_fraction << ";";
    ss << "\n    constexpr double c_potencia = " << mapa.c_potencia << ", c_reasignar = " << mapa.c_reasignar
       << ", c_sin_cruce = " << mapa.c_sin_cruce << ";";
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

// --- políticas sobre NeatPool: validación y evaluación final ---
struct Eval {
    std::vector<double> final_test, score;
};

// Decisión de una política para todas las ranuras: P y, en el plan B, intensidades (vacío = las vigentes)
struct Decisiones {
    std::vector<std::vector<double>> P;
    std::vector<neat::Variacion> V;
};

// Corre `n` corridas con semillas seed_base, seed_base + 1, … (en bloques de hasta 16) decidiendo P con `politica`.
// Con `dir`, cada corrida deja dir/seed_XX/generations.csv y summary.txt con el formato de neat_control.
template <typename POLITICA>
Eval evaluarPolitica(const aos::NeatRunConfig& cfg, uint64_t seed_base, int n, double w, double escala, const std::string& dir,
                     const std::vector<double>& P_log, POLITICA&& politica) {
    Eval e;
    while (static_cast<int>(e.final_test.size()) < n) {
        const int m = std::min<int>(N_CORRIDAS, n - static_cast<int>(e.final_test.size()));
        aos::NeatPool pool(cfg, m, seed_base + e.final_test.size(), w, escala, true);
        if (!dir.empty()) {
            std::filesystem::create_directories(dir);
            pool.enableCsv(dir, true, P_log);
            pool.enableDecisionLog(dir + "/decisiones.csv", !e.final_test.empty());   // estado de cada decisión (análisis)
        }
        std::vector<int> all(m);
        std::iota(all.begin(), all.end(), 0);
        pool.start(all);
        for (int t = 0; t < cfg.T; t++) {
            const Decisiones d = politica(pool);
            pool.advance(d.P, d.V.empty() ? nullptr : &d.V);
        }
        for (const auto& f : pool.takeFinished()) {
            e.final_test.push_back(f.final_test);
            e.score.push_back(f.final_score);
            if (dir.empty()) continue;
            for (int s = 0; s < m; s++) {
                aos::NeatRun& r = pool.run(s);
                if (r.seed() != f.seed) continue;
                char sub[32];
                std::snprintf(sub, sizeof sub, "/seed_%02llu", static_cast<unsigned long long>(f.seed));
                std::ofstream sum(dir + sub + "/summary.txt");
                sum << "env=" << cfg.env_name << "\nseed=" << f.seed << "\nselector=ppo\nfinal_test=" << f.final_test
                    << "\nbest_train=" << r.champion().fitness << "\nbest_hidden=" << r.champion().numHidden()
                    << "\nbest_conns=" << r.champion().numEnabledConns() << "\nresample_train=1\nbest_validation_test="
                    << r.bestValidation() << "\n";
            }
        }
    }
    return e;
}

// La regla escrita a mano como política: el empuje de la primera regla que se cumple pasa por el mapa de acciones
Eval evaluarRegla(const std::vector<Regla>& reglas, const aos::NeatRunConfig& cfg, const aos::MapaAccion& mapa, uint64_t seed_base,
                  int n, double w, double escala, const std::string& dir = "") {
    const size_t K = static_cast<size_t>(mapa.K());
    std::vector<double> P0n(K);
    for (size_t k = 0; k < K; k++) P0n[k] = std::exp(mapa.logp0[k]);
    return evaluarPolitica(cfg, seed_base, n, w, escala, dir, P0n, [&](aos::NeatPool& pool) {
        Decisiones d;
        d.P.resize(pool.slots());
        std::vector<neat::Variacion> V(pool.slots());
        for (int i = 0; i < pool.slots(); i++) {
            std::vector<double> accion(mapa.dim(), 0.0);
            sumarRegla(reglas, pool.obs(i), accion.data());
            mapa.decodificar(accion.data(), d.P[i], V[i]);
        }
        if (mapa.mueveIntensidades()) d.V = std::move(V);
        return d;
    });
}

template <TI OBS_DIM, TI ACT_DIM>
Eval evaluarActor(DEVICE& device, typename Tipos<OBS_DIM, ACT_DIM>::STATE& ts, const aos::NeatRunConfig& cfg, const aos::MapaAccion& mapa,
                  uint64_t seed_base, int n, double w, double escala, const std::string& dir = "",
                  const std::vector<Regla>* base = nullptr) {
    using TP = Tipos<OBS_DIM, ACT_DIM>;
    constexpr TI N = TP::N, D = TP::OBS, A = TP::ACT;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, D>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, A>>> output;
    typename TP::CONFIG::NN::ACTOR_TYPE::template State<true> pstate;
    RNG rng;
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, pstate);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    rlt::reset(device, ts.ppo.actor, pstate, rng);
    const int K = mapa.K();
    std::vector<double> P0n(K);
    for (int k = 0; k < K; k++) P0n[k] = std::exp(mapa.logp0[k]);
    auto politica = [&](aos::NeatPool& pool) {
        const int m = pool.slots();
        rlt::set_all(device, input, static_cast<T>(0));
        for (int i = 0; i < m; i++)
            for (TI j = 0; j < D; j++) rlt::set(device, input, pool.obs(i)[j], i, j);
        rlt::evaluate_step(device, ts.ppo.actor, input, pstate, output, ts.actor_eval_buffers, rng, rlt::Mode<rlt::mode::Rollout<>>{});
        Decisiones d;
        d.P.resize(m);
        std::vector<neat::Variacion> V(m);
        for (int i = 0; i < m; i++) {
            double a[A];
            for (TI j = 0; j < A; j++) a[j] = rlt::get(device, output, i, j);
            if (base) sumarRegla(*base, pool.obs(i), a);   // PPO residual
            mapa.decodificar(a, d.P[i], V[i]);
        }
        if (mapa.mueveIntensidades()) d.V = std::move(V);
        return d;
    };
    Eval e = evaluarPolitica(cfg, seed_base, n, w, escala, dir, P0n, politica);
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, pstate);
    rlt::free(device, rng);
    return e;
}

Eval evaluarFija(const std::vector<double>& P, const aos::NeatRunConfig& cfg, uint64_t seed_base, int n, double w, double escala,
                 const std::string& dir = "") {
    return evaluarPolitica(cfg, seed_base, n, w, escala, dir, P,
                           [&](aos::NeatPool& pool) { return Decisiones{std::vector<std::vector<double>>(pool.slots(), P), {}}; });
}

template <TI OBS_DIM, TI ACT_DIM>
double klAprox(DEVICE& device, typename Tipos<OBS_DIM, ACT_DIM>::STATE& ts) {
    using TP = Tipos<OBS_DIM, ACT_DIM>;
    constexpr TI N = TP::N, D = TP::OBS, K = TP::ACT;
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
template <TI OBS_DIM, TI ACT_DIM>
void acumularEstandarizacion(DEVICE& device, typename Tipos<OBS_DIM, ACT_DIM>::STATE& ts) {
    using TP = Tipos<OBS_DIM, ACT_DIM>;
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

template <TI OBS_DIM, TI ACT_DIM>
int entrenar(const Opciones& o, const std::vector<aos::NeatRunConfig>& cfgs, const std::vector<double>& P0, const aos::MapaAccion& mapa,
             const std::string& linea, const std::string& sha) {
    using TP = Tipos<OBS_DIM, ACT_DIM>;
    const int K = mapa.K();
    constexpr TI A = TP::ACT;
    using CONFIG = typename TP::CONFIG;
    constexpr TI N = TP::N;
    // Con dos entornos (--mezcla), la mitad de las ranuras corre cada uno; la observación no tiene unidades del entorno
    aos::NeatRunConfig cfg = cfgs.front();
    std::vector<aos::NeatRunConfig> por_ranura;
    for (TI s = 0; s < N; s++) por_ranura.push_back(cfgs[cfgs.size() == 1 ? 0 : (s < N / 2 ? 0 : 1)]);
    for (size_t i = 1; i < cfgs.size(); i++) cfg.env_name += "+" + cfgs[i].env_name;
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
        if (std::stoi(kv.at("K")) != K || std::stoi(kv.at("OBS")) != static_cast<int>(TP::OBS))
            throw std::runtime_error("--reanudar: el checkpoint es de otro catálogo");
        const std::string accion_ck = kv.count("accion") ? kv.at("accion") : "p";
        if (accion_ck != nombreModo(mapa.modo)) throw std::runtime_error("--reanudar: el checkpoint es de otra acción (" + accion_ck + ")");
        if (kv.at("binario") != sha) std::printf("aviso: el checkpoint se guardó con otro binario (%s)\n", kv.at("binario").c_str());
        iter = std::stoi(kv.at("iteracion"));
        corridas = std::stol(kv.at("corridas"));
        seed_base = std::stoull(kv.at("semilla_en_curso"));
        std::printf("reanudando: iteración %d, %ld corridas, corridas en curso desde la semilla %llu\n", iter, corridas,
                    static_cast<unsigned long long>(seed_base));
    }

    aos::NeatPool pool(por_ranura, seed_base, o.w, o.escala, !o.sin_test_final);
    pool.enableDecisionLog(o.out + "/decisiones.csv", o.reanudar, mapa.mueveIntensidades());
    typename TP::STATE ts;
    rlt::malloc(device, ts);
    ts.environment.pool = &pool;
    ts.environment.logp0 = aos::logP0(P0);
    ts.environment.c = o.c;
    ts.environment.mapa = &mapa;
    std::vector<Regla> base;   // PPO residual (--regla-base)
    if (!o.regla_base.empty()) {
        base = leerReglas(o.regla_base, cfg.ncfg.catalog, static_cast<int>(TP::OBS));
        ts.environment.residual = [&pool, &base](int i, double* a) { sumarRegla(base, pool.obs(i), a); };
    }
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
        for (int k = 0; k < K; k++) ppo_csv << ",p" << k;
        if (mapa.mueveIntensidades()) ppo_csv << ",potencia,reasignar,solo_mutacion";
        ppo_csv << "\n";
        corr_csv << "iteracion,semilla,entorno,test_final,score,retorno,mejor_validacion\n";
        std::ofstream c(o.out + "/config.txt");
        c << "linea=" << linea << "\nbinario=" << sha << "\nK=" << K << "\nOBS=" << TP::OBS << "\nACT=" << A
          << "\nestado=" << (cfg.estado_snn ? "snn" : cfg.estado_compacto ? "compacto" : "completo") << "\naccion=" << nombreModo(mapa.modo)
          << "\nbase_potencia=" << mapa.base.weight_perturb_power << "\nbase_reasignar=" << 1.0 - mapa.base.weight_perturb_prob
          << "\nbase_sin_cruce=" << mapa.base.mutation_only_fraction << "\nN=" << N
          << "\npasos_por_iteracion=" << PASOS_POR_ITER << "\nT=" << cfg.T << "\ngeneraciones=" << cfg.generations
          << "\npoblacion=" << cfg.ncfg.population_size << "\nP0=" << join(P0) << "\nref_trivial=" << cfg.ref_trivial
          << "\nref_rl=" << cfg.ref_rl << "\nc=" << o.c << "\nw=" << o.w << "\nescala=" << o.escala << "\nstd0=" << o.std0
          << "\nlr=" << o.lr << "\nlr_critico=" << o.lr_critico << "\nregla_base=" << o.regla_base << "\ngamma=0.99\nlambda=1\nepocas=20\nhilos=" << omp_get_max_threads() << "\n";
    }
    std::printf("PPO sobre NEAT: entorno %s, K=%d, OBS=%d (estado %s), ACT=%d (acción %s), %d corridas en paralelo, T=%d, G=%d, "
                "presupuesto %d corridas, hilos %d\n",
                cfg.env_name.c_str(), K, static_cast<int>(TP::OBS), cfg.estado_snn ? "snn" : cfg.estado_compacto ? "compacto" : "completo",
                static_cast<int>(A),
                nombreModo(mapa.modo), static_cast<int>(N), cfg.T, cfg.generations, o.corridas, omp_get_max_threads());
    std::printf("P0 = (%s)  refs %.4g → %.4g\n", join(P0, "%.4f").c_str(), cfg.ref_trivial, cfg.ref_rl);
    std::fflush(stdout);

    while (corridas < o.corridas) {
        const double ti = ahora();
        rlt::collect(device, ts.on_policy_runner_dataset, ts.on_policy_runner, ts.on_policy_runner_buffer, ts.environment, ts.ppo.actor,
                     ts.actor_eval_buffers, ts.ppo.critic, ts.critic_collection_state, ts.critic_collection_buffer, ts.rng,
                     typename decltype(ts.ppo)::SPEC::COLLECTION_MODE{});
        if (iter < o.iter_normalizar) acumularEstandarizacion<OBS_DIM, ACT_DIM>(device, ts);
        rlt::estimate_generalized_advantages(device, ts.on_policy_runner_dataset, ts.on_policy_runner_dataset.bootstrap_values,
                                             typename CONFIG::PPO_TYPE::SPEC::PARAMETERS{});
        rlt::train(device, ts.ppo, ts.on_policy_runner_dataset, ts.actor_optimizer, ts.critic_optimizer, ts.ppo_buffers, ts.actor_buffers,
                   ts.critic_buffers, ts.rng);
        iter++;
        const double kl = klAprox<OBS_DIM, ACT_DIM>(device, ts);
        auto& last = rlt::get_last_layer(ts.ppo.actor);
        double ls = 0.0;
        for (TI j = 0; j < A; j++) ls += rlt::get(device, last.log_std.parameters, j);
        ls /= A;
        const auto fin = pool.takeFinished();
        const auto pm = pool.takeMeanP();
        const auto pv = pool.takeMeanVariacion();
        corridas += static_cast<long>(fin.size());
        std::vector<double> sc, re, bv;
        for (const auto& f : fin) {
            sc.push_back(f.final_score);
            re.push_back(f.ret);
            bv.push_back(f.best_validation);
            corr_csv << iter << "," << f.seed << "," << f.env << "," << f.final_test << "," << f.final_score << "," << f.ret << ","
                     << f.best_validation << "\n";
        }
        corr_csv.flush();
        const double secs = ahora() - ti;
        ppo_csv << iter << "," << corridas << "," << secs << "," << media(sc) << "," << media(re) << "," << media(bv) << "," << ls << ","
                << kl;
        for (double p : pm) ppo_csv << "," << p;
        if (mapa.mueveIntensidades())
            for (int k = 0; k < 3; k++) ppo_csv << "," << (pv.empty() ? NAN : pv[k]);
        ppo_csv << "\n";
        ppo_csv.flush();
        std::printf("iter %3d  corridas %4ld  %6.0f s  score %s  retorno %s  log σ %.3f  KL %.5f  P=(%s)%s%s\n", iter, corridas, secs,
                    fin.empty() ? "   —  " : (std::to_string(media(sc)).substr(0, 6)).c_str(),
                    fin.empty() ? "   —  " : (std::to_string(media(re)).substr(0, 6)).c_str(), ls, kl, join(pm, "%.3f").c_str(),
                    pv.empty() ? "" : "  intensidades=(", pv.empty() ? "" : (join(pv, "%.4g") + ")").c_str());
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
              << "\nOBS=" << TP::OBS << "\naccion=" << nombreModo(mapa.modo) << "\nbinario=" << sha << "\n";
            e.close();
            std::filesystem::rename(f_estado_txt + ".tmp", f_estado_txt);
        }
        if (o.exportar_cada > 0 && iter % o.exportar_cada == 0) {
            char d[32];
            std::snprintf(d, sizeof d, "/actor_iter%04d", iter);
            exportarActor<OBS_DIM, ACT_DIM>(device, ts, o.out + d, cfg, P0, mapa, o, iter);
        }
        if (o.max_iter > 0 && iter >= o.max_iter) {
            std::printf("detenido en la iteración %d (--max-iter)\n", iter);
            rlt::free(device, ts);
            return 0;
        }
    }
    exportarActor<OBS_DIM, ACT_DIM>(device, ts, o.out + "/actor_final", cfg, P0, mapa, o, iter);
    guardarEstado(device, ts, o.out + "/estado_final.tar");
    std::printf("entrenamiento terminado: %d iteraciones, %ld corridas, %.1f h\n", iter, corridas, (ahora() - t0) / 3600.0);

    if (o.validar > 0) {
        std::ofstream v(o.out + "/validacion.txt");
        for (const auto& cv : cfgs) {   // validación pareada por entorno
            const Eval ppo = evaluarActor<OBS_DIM, ACT_DIM>(device, ts, cv, mapa, o.semilla_validacion, o.validar, o.w, o.escala, "",
                                                            base.empty() ? nullptr : &base);
            // sin residual, contra P₀ (acción 0); con residual, contra la regla base (también acción 0 del agente)
            const Eval ref = base.empty() ? evaluarFija(std::vector<double>(P0), cv, o.semilla_validacion, o.validar, o.w, o.escala)
                                          : evaluarRegla(base, cv, mapa, o.semilla_validacion, o.validar, o.w, o.escala);
            std::vector<double> d(ppo.score.size());
            int gana = 0;
            for (size_t i = 0; i < d.size(); i++) {
                d[i] = ppo.score[i] - ref.score[i];
                gana += d[i] > 0;
            }
            v << "[" << cv.env_name << "]\nsemillas=" << o.semilla_validacion << "-" << o.semilla_validacion + o.validar - 1
              << "\nscore_ppo=" << media(ppo.score) << "\nscore_p0=" << media(ref.score) << "\ndelta=" << media(d)
              << "\nee_delta=" << desv(d) / std::sqrt(d.size()) << "\nppo_gana=" << gana << "/" << d.size() << "\ntest_final_ppo="
              << join(ppo.final_test) << "\ntest_final_p0=" << join(ref.final_test) << "\n";
            std::printf("validación %s (%d semillas): PPO %.4f, P0 %.4f, Δ %+.4f (ee %.4f), PPO gana en %d\n", cv.env_name.c_str(), o.validar,
                        media(ppo.score), media(ref.score), media(d), desv(d) / std::sqrt(d.size()), gana);
        }
    }
    rlt::free(device, ts);
    return 0;
}

// Evaluación final: la política congelada (o una P fija, para verificar) en las semillas pedidas, con generations.csv y
// summary.txt por corrida como neat_control (NeatRun es idéntico a neat_control: verificación V1). Reemplaza integrar PPO
// en neat_control.
template <TI OBS_DIM, TI ACT_DIM>
int evaluarModo(const Opciones& o, const aos::NeatRunConfig& cfg, const std::vector<double>& P0, const aos::MapaAccion& mapa) {
    const size_t K = static_cast<size_t>(mapa.K());
    const auto guion = o.semillas.find('-');
    const uint64_t a = std::stoull(o.semillas.substr(0, guion));
    const uint64_t b = guion == std::string::npos ? a : std::stoull(o.semillas.substr(guion + 1));
    const int n = static_cast<int>(b - a + 1);
    Eval e;
    if (!o.regla.empty()) {
        e = evaluarRegla(leerReglas(o.regla, cfg.ncfg.catalog, static_cast<int>(OBS_DIM)), cfg, mapa, a, n, o.w, o.escala, o.out);
    } else if (!o.politica_fija.empty()) {
        // misma normalización que neat_control con --p
        const auto P = neat::StaticSelector(parseP(o.politica_fija, K)).probabilities();
        e = evaluarFija(P, cfg, a, n, o.w, o.escala, o.out);
    } else {
        using TP = Tipos<OBS_DIM, ACT_DIM>;
        DEVICE device;
        typename TP::STATE ts;
        rlt::malloc(device, ts);
        ts.environment.logp0 = aos::logP0(P0);
        ts.environment.c = o.c;
        ts.environment.mapa = &mapa;
        rlt::init(device, ts.rng, 0);
        rlt::init(device, ts.ppo, ts.actor_optimizer, ts.critic_optimizer, ts.rng);
        const std::string f = std::filesystem::exists(o.evaluar + "/estado_final.tar") ? o.evaluar + "/estado_final.tar" : o.evaluar + "/estado.tar";
        cargarEstado(device, ts, f);
        std::printf("política: %s\n", f.c_str());
        std::vector<Regla> base;
        if (!o.regla_base.empty()) base = leerReglas(o.regla_base, cfg.ncfg.catalog, static_cast<int>(OBS_DIM));
        e = evaluarActor<OBS_DIM, ACT_DIM>(device, ts, cfg, mapa, a, n, o.w, o.escala, o.out, base.empty() ? nullptr : &base);
        rlt::free(device, ts);
    }
    for (int i = 0; i < n; i++) std::printf("semilla %llu: test final %.6g (score %.4f)\n", static_cast<unsigned long long>(a + i), e.final_test[i], e.score[i]);
    std::printf("score medio %.4f (%d semillas)\n", media(e.score), n);
    return 0;
}

template <TI OBS_DIM, TI ACT_DIM>
int lanzar(const Opciones& o, const std::vector<aos::NeatRunConfig>& cfgs, const std::vector<double>& P0, const aos::MapaAccion& mapa,
           const std::string& l, const std::string& sha) {
    if (static_cast<TI>(mapa.dim()) != ACT_DIM) throw std::logic_error("despachar: dimensión de la acción inconsistente");
    if (!o.evaluar.empty() || !o.politica_fija.empty() || !o.regla.empty()) return evaluarModo<OBS_DIM, ACT_DIM>(o, cfgs.front(), P0, mapa);
    return entrenar<OBS_DIM, ACT_DIM>(o, cfgs, P0, mapa, l, sha);
}

int despachar(size_t K, const Opciones& o, const std::vector<aos::NeatRunConfig>& cfg, const std::vector<double>& P0, const std::string& l,
              const std::string& sha) {
    const aos::ModoAccion modo = o.accion == "intensidades" ? aos::ModoAccion::Intensidades
                               : o.accion == "ambas"        ? aos::ModoAccion::Ambas
                                                            : aos::ModoAccion::P;
    const neat::Variacion base = neat::variacionDe(cfg.front().ncfg);
    for (const auto& c : cfg) {   // con --mezcla, las intensidades base deben coincidir (un solo mapa de acciones)
        const neat::Variacion v = neat::variacionDe(c.ncfg);
        if (modo != aos::ModoAccion::P && (v.weight_perturb_power != base.weight_perturb_power ||
                                           v.weight_perturb_prob != base.weight_perturb_prob ||
                                           v.mutation_only_fraction != base.mutation_only_fraction))
            throw std::invalid_argument("--mezcla con --accion " + o.accion + ": las intensidades base de los dos entornos deben coincidir");
    }
    const aos::MapaAccion mapa = aos::MapaAccion::crear(modo, P0, o.c, base);
    if (o.estado == "snn") {
        constexpr TI D = aos::FeatureExtractor::DIM_SNN;
        if (modo == aos::ModoAccion::P && K == 3) return lanzar<D, 3>(o, cfg, P0, mapa, l, sha);
        if (modo == aos::ModoAccion::P && K == 10) return lanzar<D, 10>(o, cfg, P0, mapa, l, sha);
        if (modo == aos::ModoAccion::P && K == 14) return lanzar<D, 14>(o, cfg, P0, mapa, l, sha);   // + F1–F4
        throw std::invalid_argument("estado snn con acción " + o.accion + " y K = " + std::to_string(K) + ": agregar en despachar()");
    }
    if (o.estado == "compacto") {
        constexpr TI D = aos::FeatureExtractor::DIM_COMPACTO;
        if (modo == aos::ModoAccion::Intensidades) return lanzar<D, 3>(o, cfg, P0, mapa, l, sha);
        if (modo == aos::ModoAccion::P && K == 3) return lanzar<D, 3>(o, cfg, P0, mapa, l, sha);
        if (modo == aos::ModoAccion::P && K == 10) return lanzar<D, 10>(o, cfg, P0, mapa, l, sha);
        if (modo == aos::ModoAccion::Ambas && K == 3) return lanzar<D, 6>(o, cfg, P0, mapa, l, sha);
        if (modo == aos::ModoAccion::Ambas && K == 10) return lanzar<D, 13>(o, cfg, P0, mapa, l, sha);
        throw std::invalid_argument("estado compacto con acción " + o.accion + " y K = " + std::to_string(K) + ": agregar en despachar()");
    }
    if (modo != aos::ModoAccion::P) throw std::invalid_argument("--accion " + o.accion + " requiere --estado compacto");
    switch (K) {
        case 3: return lanzar<14 + 2 * 3, 3>(o, cfg, P0, mapa, l, sha);
        case 4: return lanzar<14 + 2 * 4, 4>(o, cfg, P0, mapa, l, sha);
        case 5: return lanzar<14 + 2 * 5, 5>(o, cfg, P0, mapa, l, sha);
        case 6: return lanzar<14 + 2 * 6, 6>(o, cfg, P0, mapa, l, sha);
        case 7: return lanzar<14 + 2 * 7, 7>(o, cfg, P0, mapa, l, sha);
        case 8: return lanzar<14 + 2 * 8, 8>(o, cfg, P0, mapa, l, sha);
        case 9: return lanzar<14 + 2 * 9, 9>(o, cfg, P0, mapa, l, sha);
        case 10: return lanzar<14 + 2 * 10, 10>(o, cfg, P0, mapa, l, sha);
        case 15: return lanzar<14 + 2 * 15, 15>(o, cfg, P0, mapa, l, sha);
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
    app.add_option("--mezcla", o.mezcla, "Opciones de NEAT de un segundo entorno (entre comillas): la mitad de las ranuras lo corre");
    app.add_option("--evaluar", o.evaluar, "Evaluar la política de este directorio (estado_final.tar) en --semillas");
    app.add_option("--semillas", o.semillas, "Semillas de la evaluación (a-b)");
    app.add_option("--politica-fija", o.politica_fija, "Evaluar con una P fija (verificación)");
    app.add_option("--regla-base", o.regla_base, "PPO residual: el agente aprende correcciones sobre esta regla (acción 0 = la regla)");
    app.add_option("--ultima-capa", o.ultima_capa, "Escala inicial de la última capa del actor (verificación: 0 → acción 0 exacta)");
    app.add_option("--regla", o.regla, "Evaluar una política escrita a mano (archivo de reglas sobre el estado; docs/diseno_ppo.md §11)");
    app.add_option("--estado", o.estado, "Estado del agente: completo (14 + 2K rasgos) o compacto (9, plan B)")
        ->check(CLI::IsMember({"completo", "compacto", "snn"}));
    app.add_option("--accion", o.accion, "Acción del agente: p (P sobre el catálogo), intensidades (plan B) o ambas")
        ->check(CLI::IsMember({"p", "intensidades", "ambas"}));
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
        auto armar = [&](const opciones_neat::Opciones& x, bool refs_cli) {
            aos::NeatRunConfig cfg;
            cfg.env_name = x.env_name;
            cfg.env_opt = x.env_opt;
            cfg.ecfg = x.ecfg;
            cfg.ncfg = x.ncfg;
            cfg.generations = x.generations;
            cfg.episodes = x.episodes;
            cfg.test_episodes = x.test_episodes;
            cfg.final_episodes = x.final_episodes;
            cfg.reward_episodes = o.reward_episodes;
            cfg.T = o.T;
            cfg.estado_compacto = o.estado == "compacto";
            cfg.estado_snn = o.estado == "snn";
            const auto it = kReferencias.find(x.env_name);
            const bool cli = refs_cli && !std::isnan(o.ref_trivial) && !std::isnan(o.ref_rl);
            if (!cli && it == kReferencias.end()) throw std::invalid_argument("entorno sin referencias conocidas: dar --ref-trivial y --ref-rl");
            cfg.ref_trivial = refs_cli && !std::isnan(o.ref_trivial) ? o.ref_trivial : it->second.first;
            cfg.ref_rl = refs_cli && !std::isnan(o.ref_rl) ? o.ref_rl : it->second.second;
            return cfg;
        };
        std::vector<aos::NeatRunConfig> cfgs{armar(on, true)};
        if (!o.mezcla.empty()) {
            CLI::App app2{"opciones de NEAT del segundo entorno"};
            opciones_neat::Opciones on2;
            opciones_neat::agregar(app2, on2);
            app2.add_option("--selector", selector);
            app2.add_option("--print-every", print_every);
            std::string p2_ignorada;
            app2.add_option("--p", p2_ignorada, "Ignorada: P₀ es la de --p (la misma en ambos entornos)");
            app2.parse(o.mezcla, false);
            opciones_neat::finalizar(on2);
            if (!on2.resample_train) throw std::invalid_argument("--mezcla: falta --resample-train");
            if (on2.ncfg.catalog != on.ncfg.catalog) throw std::invalid_argument("--mezcla: el catálogo debe ser el mismo");
            cfgs.push_back(armar(on2, false));
        }
        if (o.T % PASOS_POR_ITER != 0 && PASOS_POR_ITER % o.T != 0)
            std::printf("aviso: T no es múltiplo ni divisor de %d: los checkpoints quedarán espaciados\n", static_cast<int>(PASOS_POR_ITER));
        return despachar(K, o, cfgs, P0, linea, sha16(std::filesystem::read_symlink("/proc/self/exe").string()));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
