// Paso 1 del agente PPO (docs/diseno_ppo.md §5, V2–V3): el PPO de RLtools sobre el AOS sintético
// (src/experiment/aos_toy.hpp), con el lote propio (src/experiment/ppo_lote.hpp).
//
//   calibrar   mejor P fija (búsqueda aleatoria + refinamiento) y política oráculo, en semillas nuevas
//   evaluar    score de una P fija
//   entrenar   PPO; curva de aprendizaje contra P₀ con las mismas semillas de evaluación (pareado)
//   verificar  contabilidad del runner (V2): cada corrida recibe exactamente T decisiones, sin semillas repetidas
//
// Variantes (fijadas al compilar, porque RLtools las pide como constantes): base (N = 16, γ = 1, λ = 0,95,
// 10 épocas), g099 (γ = 0,99), l090 / l099 / l100 (λ = 0,9 / 0,99 / 1), l100e4 (λ = 1, 4 épocas), g099l100
// (γ = 0,99 y λ = 1), g099l100e20 (… y 20 épocas), g099l100c300 (… y recorte 0,3), n8 y n32 (N = 8 y 32).
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

#include <CLI/CLI.hpp>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "aos/action_map.hpp"
#include "experiment/aos_toy.hpp"

namespace rlt = rl_tools;

namespace {

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
using T = float;
using TI = typename DEVICE::index_t;
using RNG = rlt::devices::generic::random::ArrayENGINE<rlt::devices::generic::random::ArraySpecification<TI, 1024, true>>;
using ENV = ppo_lote::Entorno<T, TI, aos::toy::OBS, aos::toy::K>;
constexpr int K = aos::toy::K;
constexpr int D = aos::toy::OBS;

// Hiperparámetros de docs/diseno_ppo.md §4.1 (los que RLtools fija al compilar)
template <TI T_N, int GAMMA_MIL, int LAMBDA_MIL, int EPOCHS, int CLIP_MIL>
struct Parametros : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENV> {
    static constexpr TI N_ENVIRONMENTS = T_N;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 25;
    static constexpr TI BATCH_SIZE = 100;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI ACTOR_NUM_LAYERS = 3;   // entrada, 1 oculta, salida: 2 capas ocultas de 64 activaciones
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI CRITIC_HIDDEN_DIM = 128;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr bool NORMALIZE_OBSERVATIONS = false;
    using ACTOR_OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;
    using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;
    struct PPO_PARAMETERS : rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE> {
        static constexpr T GAMMA = GAMMA_MIL / 1000.0f;
        static constexpr T LAMBDA = LAMBDA_MIL / 1000.0f;
        static constexpr TI N_EPOCHS = EPOCHS;
        static constexpr T EPSILON_CLIP = CLIP_MIL / 1000.0f;
    };
};

template <TI T_N, int GAMMA_MIL, int LAMBDA_MIL, int EPOCHS = 10, int CLIP_MIL = 200>
struct Variante {
    static constexpr TI N = T_N;
    using PARAMS = Parametros<T_N, GAMMA_MIL, LAMBDA_MIL, EPOCHS, CLIP_MIL>;
    using BASE = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENV, PARAMS,
                                                               rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;
    using LOTE = ppo_lote::Lote<ppo_lote::LoteSpec<ENV, T_N, aos::toy::Pool>>;
    using CONFIG = ppo_lote::ConfigConLote<BASE, LOTE>;
    using STATE = rlt::rl::algorithms::ppo::loop::core::State<CONFIG>;
};

struct Opciones {
    std::string variante = "base";
    int corridas = 192;                 // presupuesto de entrenamiento, en corridas completas
    uint64_t semilla = 1;               // semilla de PPO (inicialización y muestreo de acciones)
    uint64_t semilla_corridas = 1001;   // primera semilla de las corridas de entrenamiento
    std::vector<double> P0;
    double c = 3.0, w = 0.5, escala = 10.0, std0 = 0.5, lr = 3e-4, lr_critico = -1.0, ultima_capa = 0.01;   // lr_critico < 0: igual a lr
    bool barajar = false;               // control negativo: recompensas barajadas antes de GAE
    aos::toy::Config toy;
    int eval_corridas = 2000;
    uint64_t semilla_eval = 900001;
    int eval_cada = 48;                 // corridas de entrenamiento entre evaluaciones
    int calentar = 0;                   // iteraciones iniciales con el actor congelado (solo aprende el crítico)
    std::string csv;
    bool verificar = false;
};

std::vector<double> parseP(const std::string& txt) {
    std::vector<double> p;
    std::stringstream ss(txt);
    std::string tok;
    while (std::getline(ss, tok, ',')) p.push_back(std::stod(tok));
    if (static_cast<int>(p.size()) != K) throw std::invalid_argument("se esperan " + std::to_string(K) + " valores en P");
    double s = 0.0;
    for (double x : p) {
        if (x < 0.0) throw std::invalid_argument("P con valores negativos");
        s += x;
    }
    for (auto& x : p) x /= s;
    return p;
}

std::string joinP(const std::vector<double>& p) {
    std::string out;
    char buf[32];
    for (size_t i = 0; i < p.size(); i++) {
        std::snprintf(buf, sizeof buf, "%s%.4f", i ? "," : "", p[i]);
        out += buf;
    }
    return out;
}

double media(const std::vector<double>& x) { return std::accumulate(x.begin(), x.end(), 0.0) / x.size(); }
double desv(const std::vector<double>& x) {
    const double m = media(x);
    double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return std::sqrt(s / (x.size() - 1));
}

// Política determinista (acción media) en corridas nuevas con semillas semilla_eval, semilla_eval + 1, …
struct Eval {
    std::vector<double> score, ret;
};

// P fija en el mismo pool (mismas semillas y misma recompensa que la evaluación del actor)
Eval evaluarFija(const std::vector<double>& P, const Opciones& o, int N) {
    if (o.eval_corridas % N != 0) throw std::invalid_argument("--eval-corridas debe ser múltiplo de N");
    aos::toy::Pool pool(o.toy, N, o.semilla_eval, o.w, o.escala);
    std::vector<int> all(N);
    std::iota(all.begin(), all.end(), 0);
    const std::vector<std::vector<double>> PP(N, P);
    Eval e;
    while (static_cast<int>(e.score.size()) < o.eval_corridas) {
        pool.start(all);
        for (int t = 0; t < o.toy.T; t++) pool.advance(PP);
        for (const auto& f : pool.takeFinished()) {
            e.score.push_back(f.final_score);
            e.ret.push_back(f.ret);
        }
    }
    return e;
}

template <typename V>
Eval evaluarActor(DEVICE& device, typename V::STATE& ts, const Opciones& o) {
    constexpr TI N = V::N;
    if (o.eval_corridas % N != 0) throw std::invalid_argument("--eval-corridas debe ser múltiplo de N");
    aos::toy::Pool pool(o.toy, N, o.semilla_eval, o.w, o.escala);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, D>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, K>>> output;
    typename V::CONFIG::NN::ACTOR_TYPE::template State<true> pstate;
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
    while (static_cast<int>(e.score.size()) < o.eval_corridas) {
        pool.start(all);
        for (int t = 0; t < o.toy.T; t++) {
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
            e.score.push_back(f.final_score);
            e.ret.push_back(f.ret);
        }
    }
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, pstate);
    rlt::free(device, rng);
    return e;
}

// KL aproximada entre la política que recolectó el lote y la actual: media de (log π_viejo − log π_nuevo)
// sobre las transiciones del lote (train baraja las filas, pero las mantiene alineadas)
template <typename V>
double klAprox(DEVICE& device, typename V::STATE& ts) {
    constexpr TI N = V::N;
    auto& ds = ts.on_policy_runner_dataset;
    constexpr TI STEPS_TOTAL = V::CONFIG::ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, D>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, K>>> output;
    typename V::CONFIG::NN::ACTOR_TYPE::template State<true> pstate;
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

template <typename DS>
void barajarRecompensas(DS& ds, std::mt19937_64& rng) {
    constexpr TI n = DS::STEPS_TOTAL;
    for (TI i = 0; i + 1 < n; i++) {
        const TI j = i + static_cast<TI>(rng() % (n - i));
        const auto a = rlt::get(ds.rewards, i, 0);
        rlt::set(ds.rewards, i, 0, rlt::get(ds.rewards, j, 0));
        rlt::set(ds.rewards, j, 0, a);
    }
}

template <typename V>
int entrenar(const Opciones& o) {
    using CONFIG = typename V::CONFIG;
    constexpr TI N = V::N;
    DEVICE device;
    aos::toy::Pool pool(o.toy, N, o.semilla_corridas, o.w, o.escala);
    typename V::STATE ts;
    rlt::malloc(device, ts);
    ts.environment.pool = &pool;
    ts.environment.logp0 = aos::logP0(o.P0);
    ts.environment.c = o.c;
    rlt::init(device, ts, o.semilla);
    // Arranque en P₀: última capa del actor 100× menor (Andrychowicz et al. 2021), σ inicial y tasa de aprendizaje
    {
        auto& last = rlt::get_last_layer(ts.ppo.actor);
        rlt::scale(device, last.output_layer.weights.parameters, static_cast<T>(o.ultima_capa));
        rlt::scale(device, last.output_layer.biases.parameters, static_cast<T>(o.ultima_capa));
        rlt::set_all(device, last.log_std.parameters, static_cast<T>(std::log(o.std0)));
        rlt::get_ref(device, ts.actor_optimizer.parameters, 0).alpha = static_cast<T>(o.calentar > 0 ? 0.0 : o.lr);
        rlt::get_ref(device, ts.critic_optimizer.parameters, 0).alpha = static_cast<T>(o.lr_critico > 0 ? o.lr_critico : o.lr);
    }
    std::mt19937_64 rng_barajar(o.semilla * 7919 + 13);

    // Línea base pareada: P₀ (= la política con a = 0) en las mismas semillas de evaluación
    std::vector<double> P0n(K);
    for (int i = 0; i < K; i++) P0n[i] = std::exp(ts.environment.logp0[i]);
    Eval base;
    if (!o.verificar) base = evaluarFija(P0n, o, N);

    std::ofstream csv;
    if (!o.csv.empty()) {
        csv.open(o.csv);
        csv << "tipo,iteracion,corridas,score,retorno,log_std,kl,delta,ee_delta,mejoran,delta_ret,ee_delta_ret";
        for (int i = 0; i < K; i++) csv << ",p" << i;
        csv << "\n";
    }
    std::printf("variante=%s semilla=%lu corridas=%d P0=(%s) c=%.2f w=%.2f escala=%.1f std0=%.2f lr=%.1e calentar=%d barajar=%d "
                "sin_estado=%d lr_critico=%.1e\n",
                o.variante.c_str(), static_cast<unsigned long>(o.semilla), o.corridas, joinP(P0n).c_str(), o.c, o.w, o.escala,
                o.std0, o.lr, o.calentar, o.barajar, !o.toy.con_estado, o.lr_critico > 0 ? o.lr_critico : o.lr);
    if (!o.verificar)
        std::printf("P0 en evaluación: %.4f (DE %.4f, %d corridas)  retorno %.3f\n", media(base.score), desv(base.score), o.eval_corridas,
                    media(base.ret));

    auto evaluar = [&](int iter, long corridas) {
        const auto e = evaluarActor<V>(device, ts, o);
        const size_t n = e.score.size();
        std::vector<double> d(n), dr(n);
        int mejoran = 0;
        for (size_t i = 0; i < n; i++) {
            d[i] = e.score[i] - base.score[i];
            dr[i] = e.ret[i] - base.ret[i];
            mejoran += d[i] > 0;
        }
        const double ee = desv(d) / std::sqrt(static_cast<double>(n));
        const double eer = desv(dr) / std::sqrt(static_cast<double>(n));
        std::printf("  eval  corridas %4ld  score %.4f  Δ vs P0 %+.4f (ee %.4f)  mejoran %.2f  Δ retorno %+.4f (ee %.4f)\n", corridas,
                    media(e.score), media(d), ee, static_cast<double>(mejoran) / n, media(dr), eer);
        if (csv.is_open())
            csv << "eval," << iter << "," << corridas << "," << media(e.score) << "," << media(e.ret) << ",,," << media(d) << "," << ee
                << "," << static_cast<double>(mejoran) / n << "," << media(dr) << "," << eer << std::string(K, ',') << "\n";
        return media(d);
    };

    long corridas = 0;
    long proxima_eval = o.eval_cada;
    int iter = 0;
    double delta_final = 0.0;
    while (corridas < o.corridas) {
        rlt::collect(device, ts.on_policy_runner_dataset, ts.on_policy_runner, ts.on_policy_runner_buffer, ts.environment, ts.ppo.actor,
                     ts.actor_eval_buffers, ts.ppo.critic, ts.critic_collection_state, ts.critic_collection_buffer, ts.rng,
                     typename decltype(ts.ppo)::SPEC::COLLECTION_MODE{});
        if (o.barajar) barajarRecompensas(ts.on_policy_runner_dataset, rng_barajar);
        rlt::estimate_generalized_advantages(device, ts.on_policy_runner_dataset, ts.on_policy_runner_dataset.bootstrap_values,
                                             typename CONFIG::PPO_TYPE::SPEC::PARAMETERS{});
        rlt::train(device, ts.ppo, ts.on_policy_runner_dataset, ts.actor_optimizer, ts.critic_optimizer, ts.ppo_buffers, ts.actor_buffers,
                   ts.critic_buffers, ts.rng);
        iter++;
        if (iter == o.calentar) rlt::get_ref(device, ts.actor_optimizer.parameters, 0).alpha = static_cast<T>(o.lr);
        const double kl = klAprox<V>(device, ts);
        auto& last = rlt::get_last_layer(ts.ppo.actor);
        double ls = 0.0;
        for (int j = 0; j < K; j++) ls += rlt::get(device, last.log_std.parameters, j);
        ls /= K;
        const auto fin = pool.takeFinished();
        const auto pm = pool.takeMeanP();
        corridas += static_cast<long>(fin.size());
        if (!fin.empty()) {
            double sc = 0.0, re = 0.0;
            for (const auto& f : fin) {
                sc += f.final_score;
                re += f.ret;
            }
            sc /= fin.size();
            re /= fin.size();
            std::printf("iter %3d  corridas %4ld  score %.4f  retorno %.3f  log σ %.3f  KL %.5f  P=(%s)\n", iter, corridas, sc, re, ls, kl,
                        joinP(std::vector<double>(pm.begin(), pm.end())).c_str());
            if (csv.is_open()) {
                csv << "iter," << iter << "," << corridas << "," << sc << "," << re << "," << ls << "," << kl << ",,,,,";
                for (double p : pm) csv << "," << p;
                csv << "\n";
            }
        }
        if (!o.verificar && (corridas >= proxima_eval || corridas >= o.corridas)) {
            delta_final = evaluar(iter, corridas);
            while (proxima_eval <= corridas) proxima_eval += o.eval_cada;
        }
    }

    int rc = 0;
    if (o.verificar) {
        // V2: el pool ya lanza una excepción si una corrida recibe una decisión de más o se reinicia sin
        // terminar; aquí se confirma que lo contado cuadra
        const long en_curso = pool.started() - pool.finishedCount();
        const bool ok = en_curso == N && pool.decisions() == pool.finishedCount() * o.toy.T &&
                        pool.nextSeed() == o.semilla_corridas + static_cast<uint64_t>(pool.started());
        std::printf("verificar: iniciadas %ld, terminadas %ld, en curso %ld, decisiones %ld (= terminadas × T: %s), semillas %lu…%lu: %s\n",
                    pool.started(), pool.finishedCount(), en_curso, pool.decisions(),
                    pool.decisions() == pool.finishedCount() * o.toy.T ? "sí" : "NO", static_cast<unsigned long>(o.semilla_corridas),
                    static_cast<unsigned long>(pool.nextSeed() - 1), ok ? "OK" : "FALLA");
        rc = ok ? 0 : 1;
    } else {
        std::printf("RESULTADO variante=%s semilla=%lu corridas=%ld barajar=%d sin_estado=%d delta=%.5f\n", o.variante.c_str(),
                    static_cast<unsigned long>(o.semilla), corridas, o.barajar, !o.toy.con_estado, delta_final);
    }
    rlt::free(device, ts);
    return rc;
}

int despachar(const Opciones& o) {
    if (o.variante == "base") return entrenar<Variante<16, 1000, 950>>(o);
    if (o.variante == "g099") return entrenar<Variante<16, 990, 950>>(o);
    if (o.variante == "l090") return entrenar<Variante<16, 1000, 900>>(o);
    if (o.variante == "n8") return entrenar<Variante<8, 1000, 950>>(o);
    if (o.variante == "l099") return entrenar<Variante<16, 1000, 990>>(o);
    if (o.variante == "l100") return entrenar<Variante<16, 1000, 1000>>(o);
    if (o.variante == "l100e4") return entrenar<Variante<16, 1000, 1000, 4>>(o);
    if (o.variante == "g099l100") return entrenar<Variante<16, 990, 1000>>(o);
    if (o.variante == "n32") return entrenar<Variante<32, 1000, 950>>(o);
    if (o.variante == "g099l100e20") return entrenar<Variante<16, 990, 1000, 20>>(o);
    if (o.variante == "g099l100c300") return entrenar<Variante<16, 990, 1000, 10, 300>>(o);
    throw std::invalid_argument("variante desconocida: " + o.variante);
}

// Mejor P fija: candidatos aleatorios (Dirichlet) y estructurados, todos con las mismas semillas; los 30
// mejores se reevalúan con más corridas y el mejor se refina con perturbaciones en escala logarítmica.
// Los números finales salen de semillas nuevas, para no arrastrar el sesgo de la selección.
int calibrar(const aos::toy::Config& cfg, int n_cand, int r_busqueda, int r_final, int r_test, uint64_t semilla) {
    std::mt19937_64 rng(semilla);
    std::gamma_distribution<double> gam(0.3, 1.0);
    std::vector<std::vector<double>> cand;
    auto norm = [](std::vector<double> p) {
        double s = 0.0;
        for (double x : p) s += x;
        for (auto& x : p) x /= s;
        return p;
    };
    cand.push_back(norm({1, 1, 1, 1, 1, 1, 1, 1}));
    cand.push_back(norm({1, 0, 0, 0, 0, 0, 0, 0}));
    cand.push_back(norm({0.8, 0, 0.1, 0.1, 0, 0, 0, 0}));
    cand.push_back(norm({0.6, 0.2, 0.05, 0.15, 0, 0, 0, 0}));
    cand.push_back(norm({0.5, 0.2, 0.1, 0.15, 0.03, 0.02, 0, 0}));
    while (static_cast<int>(cand.size()) < n_cand) {
        std::vector<double> p(K);
        for (auto& x : p) x = gam(rng) + 1e-12;
        cand.push_back(norm(p));
    }
    auto puntuar = [&](const std::vector<std::vector<double>>& cs, uint64_t base, int r) {
        std::vector<double> m(cs.size());
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < cs.size(); i++) m[i] = media(aos::toy::evalStatic(cs[i], cfg, base, r));
        return m;
    };
    auto m1 = puntuar(cand, 500001, r_busqueda);
    std::vector<size_t> idx(cand.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return m1[a] > m1[b]; });
    std::vector<std::vector<double>> top;
    for (size_t i = 0; i < std::min<size_t>(30, idx.size()); i++) top.push_back(cand[idx[i]]);
    auto m2 = puntuar(top, 600001, r_final);
    size_t ib = std::max_element(m2.begin(), m2.end()) - m2.begin();
    std::vector<double> best = top[ib];
    double best_m = m2[ib];
    std::printf("búsqueda: %zu candidatos × %d corridas; los 30 mejores × %d corridas; mejor %.4f\n", cand.size(), r_busqueda, r_final,
                best_m);
    // refinamiento: perturbaciones log-normales del mejor, por rondas
    std::normal_distribution<double> n01(0.0, 1.0);
    for (int ronda = 0; ronda < 4; ronda++) {
        std::vector<std::vector<double>> vec;
        for (int j = 0; j < 64; j++) {
            std::vector<double> p(K);
            for (int i = 0; i < K; i++) p[i] = std::max(best[i], 1e-4) * std::exp(0.4 * n01(rng));
            vec.push_back(norm(p));
        }
        auto mv = puntuar(vec, 600001, r_final);
        size_t jb = std::max_element(mv.begin(), mv.end()) - mv.begin();
        if (mv[jb] > best_m) {
            best_m = mv[jb];
            best = vec[jb];
        }
        std::printf("refinamiento %d: mejor %.4f\n", ronda + 1, best_m);
    }
    // números finales en semillas nuevas (700001…)
    const auto s_best = aos::toy::evalStatic(best, cfg, 700001, r_test);
    const auto s_or = aos::toy::evalOracle(cfg, 700001, r_test);
    const auto s_uni = aos::toy::evalStatic(cand[0], cfg, 700001, r_test);
    const auto s_st = aos::toy::evalStatic(cand[2], cfg, 700001, r_test);
    std::vector<double> d(r_test);
    for (int i = 0; i < r_test; i++) d[i] = s_or[i] - s_best[i];
    std::printf("\nPrueba en %d corridas nuevas:\n", r_test);
    std::printf("  mejor P fija  %.4f (DE %.4f)  P=(%s)\n", media(s_best), desv(s_best), joinP(best).c_str());
    std::printf("  oráculo       %.4f (DE %.4f)  ganancia sobre la mejor fija %+.4f (ee %.4f)\n", media(s_or), desv(s_or), media(d),
                desv(d) / std::sqrt(static_cast<double>(r_test)));
    std::printf("  uniforme      %.4f (DE %.4f)\n", media(s_uni), desv(s_uni));
    std::printf("  tipo Stanley  %.4f (DE %.4f)\n", media(s_st), desv(s_st));
    std::printf("P0=%s\n", joinP(best).c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"PPO de RLtools sobre el AOS sintético (paso 1 del agente PPO, docs/diseno_ppo.md §5)"};
    app.require_subcommand(1);
    Opciones o;
    std::string p0_txt, p_txt;
    bool sin_estado = false;
    int n_cand = 3000, r_busqueda = 200, r_final = 2000, r_test = 4000, r_eval = 4000;
    uint64_t semilla_cal = 1;

    auto agregarToy = [&](CLI::App* sc) {
        sc->add_option("--T", o.toy.T, "Decisiones por corrida");
        sc->add_option("--k", o.toy.k, "Generaciones por decisión");
        sc->add_flag("--sin-estado", sin_estado, "Variante cuyo óptimo es una P fija (control)");
    };

    auto* cal = app.add_subcommand("calibrar", "Mejor P fija y política oráculo");
    agregarToy(cal);
    cal->add_option("--candidatos", n_cand);
    cal->add_option("--corridas-busqueda", r_busqueda);
    cal->add_option("--corridas-final", r_final);
    cal->add_option("--corridas-prueba", r_test);
    cal->add_option("--semilla", semilla_cal);

    auto* ev = app.add_subcommand("evaluar", "Score de una P fija");
    agregarToy(ev);
    ev->add_option("--p", p_txt, "P separada por comas")->required();
    ev->add_option("--corridas", r_eval);

    auto agregarEntrenar = [&](CLI::App* sc) {
        agregarToy(sc);
        sc->add_option("--variante", o.variante, "base | g099 | l090 | l099 | l100 | l100e4 | g099l100 | g099l100e20 | g099l100c300 | n8 | n32");
        sc->add_option("--calentar", o.calentar, "Iteraciones iniciales con el actor congelado");
        sc->add_option("--corridas", o.corridas, "Corridas de entrenamiento");
        sc->add_option("--semilla", o.semilla, "Semilla de PPO");
        sc->add_option("--semilla-corridas", o.semilla_corridas, "Primera semilla de las corridas de entrenamiento");
        sc->add_option("--p0", p0_txt, "P0 separada por comas")->required();
        sc->add_option("--c", o.c, "Cota de la desviación de cada logit");
        sc->add_option("--w", o.w, "Peso del área bajo la curva en la recompensa");
        sc->add_option("--escala", o.escala, "Escala de la recompensa");
        sc->add_option("--std0", o.std0, "Desviación estándar inicial de la acción");
        sc->add_option("--lr", o.lr, "Tasa de aprendizaje de Adam (actor y crítico)");
        sc->add_option("--lr-critico", o.lr_critico, "Tasa de aprendizaje del crítico, si distinta");
        sc->add_option("--ultima-capa", o.ultima_capa, "Factor de la última capa del actor al iniciar");
        sc->add_option("--eval-corridas", o.eval_corridas, "Corridas de evaluación (múltiplo de N)");
        sc->add_option("--eval-cada", o.eval_cada, "Corridas de entrenamiento entre evaluaciones");
        sc->add_option("--csv", o.csv, "Registro por iteración");
    };
    auto* en = app.add_subcommand("entrenar", "PPO sobre el AOS sintético");
    agregarEntrenar(en);
    en->add_flag("--barajar", o.barajar, "Control negativo: recompensas barajadas antes de GAE");
    auto* ve = app.add_subcommand("verificar", "Contabilidad del runner (V2)");
    agregarEntrenar(ve);

    CLI11_PARSE(app, argc, argv);
    o.toy.con_estado = !sin_estado;
    try {
        if (*cal) return calibrar(o.toy, n_cand, r_busqueda, r_final, r_test, semilla_cal);
        if (*ev) {
            const auto s = aos::toy::evalStatic(parseP(p_txt), o.toy, 700001, r_eval);
            std::printf("P=(%s)  score %.4f (DE %.4f, %d corridas)\n", joinP(parseP(p_txt)).c_str(), media(s), desv(s), r_eval);
            return 0;
        }
        o.P0 = parseP(p0_txt);
        if (*ve) {
            o.verificar = true;
            return despachar(o);
        }
        return despachar(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
