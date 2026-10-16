#ifndef EXPERIMENT_PPO_LOTE_HPP
#define EXPERIMENT_PPO_LOTE_HPP

// Lote propio de entornos para el runner on-policy de RLtools (docs/diseno_ppo.md §3.2).
//
// El runner de RLtools es genérico en el tipo de lote, pero el único que trae (`batch::Independent`) avanza
// las instancias una por una, y el núcleo del loop de PPO lo fija en su configuración. Aquí:
//   - `Entorno` es el tipo de instancia (solo definiciones). Su `State` es POD, porque el runner copia los
//     estados por valor: {ranura, decisión t, Φ}. El estado de verdad (la corrida) vive en el pool.
//   - `Lote` apunta a un pool de corridas con la interfaz start / advance / obs / phi / t / T / reward.
//     `step` traduce las acciones a P y avanza **todas** las ranuras en una sola llamada al pool, que puede
//     paralelizar como quiera.
//   - `ConfigConLote` deriva la configuración del núcleo (`ppo::loop::core::Config`) y reemplaza el lote y los
//     tipos del runner que dependen de él. El `State` del núcleo y sus funciones son genéricos en la config.
// La acción a se convierte en P con aos::actionToP (la misma función del despliegue).
//
// Requiere incluir antes las operaciones de RLtools (rl_tools/operations/cpu_mux.h y las de nn/). Si la unidad
// envuelve RLtools en un espacio de nombres propio (RL_TOOLS_NAMESPACE_WRAPPER, para no chocar con otra unidad
// que también lo incluye, como libenvs), todo este archivo queda dentro del mismo envoltorio.
#include <functional>
#include <type_traits>
#include <vector>
#include "aos/action_map.hpp"

#ifndef RL_TOOLS_NAMESPACE_WRAPPER_START
#error "ppo_lote.hpp: incluir antes las operaciones de RLtools"
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace ppo_lote {

template <typename T_T, typename T_TI, T_TI T_OBS_DIM, T_TI T_ACTION_DIM, T_TI T_STEP_LIMIT = 100000>
struct Entorno {
    using T = T_T;
    using TI = T_TI;
    struct Parameters {};
    struct State {
        TI slot;
        TI t;
        T phi;
    };
    struct Observation {
        static constexpr TI DIM = T_OBS_DIM;
        using SHAPE = rl_tools::tensor::Shape<TI, DIM>;
    };
    // El crítico ve además Φ (información privilegiada, solo al entrenar; §2.3)
    struct ObservationPrivileged {
        static constexpr TI DIM = T_OBS_DIM + 1;
        using SHAPE = rl_tools::tensor::Shape<TI, DIM>;
    };
    static constexpr TI N_AGENTS = 1;
    static constexpr TI ACTION_DIM = T_ACTION_DIM;
    // Mayor que T: el episodio termina por `terminated` (t = T), nunca por truncamiento
    static constexpr TI EPISODE_STEP_LIMIT = T_STEP_LIMIT;
};

template <typename T_ENVIRONMENT, typename T_ENVIRONMENT::TI T_INSTANCES, typename T_POOL>
struct LoteSpec {
    using ENVIRONMENT = T_ENVIRONMENT;
    using TI = typename ENVIRONMENT::TI;
    using POOL = T_POOL;
    static constexpr TI INSTANCES = T_INSTANCES;
};

template <typename T_SPEC>
struct Lote {
    using SPEC = T_SPEC;
    using ENVIRONMENT = typename SPEC::ENVIRONMENT;
    using T = typename ENVIRONMENT::T;
    using TI = typename ENVIRONMENT::TI;
    using Parameters = typename ENVIRONMENT::Parameters;
    using State = typename ENVIRONMENT::State;
    using Observation = typename ENVIRONMENT::Observation;
    using ObservationPrivileged = typename ENVIRONMENT::ObservationPrivileged;
    static constexpr TI INSTANCES = SPEC::INSTANCES;
    static constexpr TI N_AGENTS = ENVIRONMENT::N_AGENTS;
    static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;

    typename SPEC::POOL* pool = nullptr;
    std::vector<double> logp0;   // log P₀ con piso (aos::logP0)
    double c = 3.0;              // cota de la desviación de cada logit
    // Plan B: si está, la acción se decodifica con este mapa (P y/o intensidades de variación) en vez de logp0 y c
    const aos::MapaAccion* mapa = nullptr;
    // PPO residual: si está, suma a la acción de la ranura la de una política base (la regla escrita a mano) antes de
    // decodificar; con acción 0 del agente, la decisión es exactamente la de la base
    std::function<void(int, double*)> residual;
};

// Configuración del núcleo de PPO con nuestro lote en lugar de batch::Independent
template <typename BASE, typename T_LOTE>
struct ConfigConLote : BASE {
    using BATCH_ENVIRONMENT = T_LOTE;
    using ON_POLICY_RUNNER_SPEC = rl_tools::rl::components::on_policy_runner::Specification<
        typename BASE::TYPE_POLICY, BATCH_ENVIRONMENT,
        typename BASE::NN::ACTOR_TYPE::template State<BASE::DYNAMIC_ALLOCATION>,
        typename BASE::ENVIRONMENT::Observation, typename BASE::ENVIRONMENT::ObservationPrivileged,
        typename BASE::T, typename BASE::T, BASE::CORE_PARAMETERS::EPISODE_STEP_LIMIT,
        BASE::CORE_PARAMETERS::PPO_PARAMETERS::TRUNCATE_ON_EACH_ITERATION, BASE::DYNAMIC_ALLOCATION,
        BASE::CORE_PARAMETERS::PPO_PARAMETERS::BOOTSTRAP_TRUNCATIONS || BASE::CORE_PARAMETERS::PPO_PARAMETERS::IGNORE_TERMINATION>;
    using ON_POLICY_RUNNER_TYPE = rl_tools::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
    using ON_POLICY_RUNNER_BUFFER_TYPE = rl_tools::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_SPEC>;
    using ON_POLICY_RUNNER_DATASET_SPEC = rl_tools::rl::components::on_policy_runner::DatasetSpecification<
        ON_POLICY_RUNNER_SPEC, BASE::CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV, BASE::DYNAMIC_ALLOCATION>;
    using ON_POLICY_RUNNER_DATASET_TYPE = rl_tools::rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;
    using CRITIC_COLLECTION_STATE =
        rl_tools::rl::components::on_policy_runner::ValueState<typename BASE::NN::CRITIC_TYPE, ON_POLICY_RUNNER_DATASET_SPEC>;
    using CRITIC_COLLECTION_BUFFER =
        rl_tools::rl::components::on_policy_runner::ValueBuffer<typename BASE::NN::CRITIC_TYPE, ON_POLICY_RUNNER_DATASET_SPEC>;
};

}  // namespace ppo_lote

// Operaciones del lote que llama el runner (on_policy_runner/operations_generic.h). Están en el espacio de
// nombres de RLtools; las encuentra ADL por los argumentos Tensor.
namespace rl_tools {

template <typename DEVICE, typename SPEC>
void malloc(DEVICE&, ppo_lote::Lote<SPEC>&) {}

template <typename DEVICE, typename SPEC>
void free(DEVICE&, ppo_lote::Lote<SPEC>&) {}

template <typename DEVICE, typename SPEC>
void init(DEVICE&, ppo_lote::Lote<SPEC>& lote) {
    if (!lote.pool) throw std::logic_error("ppo_lote: lote sin pool");
    const auto dim = lote.mapa ? lote.mapa->dim() : static_cast<int>(lote.logp0.size());
    if (static_cast<typename SPEC::TI>(dim) != ppo_lote::Lote<SPEC>::ACTION_DIM)
        throw std::logic_error("ppo_lote: la acción del mapa no tiene ACTION_DIM dimensiones");
}

template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
void sample_initial_parameters(DEVICE&, ppo_lote::Lote<SPEC>&, Tensor<PARAMETER_SPEC>&, const Tensor<RESET_SPEC>&, RNG&) {}

// Nuevas corridas (siguiente semilla) en las ranuras marcadas; el pool las inicia todas juntas
template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
void sample_initial_state(DEVICE& device, ppo_lote::Lote<SPEC>& lote, Tensor<PARAMETER_SPEC>&, Tensor<STATE_SPEC>& states,
                          const Tensor<RESET_SPEC>& reset, RNG&) {
    using TI = typename SPEC::TI;
    std::vector<int> slots;
    for (TI i = 0; i < SPEC::INSTANCES; i++)
        if (get(device, reset, i)) slots.push_back(static_cast<int>(i));
    if (slots.empty()) return;
    lote.pool->start(slots);
    for (int i : slots) {
        auto& s = get_ref(device, states, i);
        s.slot = i;
        s.t = 0;
        s.phi = static_cast<typename SPEC::ENVIRONMENT::T>(lote.pool->phi(i));
    }
}

// Una decisión para todas las ranuras: a → P (aos::actionToP) o, con mapa, a → P e intensidades; el pool avanza k
// generaciones en cada una
template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC,
          typename NEXT_STATE_SPEC, typename RNG>
void step(DEVICE& device, ppo_lote::Lote<SPEC>& lote, Tensor<PARAMETER_SPEC>&, Tensor<STATE_SPEC>& states,
          const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG&) {
    using TI = typename SPEC::TI;
    using T = typename SPEC::ENVIRONMENT::T;
    constexpr TI N = SPEC::INSTANCES;
    constexpr TI K = SPEC::ENVIRONMENT::ACTION_DIM;
    std::vector<std::vector<double>> P(N);
    std::vector<neat::Variacion> V(lote.mapa ? N : 0);
    for (TI i = 0; i < N; i++) {
        double a[K];
        for (TI j = 0; j < K; j++) a[j] = static_cast<double>(get(device, actions, i, j));
        if (lote.residual) lote.residual(static_cast<int>(i), a);
        if (lote.mapa)
            lote.mapa->decodificar(a, P[i], V[i]);
        else
            P[i] = aos::actionToP(a, lote.logp0, lote.c);
    }
    lote.pool->advance(P, lote.mapa && lote.mapa->mueveIntensidades() ? &V : nullptr);
    for (TI i = 0; i < N; i++) {
        const auto& s = get_ref(device, states, i);
        auto& ns = get_ref(device, next_states, i);
        if (s.slot != i) throw std::logic_error("ppo_lote::step: estado en una ranura ajena");
        ns.slot = s.slot;
        ns.t = s.t + 1;
        if (lote.pool->t(static_cast<int>(i)) != static_cast<int>(ns.t))
            throw std::logic_error("ppo_lote::step: la decisión del pool no coincide con la del estado");
        ns.phi = static_cast<T>(lote.pool->phi(static_cast<int>(i)));
    }
}

// r_t = escala · [(Φ_{t+1} − Φ_t) + (w/T)·Φ_{t+1}], solo con los campos POD de los estados
template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC,
          typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
void reward(DEVICE& device, ppo_lote::Lote<SPEC>& lote, Tensor<PARAMETER_SPEC>&, Tensor<STATE_SPEC>& states,
            const Tensor<ACTION_SPEC>&, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG&) {
    using TI = typename SPEC::TI;
    for (TI i = 0; i < SPEC::INSTANCES; i++) {
        const double r = lote.pool->reward(get_ref(device, states, i).phi, get_ref(device, next_states, i).phi);
        set(device, rewards, static_cast<typename SPEC::ENVIRONMENT::T>(r), i);
    }
}

template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
void terminated(DEVICE& device, ppo_lote::Lote<SPEC>& lote, Tensor<PARAMETER_SPEC>&, Tensor<STATE_SPEC>& states,
                Tensor<TERMINATED_SPEC>& flags, RNG&) {
    using TI = typename SPEC::TI;
    for (TI i = 0; i < SPEC::INSTANCES; i++)
        set(device, flags, static_cast<int>(get_ref(device, states, i).t) >= lote.pool->T(), i);
}

// Rasgos guardados de cada corrida; la versión privilegiada (crítico) agrega Φ al final
template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION,
          typename OBSERVATION_SPEC, typename RNG>
void observe(DEVICE& device, ppo_lote::Lote<SPEC>& lote, Tensor<PARAMETER_SPEC>&, Tensor<STATE_SPEC>&, const OBSERVATION,
             Tensor<OBSERVATION_SPEC>& observations, RNG&) {
    using TI = typename SPEC::TI;
    using T = typename SPEC::ENVIRONMENT::T;
    using ENV = typename SPEC::ENVIRONMENT;
    constexpr bool PRIVILEGED = std::is_same_v<OBSERVATION, typename ENV::ObservationPrivileged>;
    static_assert(PRIVILEGED || std::is_same_v<OBSERVATION, typename ENV::Observation>, "ppo_lote::observe: tipo de observación desconocido");
    constexpr TI D = ENV::Observation::DIM;
    for (TI i = 0; i < SPEC::INSTANCES; i++) {
        const auto& o = lote.pool->obs(static_cast<int>(i));
        for (TI j = 0; j < D; j++) set(device, observations, static_cast<T>(o[j]), i, j);
        if constexpr (PRIVILEGED) set(device, observations, static_cast<T>(lote.pool->phi(static_cast<int>(i))), i, D);
    }
}

}  // namespace rl_tools
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
