// car (sin pista, horizonte 500) y car-track (pista 100×100 a escala 20, horizonte 1000); TESIS_CAR_TRACK = 0 / 1.
// Algoritmo: el entrenador PPO de RLtools para CarTrack (src/rl/environments/car/training_ppo.h, la misma pista
// que nuestro car-track), con sus hiperparámetros pasados a la API actual. El car sin pista no tiene entrenador
// en RLtools: se usa el mismo. Sin TD3: el runner off-policy exige ENVIRONMENT::EPISODE_STEP_LIMIT, que Car no
// define (training_td3.h usa una API anterior).
#include "comun.h"
#ifndef TESIS_REF_STAGE2
#include <rl_tools/rl/environments/car/operations_generic.h>
#include <rl_tools/rl/environments/car/operations_json.h>   // json(env) para checkpoints y trayectorias
// Bug de RLtools (car/operations_generic.h:74): sample_initial_state de Car llama a initial_state sin
// `parameters` y no compila. Esta sobrecarga (más específica) replica su intención, como
// envs::InitialState<Car> en src/envs/rltools_envs.cpp: x, y ~ U(±0,2), orientación ~ U(±π), resto 0.
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
template <typename DEVICE, typename T, typename TI, typename RNG>
RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(
    DEVICE& device, const rl::environments::Car<rl::environments::car::Specification<T, TI>>& env,
    typename rl::environments::Car<rl::environments::car::Specification<T, TI>>::Parameters& parameters,
    typename rl::environments::Car<rl::environments::car::Specification<T, TI>>::State& state, RNG& rng) {
    initial_state(device, env, parameters, state);
    state.x = random::uniform_real_distribution(device.random, (T)-0.2, (T)0.2, rng);
    state.y = random::uniform_real_distribution(device.random, (T)-0.2, (T)0.2, rng);
    state.mu = random::uniform_real_distribution(device.random, -math::PI<T>, math::PI<T>, rng);
}
// json(device, env): descripción del entorno que el zoo guarda junto a cada checkpoint (solo metadato).
// RLtools no la define para Car/CarTrack.
template <typename DEVICE, typename SPEC>
std::string json(DEVICE&, rl::environments::Car<SPEC>&) { return "{}"; }
}  // namespace rl_tools
RL_TOOLS_NAMESPACE_WRAPPER_END
#else
namespace tesis_ref {
using T = typename TYPE_POLICY::DEFAULT;
#if TESIS_CAR_TRACK
using ENVIRONMENT = rlt::rl::environments::CarTrack<rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 20>>;
constexpr TI HORIZONTE = 1000;
#else
using ENVIRONMENT = rlt::rl::environments::Car<rlt::rl::environments::car::Specification<T, TI>>;
constexpr TI HORIZONTE = 500;
#endif
}  // namespace tesis_ref

#if defined(RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
namespace tesis_ref {
struct PARAMETERS : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    static constexpr TI BATCH_SIZE = 512;
    struct PPO_PARAMETERS : rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE> {
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.001;
        static constexpr TI N_EPOCHS = 2;
        static constexpr T GAMMA = 0.995;
        static constexpr T LAMBDA = 0.975;
        static constexpr bool ADAPTIVE_LEARNING_RATE = false;
    };
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 512;
    static constexpr TI N_ENVIRONMENTS = 32;
    static constexpr TI TOTAL_STEP_LIMIT = 10000000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = HORIZONTE;
};
}  // namespace tesis_ref
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, tesis_ref::ENVIRONMENT,
    tesis_ref::PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, DYNAMIC_ALLOCATION>;
#else
#error "car: solo PPO"
#endif
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES : BASE {};
#endif
