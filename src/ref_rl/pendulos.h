// Variantes de pendulum sin entrenador en el zoo (o sin versión con memoria). TESIS_PENDULO elige el entorno:
//   1 = PendulumMultiTask (nuestro pendulum-multitask; el "pendulum-multitask-v1" del zoo es PendulumMeta)
//   2 = PendulumVelocity  (solo velocidad: requiere memoria)
//   3 = PendulumPosition  (solo posición; el zoo trae SAC/TD3/PPO sin memoria; aquí se agrega SAC-GRU)
// Algoritmos (configuraciones del zoo más cercanas, horizonte 200 = el nuestro):
//   SAC → SAC-GRU de pendulum-multitask-v1 (la única del zoo con memoria para péndulos);
//   TD3 / PPO → pendulum-multitask-v1 (para 1) o pendulum-position-v1 (para 2: observación parcial).
#include "comun.h"
#ifdef TESIS_REF_STAGE2
namespace tesis_ref {
using T = typename TYPE_POLICY::DEFAULT;
using SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
#if TESIS_PENDULO == 1
using ENVIRONMENT = rlt::rl::environments::PendulumMultiTask<SPEC>;
namespace zoo_mlp = rlt::rl::zoo::pendulum_multitask_v1;
#elif TESIS_PENDULO == 2
using ENVIRONMENT = rlt::rl::environments::PendulumVelocity<SPEC>;
namespace zoo_mlp = rlt::rl::zoo::pendulum_position_v1;
#elif TESIS_PENDULO == 3
using ENVIRONMENT = rlt::rl::environments::PendulumPosition<SPEC>;
namespace zoo_mlp = rlt::rl::zoo::pendulum_position_v1;
#else
#error "TESIS_PENDULO debe ser 1, 2 o 3"
#endif
}  // namespace tesis_ref

#if defined(RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
namespace tesis_ref {
using ZOO = rlt::rl::zoo::pendulum_multitask_v1::sac::FACTORY<DEVICE, TYPE_POLICY, TI, RNG, DYNAMIC_ALLOCATION>;
}
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, tesis_ref::ENVIRONMENT,
    tesis_ref::ZOO::LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsGRU, DYNAMIC_ALLOCATION>;
template <typename BASE>
using LOOP_EVALUATION_PARAMETER_OVERWRITES = tesis_ref::ZOO::LOOP_EVALUATION_PARAMETER_OVERWRITES<BASE>;
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
namespace tesis_ref {
using ZOO = zoo_mlp::td3::FACTORY<DEVICE, TYPE_POLICY, TI, RNG, DYNAMIC_ALLOCATION>;
}
using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<TYPE_POLICY, TI, RNG, tesis_ref::ENVIRONMENT,
    tesis_ref::ZOO::LOOP_CORE_PARAMETERS, rlt::rl::algorithms::td3::loop::core::ConfigApproximatorsMLP, DYNAMIC_ALLOCATION>;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES : BASE {};
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
namespace tesis_ref {
using ZOO = zoo_mlp::ppo::FACTORY<DEVICE, TYPE_POLICY, TI, RNG, DYNAMIC_ALLOCATION>;
}
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, tesis_ref::ENVIRONMENT,
    tesis_ref::ZOO::LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, DYNAMIC_ALLOCATION>;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES : BASE {};
#endif
#endif
