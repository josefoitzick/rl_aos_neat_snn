// acrobot (clásico, recompensa −1 por paso hasta subir la punta) con torque ±TESIS_ACROBOT_TORQUE.
// Nuestro acrobot_t5 = parámetros por defecto (±5); acrobot_t1 = torque_scale 0,2. Como la acción se mapea
// linealmente a [MIN_TORQUE, MAX_TORQUE] y la recompensa no depende de la acción, ±1 equivale exactamente
// a escalar la acción por 0,2.
// Algoritmo: SAC de acrobot-swingup-v0 (zoo), con el horizonte nuestro (500 pasos; el zoo usa 20 s / DT).
#include "comun.h"
#ifdef TESIS_REF_STAGE2
#ifndef RL_TOOLS_RL_ZOO_ALGORITHM_SAC
#error "acrobot: solo SAC"
#endif
namespace tesis_ref {
using T = typename TYPE_POLICY::DEFAULT;
struct AcrobotParams : rlt::rl::environments::acrobot::DefaultParameters<T> {
    static constexpr T MIN_TORQUE = -TESIS_ACROBOT_TORQUE;
    static constexpr T MAX_TORQUE = +TESIS_ACROBOT_TORQUE;
};
using ENVIRONMENT = rlt::rl::environments::Acrobot<rlt::rl::environments::acrobot::Specification<T, TI, AcrobotParams>>;
using ZOO = rlt::rl::zoo::acrobot_swingup_v0::sac::FACTORY<DEVICE, TYPE_POLICY, TI, RNG, DYNAMIC_ALLOCATION>;
struct PARAMETERS : ZOO::LOOP_CORE_PARAMETERS {
    static constexpr TI EPISODE_STEP_LIMIT = 500;
};
}  // namespace tesis_ref
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, tesis_ref::ENVIRONMENT,
    tesis_ref::PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsMLP, DYNAMIC_ALLOCATION>;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES : BASE {};
#endif
