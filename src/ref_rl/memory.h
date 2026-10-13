// memory (bit por paso; recompensa −10·|u − (unos en la ventana de 10)/10|). Óptimo exacto = 0 (ver
// src/ref_rl/memory_oracle.cpp); este entrenador mide cuánto de ese óptimo alcanza RL con memoria.
// Algoritmo: SAC-GRU de pendulum-multitask-v1 (la configuración recurrente del zoo), con nuestro horizonte
// (20 pasos) y secuencias de entrenamiento de 20 (el zoo usa 40 con episodios de 200).
#include "comun.h"
#ifndef TESIS_REF_STAGE2
#include <rl_tools/rl/environments/memory/operations_cpu.h>   // incluye la genérica y json(env) para checkpoints
#else
#ifndef RL_TOOLS_RL_ZOO_ALGORITHM_SAC
#error "memory: solo SAC"
#endif
namespace tesis_ref {
using T = typename TYPE_POLICY::DEFAULT;
using ENVIRONMENT = rlt::rl::environments::Memory<rlt::rl::environments::memory::Specification<T, TI>>;
using ZOO = rlt::rl::zoo::pendulum_multitask_v1::sac::FACTORY<DEVICE, TYPE_POLICY, TI, RNG, DYNAMIC_ALLOCATION>;
struct PARAMETERS : ZOO::LOOP_CORE_PARAMETERS {
    struct SAC_PARAMETERS : ZOO::LOOP_CORE_PARAMETERS::SAC_PARAMETERS {
        static constexpr TI SEQUENCE_LENGTH = 20;
    };
    static constexpr TI EPISODE_STEP_LIMIT = 20;
};
}  // namespace tesis_ref
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, tesis_ref::ENVIRONMENT,
    tesis_ref::PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsGRU, DYNAMIC_ALLOCATION>;
template <typename BASE>
using LOOP_EVALUATION_PARAMETER_OVERWRITES = tesis_ref::ZOO::LOOP_EVALUATION_PARAMETER_OVERWRITES<BASE>;
#endif
