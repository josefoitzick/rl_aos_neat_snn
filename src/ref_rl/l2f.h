// l2f completo (obs 82, horizonte 500). El zoo compila sac_tiny.h para el slot L2F+SAC; aquí se usa la
// configuración de src/rl/zoo/l2f/sac.h (la del entorno completo, que el zoo no incluye), copiada tal cual.
// Compilar con RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F definida para conservar el curriculum de l2f del zoo
// (guidance según dificultad, zoo.cpp, bloque L2F && SAC).
#include "comun.h"
#ifdef TESIS_REF_STAGE2
#ifndef RL_TOOLS_RL_ZOO_ALGORITHM_SAC
#error "l2f: solo SAC"
#endif
namespace tesis_ref {
using ENVIRONMENT = typename rlt::rl::zoo::l2f::ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI>::ENVIRONMENT;
struct PARAMETERS : rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    using T = typename TYPE_POLICY::DEFAULT;
    struct SAC_PARAMETERS : rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI> {
        static constexpr TI ACTOR_BATCH_SIZE = 64;
        static constexpr TI CRITIC_BATCH_SIZE = 64;
        static constexpr TI TRAINING_INTERVAL = 5;
        static constexpr TI CRITIC_TRAINING_INTERVAL = 1 * TRAINING_INTERVAL;
        static constexpr TI ACTOR_TRAINING_INTERVAL = 2 * TRAINING_INTERVAL;
        static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1 * TRAINING_INTERVAL;
        static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.9;
        static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.3;
        static constexpr T GAMMA = 0.99;
        static constexpr bool IGNORE_TERMINATION = false;
        static constexpr T TARGET_ENTROPY = -((T)4);
        static constexpr TI SEQUENCE_LENGTH = 1;
    };
    static constexpr TI STEP_LIMIT = 200000;
    static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    struct OPTIMIZER_PARAMETERS_COMMON : rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY> {
        static constexpr bool ENABLE_GRADIENT_CLIPPING = true;
        static constexpr T GRADIENT_CLIP_VALUE = 1;
        static constexpr bool ENABLE_WEIGHT_DECAY = true;
        static constexpr T WEIGHT_DECAY = 0.0001;
    };
    struct ACTOR_OPTIMIZER_PARAMETERS : OPTIMIZER_PARAMETERS_COMMON {
        static constexpr T ALPHA = 1e-3;
    };
    struct CRITIC_OPTIMIZER_PARAMETERS : OPTIMIZER_PARAMETERS_COMMON {
        static constexpr T ALPHA = 1e-3;
    };
    struct ALPHA_OPTIMIZER_PARAMETERS : OPTIMIZER_PARAMETERS_COMMON {
        static constexpr T ALPHA = 1e-3;
    };
    static constexpr bool SAMPLE_ENVIRONMENT_PARAMETERS = true;
};
}  // namespace tesis_ref
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, tesis_ref::ENVIRONMENT,
    tesis_ref::PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsMLP, DYNAMIC_ALLOCATION>;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES : BASE {};
#endif
