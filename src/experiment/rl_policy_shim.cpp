// Envoltorio C de una política exportada por el zoo de RLtools (checkpoint.h, código C++ con los pesos).
// Se compila UNA vez por checkpoint como biblioteca compartida con símbolos ocultos
// (scripts/compilar_politica_rl.sh), para que sus funciones no inline de RLtools no choquen con las de
// rltools_envs.cpp (la única unidad que incluye RLtools en el resto del proyecto). rl_ref_eval la carga con dlopen.
//
// RL_POLICY_CHECKPOINT: ruta absoluta del checkpoint.h (definida al compilar).
#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>

#include RL_POLICY_CHECKPOINT

namespace rlt = rl_tools;

namespace {
using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
// Multiagente (bottleneck): el zoo exporta un multi_agent_wrapper cuya red interna (content) es la política
// compartida por agente. Se evalúa esa red con la observación de UN agente, como la pide rl_ref_eval.
template <typename M>
struct Interna {
    using TYPE = M;
    static constexpr const M& de(const M& m) { return m; }
};
template <typename SPEC>
struct Interna<rlt::nn_models::multi_agent_wrapper::ModuleForward<SPEC>> {
    using TYPE = typename SPEC::MODEL;
    static constexpr const TYPE& de(const rlt::nn_models::multi_agent_wrapper::ModuleForward<SPEC>& m) { return m.content; }
};
using EXPORTADO = rl_tools::checkpoint::actor::TYPE;
using FULL = typename Interna<EXPORTADO>::TYPE;
const FULL& red() { return Interna<EXPORTADO>::de(rl_tools::checkpoint::actor::module); }
using ACTOR = typename FULL::template CHANGE_BATCH_SIZE<TI, 1>;
using T = typename rl_tools::checkpoint::example::outputs::_0::SPEC::T;
constexpr TI OBS_DIM = ACTOR::INPUT_SHAPE::LAST;
constexpr TI ACT_DIM = ACTOR::OUTPUT_SHAPE::LAST;

// Paso a paso como rl::utils::evaluation de RLtools: evaluate_step con un estado de política (vacío en un MLP,
// estado oculto en una GRU) que se reinicia al comenzar cada episodio (rlp_reset).
struct Policy {
    DEVICE device;
    RNG rng;
    ACTOR::State<true> state;
    ACTOR::Buffer<true> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, OBS_DIM>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, ACT_DIM>>> output;
    Policy() {
        rlt::malloc(device, rng);
        rlt::init(device, rng, 0);
        rlt::malloc(device, state);
        rlt::malloc(device, buffer);
        rlt::malloc(device, input);
        rlt::malloc(device, output);
        rlt::reset(device, red(), state, rng);
    }
};
Policy& policy() {
    static Policy p;
    return p;
}
}  // namespace

#define RLP_EXPORT extern "C" __attribute__((visibility("default")))

RLP_EXPORT int rlp_obs_dim() { return static_cast<int>(OBS_DIM); }
RLP_EXPORT int rlp_act_dim() { return static_cast<int>(ACT_DIM); }
RLP_EXPORT const char* rlp_name() { return rl_tools::checkpoint::meta::name; }

// Inicio de episodio: reinicia el estado de la política (en una GRU, el estado oculto)
RLP_EXPORT void rlp_reset() {
    Policy& p = policy();
    rlt::reset(p.device, red(), p.state, p.rng);
}

// Acción determinista (modo evaluación: media de la política, aplastada a [-1, 1])
RLP_EXPORT void rlp_act(const double* obs, double* act) {
    Policy& p = policy();
    for (TI i = 0; i < OBS_DIM; i++) rlt::set(p.device, p.input, static_cast<T>(obs[i]), 0, i);
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::evaluate_step(p.device, red(), p.input, p.state, p.output, p.buffer, p.rng, mode);
    for (TI j = 0; j < ACT_DIM; j++) act[j] = static_cast<double>(rlt::get(p.device, p.output, 0, j));
}

// Control positivo: la red exportada completa (con el envoltorio multiagente, si lo hay) reproduce los
// ejemplos de entrada y salida guardados en el checkpoint. Devuelve el error absoluto medio.
RLP_EXPORT double rlp_selftest() {
    DEVICE device;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    EXPORTADO::Buffer<> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rl_tools::checkpoint::example::outputs::_0::SHAPE>> out;
    rlt::malloc(device, buffer);
    rlt::malloc(device, out);
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::evaluate(device, rl_tools::checkpoint::actor::module, rl_tools::checkpoint::example::inputs::_0::container,
                  out, buffer, rng, mode);
    const double d = rlt::abs_diff(device, rl_tools::checkpoint::example::outputs::_0::container, out) /
                     rl_tools::checkpoint::example::outputs::_0::SPEC::SIZE;
    rlt::free(device, buffer);
    rlt::free(device, out);
    return d;
}
