// Adaptador genérico de entornos de RLtools a envs::ControlEnv (uso interno de src/envs).
// Requiere incluir antes los encabezados de RLtools de los entornos que se instancien.
#ifndef ENVS_RLTOOLS_ADAPTER_HPP
#define ENVS_RLTOOLS_ADAPTER_HPP

#include <rl_tools/operations/cpu.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "control_env.hpp"

namespace envs {
namespace rlt = rl_tools;
namespace E = rlt::rl::environments;

using T = double;
using TI = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using RNG = rlt::devices::random::CPU::ENGINE<>;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

// Normalización de una dimensión de la observación: x' = clamp((wrap(x) − center) / scale, −1, 1).
// `angle` envuelve x a (−π, π] antes de normalizar (ángulos sin cota).
struct ObsDim {
    double center = 0.0, scale = 1.0;
    bool angle = false;
};
inline ObsDim unit() { return {}; }
inline ObsDim sym(double bound) { return {0.0, bound, false}; }              // [−b, b]
inline ObsDim range(double lo, double hi) { return {(lo + hi) / 2, (hi - lo) / 2, false}; }
inline ObsDim angle() { return {0.0, M_PI, true}; }

// Muestreo del estado inicial: por defecto el de RLtools; un adaptador puede especializar este trait
// antes de instanciar RLToolsEnv con su entorno (ver Car en rltools_envs.cpp).
template <typename ENV>
struct InitialState {
    static void sample(DEVICE& device, ENV& env, typename ENV::Parameters& params, typename ENV::State& state, RNG& rng) {
        rlt::sample_initial_state(device, env, params, state, rng);
    }
};

// Adaptador genérico: dispositivo, RNG, entorno, parámetros y estado de RLtools.
template <typename ENV, typename OBS_TYPE>
class RLToolsEnv : public ControlEnv {
    static constexpr TI OBS_DIM = OBS_TYPE::DIM;
    static constexpr TI ACT_DIM = ENV::ACTION_DIM;
    using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS_DIM, false>>;
    using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACT_DIM, false>>;

    DEVICE device;
    RNG rng;
    ENV env;
    typename ENV::Parameters params;
    typename ENV::State state;
    OBS_TYPE obs_type;

    std::string name_;
    std::vector<ObsDim> obs_;   // OBS_DIM entradas
    int max_steps_, agents_;
    double action_scale_;       // multiplica la acción antes de pasarla al entorno (torque_scale)
    bool non_negative_;

public:
    RLToolsEnv(std::string name, std::vector<ObsDim> obs, int max_steps, int agents = 1, double action_scale = 1.0,
               bool non_negative = false)
        : name_(std::move(name)), obs_(std::move(obs)), max_steps_(max_steps), agents_(agents),
          action_scale_(action_scale), non_negative_(non_negative) {
        if (obs_.size() != OBS_DIM) throw std::logic_error(name_ + ": normalización con dimensión distinta a la observación");
        if (ACT_DIM % agents_ != 0 || OBS_DIM % agents_ != 0) throw std::logic_error(name_ + ": dimensiones no divisibles por agentes");
        rlt::init(device, rng, 0);
        rlt::malloc(device, env);
        rlt::init(device, env);
        rlt::initial_parameters(device, env, params);
    }
    ~RLToolsEnv() override { rlt::free(device, env); }

    std::string name() const override { return name_; }
    int obsDim() const override { return static_cast<int>(OBS_DIM) / agents_; }
    int actionDim() const override { return static_cast<int>(ACT_DIM) / agents_; }
    int numAgents() const override { return agents_; }
    bool nonNegativeObs() const override { return non_negative_; }
    int maxSteps() const override { return max_steps_; }

    void reset(uint64_t seed) override {
        rlt::init(device, rng, static_cast<typename DEVICE::index_t>(seed));
        rlt::sample_initial_parameters(device, env, params, rng);
        InitialState<ENV>::sample(device, env, params, state, rng);
    }

    std::vector<double> rawObserve() override {
        ObsMatrix m;
        rlt::observe(device, env, params, state, obs_type, m, rng);
        std::vector<double> o(OBS_DIM);
        for (TI i = 0; i < OBS_DIM; i++) o[i] = rlt::get(m, 0, i);
        return o;
    }

    std::vector<double> observe() override {
        std::vector<double> o = rawObserve();
        for (TI i = 0; i < OBS_DIM; i++) {
            double x = o[i];
            if (obs_[i].angle) x = std::remainder(x, 2.0 * M_PI);
            o[i] = std::clamp((x - obs_[i].center) / obs_[i].scale, -1.0, 1.0);
        }
        return o;
    }

    StepResult step(const std::vector<double>& action) override {
        ActMatrix a;
        for (TI j = 0; j < ACT_DIM; j++)
            rlt::set(a, 0, j, std::clamp(action.at(j), -1.0, 1.0) * action_scale_);
        typename ENV::State next;
        rlt::step(device, env, params, state, a, next, rng);
        double r = rlt::reward(device, env, params, state, a, next, rng);
        state = next;
        return {r, rlt::terminated(device, env, params, state, rng)};
    }
};

template <typename ENV, typename OBS = typename ENV::Observation, typename... A>
std::unique_ptr<ControlEnv> make(A&&... args) {
    return std::make_unique<RLToolsEnv<ENV, OBS>>(std::forward<A>(args)...);
}

}  // namespace envs

#endif
