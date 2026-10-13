// Adaptadores de los entornos de RLtools (rama render @ 6bb2fda) a envs::ControlEnv.
// RLtools define funciones no-inline en sus encabezados: todos los entornos van en esta única unidad.
// Inventario, recompensas y criterios: docs/entornos_rltools.md.
#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
// car: solo la versión genérica. La de CPU carga la pista desde un .bmp relativo al directorio de
// trabajo; la genérica la copia de la pista por defecto compilada en track.h.
#include <rl_tools/rl/environments/car/operations_generic.h>
#include <rl_tools/rl/environments/flag/operations_generic.h>
#include <rl_tools/rl/environments/memory/operations_generic.h>
#include <rl_tools/rl/environments/multi_agent/bottleneck/operations_generic.h>
#include <rl_tools/rl/environments/reacher/operations_generic.h>
#include <rl_tools/containers/matrix/persist_code.h>   // requerido por l2f/persist_code.h
#include <rl_tools/numeric_types/persist_code.h>
#include <rl/zoo/l2f/environment_tiny.h>   // l2f y l2f-tiny del zoo (incluye environment.h)
#ifdef TESIS_HAVE_MUJOCO
#include <rl_tools/rl/environments/mujoco/ant/operations_cpu.h>
#endif
#ifdef TESIS_HAVE_RAYTRACING
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>
#include <rl_tools/rendering/datasets/annotations/operations_cpu.h>
#include <rl/zoo/l2f_visual/environment.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>
#include <cstdlib>
#include <filesystem>
#endif

#include <functional>
#include <map>
#include "rltools_adapter.hpp"

namespace envs {

// Car (sin pista): su sample_initial_state en RLtools (car/operations_generic.h:74) llama a
// initial_state sin el argumento `parameters` y no compila. Se replica su intención:
// x, y ~ U(±0,2 m), orientación ~ U(±π), resto 0.
using CarSpec = E::car::Specification<T, TI>;
template <>
struct InitialState<E::Car<CarSpec>> {
    static void sample(DEVICE& device, E::Car<CarSpec>&, E::Car<CarSpec>::Parameters&, E::Car<CarSpec>::State& state,
                       RNG& rng) {
        state = {};
        state.x = rlt::random::uniform_real_distribution(device.random, -0.2, 0.2, rng);
        state.y = rlt::random::uniform_real_distribution(device.random, -0.2, 0.2, rng);
        state.mu = rlt::random::uniform_real_distribution(device.random, -M_PI, M_PI, rng);
    }
};

// ── Pendulum (≈ Pendulum-v1): θ̇ ∈ [−8, 8]; torque ±2; 200 pasos; sin término ──
using PendulumSpec = E::pendulum::Specification<T, TI>;
// Acrobot: θ̇1 ∈ ±4π, θ̇2 ∈ ±9π; 500 pasos
using AcrobotSpec = E::acrobot::Specification<T, TI>;
// acrobot-swingup-v0 del zoo: DT 0,02 y torque ±5 sobre EasyParameters
struct AcrobotSwingupParams : E::acrobot::EasyParameters<T> {
    static constexpr T DT = 0.02;
    static constexpr T MIN_TORQUE = -5;
    static constexpr T MAX_TORQUE = +5;
};
using AcrobotSwingupSpec = E::acrobot::Specification<T, TI, AcrobotSwingupParams>;
// car y car-track: pista 100×100 a escala 20 (como los entrenadores de RLtools en src/rl/environments/car)
using CarTrackSpec = E::car::SpecificationTrack<T, TI, 100, 100, 20>;
// flag del zoo: 50 pasos, sin observación privilegiada
using FlagSpec = E::flag::Specification<T, TI, E::flag::DefaultParameters<T, TI, 50, false, false>>;
using MemorySpec = E::memory::Specification<T, TI>;
using ReacherSpec = E::reacher::Specification<T, TI>;
// reacher-visual(-memory)(-hard)-v0 del zoo: imagen 16×16×3 (canal 0: objetivo, canal 2: agente; ∈ [0, 1])
struct ReacherVisualParams : E::reacher::DefaultParameters<T> {
    static constexpr auto IMAGE_HEIGHT = 16;
    static constexpr auto IMAGE_WIDTH = 16;
};
struct ReacherVisualHardParams : ReacherVisualParams {
    static constexpr auto NUM_TARGETS = 2;
};
using ReacherVisualSpec = E::reacher::Specification<T, TI, ReacherVisualParams>;
using ReacherVisualHardSpec = E::reacher::Specification<T, TI, ReacherVisualHardParams>;
template <typename ENV>
static std::unique_ptr<ControlEnv> makeImage(const std::string& name) {
    const int dim = static_cast<int>(ENV::Observation::DIM);
    return make<ENV>(name, std::vector<ObsDim>(dim, unit()), static_cast<int>(ENV::EPISODE_STEP_LIMIT), 1, 1.0, true);
}
// bottleneck-v0 del zoo
struct BottleneckParams : E::multi_agent::bottleneck::DefaultParameters<T, TI> {
    static constexpr TI N_AGENTS = 5;
    static constexpr TI LIDAR_RESOLUTION = 5;
    static constexpr T LIDAR_FOV = M_PI * 90 / 180;
    static constexpr T BOTTLENECK_WIDTH = 5;
    static constexpr TI EPISODE_STEP_LIMIT = 200;
    static constexpr bool SPAWN_BOTH_SIDES = false;
    static constexpr T AGENT_MAX_SPEED = 4;
    static constexpr T AGENT_MAX_ANGULAR_VELOCITY = 4;
    static constexpr T AGENT_MAX_ACCELERATION = 20;
    static constexpr T AGENT_MAX_ANGULAR_ACCELERATION = 20;
    static constexpr T DT = 0.05;
};
using BottleneckSpec = E::multi_agent::bottleneck::Specification<TYPE_POLICY, TI, BottleneckParams>;

// Cotas de car (vx, vy, ω) sin límite analítico: máximo de |x| medido con env_probe bajo acciones
// aleatorias (car: 1,19 / 0,53 / 1,78), redondeado hacia arriba (docs/entornos_rltools.md §5).
// Una política que maneje más agresivo puede saturar: se recorta a ±1.
static std::vector<ObsDim> carObs() {
    return {sym(1), sym(1), angle(), sym(1.2), sym(0.6), sym(1.8)};
}

static std::vector<ObsDim> bottleneckObs() {
    std::vector<ObsDim> o;
    for (TI a = 0; a < BottleneckParams::N_AGENTS; a++) {
        o.push_back(range(0, BottleneckParams::ARENA_WIDTH));
        o.push_back(range(0, BottleneckParams::ARENA_HEIGHT));
        o.push_back(angle());
        o.push_back(sym(BottleneckParams::AGENT_MAX_SPEED));
        o.push_back(sym(BottleneckParams::AGENT_MAX_SPEED));
        o.push_back(sym(BottleneckParams::AGENT_MAX_ANGULAR_VELOCITY));
        o.push_back(unit());                                       // vivo = 1, muerto = −1
        // Distancia al obstáculo más cercano, muros del área incluidos (RLtools no usa LIDAR_RANGE):
        // cota = diagonal del área
        for (TI l = 0; l < BottleneckParams::LIDAR_RESOLUTION; l++)
            o.push_back(range(0, std::hypot(BottleneckParams::ARENA_WIDTH, BottleneckParams::ARENA_HEIGHT)));
    }
    return o;
}

// ── l2f (cuadricóptero Crazyflie) con las configuraciones del zoo ──
//   l2f      → src/rl/zoo/l2f/environment.h      (obs 82: posición, rotación, velocidades, 16 acciones previas)
//   l2f-tiny → src/rl/zoo/l2f/environment_tiny.h (obs 29: error de trayectoria, rotación, velocidades, 2 acciones previas)
// Cotas: posición y velocidades según la terminación del zoo (|p| > 1 m, |v| > 10 m/s, |ω| > 35 rad/s
// terminan el episodio); rotación y acciones previas ya están en [−1, 1].
static void push(std::vector<ObsDim>& o, int n, ObsDim d) { o.insert(o.end(), n, d); }

static std::unique_ptr<ControlEnv> makeL2f(const std::string& variant) {
    if (variant == "l2f") {
        using ENV = rlt::rl::zoo::l2f::ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI>::ENVIRONMENT;
        std::vector<ObsDim> o;
        push(o, 3, sym(1));      // posición
        push(o, 9, unit());      // matriz de rotación
        push(o, 3, sym(10));     // velocidad lineal
        push(o, 3, sym(35));     // velocidad angular
        push(o, static_cast<int>(ENV::Observation::DIM) - 18, unit());   // historial de acciones
        return make<ENV>("l2f", o, static_cast<int>(ENV::EPISODE_STEP_LIMIT));
    }
    if (variant == "l2f-tiny") {
        using ENV = rlt::rl::zoo::l2f::ENVIRONMENT_TINY_FACTORY<DEVICE, TYPE_POLICY, TI>::ENVIRONMENT;
        std::vector<ObsDim> o;
        push(o, 3, sym(1));      // error de posición respecto de la trayectoria
        push(o, 3, sym(10));     // error de velocidad
        push(o, 9, unit());
        push(o, 3, sym(10));
        push(o, 3, sym(35));
        push(o, static_cast<int>(ENV::Observation::DIM) - 21, unit());
        return make<ENV>("l2f-tiny", o, static_cast<int>(ENV::EPISODE_STEP_LIMIT));
    }
    throw std::invalid_argument("makeL2f: variante desconocida '" + variant + "'");
}

#ifdef TESIS_HAVE_MUJOCO
// ── ant-v4 (MuJoCo 2.3.5, como el zoo): obs = q sin x, y (13) + q̇ (14); 8 torques; 1000 pasos ──
// Cotas: máximo de |x| medido con env_probe bajo acciones aleatorias, redondeado hacia arriba
// (docs/entornos_rltools.md §5): altura del torso ∈ [0,2; 1] (fuera de ese rango el episodio termina),
// cuaternión ±1, ángulos de articulación ±1,4, velocidades del torso ±7, de articulación ±16.
using AntSpec = E::mujoco::ant::Specification<T, TI, E::mujoco::ant::DefaultParameters<T, TI>>;
static std::vector<ObsDim> antObs() {
    std::vector<ObsDim> o{range(0.2, 1.0)};
    o.insert(o.end(), 4, unit());
    o.insert(o.end(), 8, sym(1.4));
    o.insert(o.end(), 6, sym(7));
    o.insert(o.end(), 8, sym(16));
    return o;
}
#endif

#ifdef TESIS_HAVE_RAYTRACING
// ── l2f-visual: el cuadricóptero de l2f observando SOLO una cámara a bordo (raytracing en CPU) ──
// Configuración del zoo (src/rl/zoo/l2f_visual) con NUM_ENVS = 1 y cámara 16×16 RGB (768 entradas;
// 64×64 es inviable para NEAT: docs/entornos_rltools.md §6). Tarea en "modo objetivo", como el
// entrenador de CPU de RLtools (src/rl/environments/l2f_visual/training.cpp): la dinámica es la de l2f
// (estabilizar desde inicios de hasta 90°) y la casa se traslada para que el dron quede en una
// posición libre de la escena. Cada instancia carga las escenas de su partición (entrenamiento o test)
// con su propio renderizador; la casa de cada episodio es (semilla mod n.º de escenas).
namespace l2fv {
using TF = float;
using ENV = rlt::rl::zoo::l2f_visual::ENVIRONMENT_FACTORY<DEVICE, rlt::numeric_types::Policy<TF>, TI, 1, 16, 16>::ENVIRONMENT;
using RSPEC = typename ENV::SPEC::RENDERER_SPEC;
using Library = rlt::rendering::raytracing::AssetLibrary<RSPEC>;
using Renderer = rlt::rendering::raytracing::Renderer<RSPEC>;
using Annotations = rlt::rendering::datasets::annotations::FreeSpace<typename ENV::SPEC::ANNOTATIONS_SPEC>;
static constexpr TI OBS_DIM = ENV::Observation::DIM;
static constexpr TI ACT_DIM = ENV::ACTION_DIM;

// Escenas ProcTHOR del dataset rl-tools/conta (Hugging Face), por SHA-1. El dataset público trae solo
// 3 casas distintas, cada una en dos versiones (catastro: docs/entornos_rltools.md §6); el resto son objetos.
// Entrenamiento: las casas 0 y 1, las mismas que usa RLtools (hyperdrone/tasks/target_frame).
// Test: la casa 2 (no vista), en sus dos versiones.
static const std::vector<std::string> kTrainScenes = {
    "a8fda3e9872e29994a425bd35c425159598623ce",   // ProcTHOR-Train-0
    "7f1c9129532798e0b63bc41edb6b4c09251cf8a0",   // ProcTHOR-Train-1
};
static const std::vector<std::string> kTestScenes = {
    "44b5c137aaf4d6b4ed6a0d8ce57fa315c2331e67",   // casa 2, versión A (124 mallas, 9,4 × 5,7 m)
    "c6e7892a60f4d30d284dde82016086410b471385",   // casa 2, versión B
};

static std::string scenePath(const std::string& hash) {
    std::filesystem::path dir;
    if (const char* r = std::getenv("CONTA_ROOT")) dir = std::filesystem::path(r) / "data";
    else if (const char* c = std::getenv("CONTA_CACHE")) dir = c;
    else dir = std::filesystem::path(std::getenv("HOME")) / ".cache/rl_tools/conta";
    auto p = dir / hash;
    if (!std::filesystem::exists(p))
        throw std::runtime_error("l2f-visual: falta la escena " + p.string() + " (descargar del dataset rl-tools/conta)");
    return p.string();
}

struct Scene {
    Library* library = nullptr;
    Renderer* renderer = nullptr;
    Annotations* annotations = nullptr;
    TF translation[3] = {0, 0, 0};
};

class L2fVisualEnv : public ControlEnv {
    DEVICE device;
    RNG rng;
    ENV env;
    typename ENV::Parameters params;
    typename ENV::State state;
    std::vector<Scene> scenes_;
    std::string name_;

public:
    L2fVisualEnv(std::string name, const std::vector<std::string>& hashes) : name_(std::move(name)) {
        rlt::init(device, rng, 0);
        rlt::malloc(device, env);
        rlt::init(device, env);
        env.use_target_mode = true;
        typename ENV::Parameters defaults;
        rlt::initial_parameters(device, env, defaults);
        for (const auto& h : hashes) {
            Scene sc;
            sc.library = new Library{};
            sc.renderer = new Renderer{};
            sc.annotations = new Annotations{};
            rlt::malloc(device, *sc.library);
            rlt::malloc(device, *sc.renderer, *sc.library);
            rlt::rendering::Bundle<TF> bundle;
            const std::string path = scenePath(h);
            rlt::load<typename RSPEC::SHADING, RSPEC::HAS_RGB>(device, bundle, path.c_str());
            auto id = rlt::insert(device, *sc.library, bundle);
            rlt::init(device, *sc.renderer, *sc.library, id);
            rlt::generate_probe_directions(device, *sc.renderer);
            rlt::rendering::datasets::annotations::FreeSpaceParameters<TF, TI> fsp{};
            rlt::rendering::datasets::annotations::annotate(device, *sc.annotations, bundle.metadata, *sc.renderer, fsp);
            if (sc.annotations->num_positions == 0)
                throw std::runtime_error("l2f-visual: la escena " + h + " no tiene posiciones libres");
            // Como training.cpp: el origen del dron va a la primera posición libre (más el alto de la cámara)
            const auto& pos = sc.annotations->positions[0].position;
            for (int j = 0; j < 3; j++) sc.translation[j] = pos[j];
            sc.translation[2] += defaults.camera_mount.offset_body[2];
            scenes_.push_back(sc);
        }
    }
    ~L2fVisualEnv() override {
        env.renderer = nullptr;
        env.annotations = nullptr;
        rlt::free(device, env);
        for (auto& sc : scenes_) {
            rlt::free(device, *sc.renderer);
            rlt::free(device, *sc.library);
            delete sc.renderer;
            delete sc.library;
            delete sc.annotations;
        }
    }
    L2fVisualEnv(const L2fVisualEnv&) = delete;
    L2fVisualEnv& operator=(const L2fVisualEnv&) = delete;

    std::string name() const override { return name_; }
    int obsDim() const override { return static_cast<int>(OBS_DIM); }
    int actionDim() const override { return static_cast<int>(ACT_DIM); }
    bool nonNegativeObs() const override { return true; }
    int maxSteps() const override { return static_cast<int>(ENV::EPISODE_STEP_LIMIT); }

    void reset(uint64_t seed) override {
        const Scene& sc = scenes_[seed % scenes_.size()];
        env.renderer = sc.renderer;
        env.annotations = sc.annotations;
        rlt::init(device, rng, static_cast<typename DEVICE::index_t>(seed));
        rlt::sample_initial_parameters(device, env, params, rng);
        for (int j = 0; j < 3; j++) params.scene_translation[j] = sc.translation[j];
        rlt::sample_initial_state(device, env, params, state, rng);
    }

    std::vector<double> rawObserve() override {
        rlt::Matrix<rlt::matrix::Specification<TF, TI, 1, OBS_DIM, false>> m;
        rlt::observe(device, env, params, state, typename ENV::Observation{}, m, rng);
        std::vector<double> o(OBS_DIM);
        for (TI i = 0; i < OBS_DIM; i++) o[i] = rlt::get(m, 0, i);
        return o;
    }
    std::vector<double> observe() override {
        auto o = rawObserve();
        for (double& x : o) x = std::clamp(x, 0.0, 1.0);   // RGB ∈ [0, 1]
        return o;
    }
    StepResult step(const std::vector<double>& action) override {
        rlt::Matrix<rlt::matrix::Specification<TF, TI, 1, ACT_DIM, false>> a;
        for (TI j = 0; j < ACT_DIM; j++) rlt::set(a, 0, j, static_cast<TF>(std::clamp(action.at(j), -1.0, 1.0)));
        typename ENV::State next;
        rlt::step(device, env, params, state, a, next, rng);
        const double r = rlt::reward(device, env, params, state, a, next, rng);
        state = next;
        return {r, rlt::terminated(device, env, params, state, rng)};
    }
};
}  // namespace l2fv

// ── hyperdrone: mundo visual de RLtools (raytracing, por lotes); aquí un lote de 1 instancia ──
// MDP: el del entrenador oficial de hyperdrone (src/rl/environments/l2f_visual/training_hyperdrone_config.h):
// Crazyflie con cámara (crazyflie_openmv), recompensa cuadrática con −100 al terminar, terminación a 1 m o
// con velocidades extremas, inicio en la pose meta con inclinación ≤ 0,3 rad. Los ejemplos de video
// (demo_common.h) NO sirven: su recompensa tiene todos los pesos en 0 y no terminan. Diferencias con el
// entrenador oficial (docs/entornos_rltools.md §6): 500 pasos (5 s, como l2f) en vez de 10 000; cámara 16×16;
// la observación es solo imagen (el oficial agrega orientación, giro e historial de acciones y apila 10
// cuadros: aquí la memoria la da --keep-state); sin aleatorización de dominio.
// Mismas casas y particiones que l2f-visual; la casa del episodio es (semilla mod 2) vía select_scene.
//   hyperdrone-target-frame: ve su cámara + una imagen objetivo tomada desde la pose meta (1536 entradas);
//     debe volver a esa pose y mantenerse (servo-control visual).
//   hyperdrone-moving-gate:  ve su cámara (768 entradas); +10 al pasar por una "puerta" que oscila y fin si
//     choca. El modelo 3D de la puerta del ejemplo oficial (x500.glb) no es público: se usa un dron del
//     dataset público rl-tools/conta (87d2b8b1, nodos body/prop_*). Pasar o chocar es geométrico (radio
//     0,4 m) y no depende del modelo; solo cambia la apariencia.
namespace hd {
using TF = float;
namespace l2f = E::l2f;
struct DYN {
    using T = TF;
    using TI = envs::TI;
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr TI CLOSED_FORM = false;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
    using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
            l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
            l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
            l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using PARAMETERS = PARAMETERS_TYPE;
    // Valores de training_hyperdrone_config.h
    static constexpr REWARD_FUNCTION reward_function = {
        false, 0.10, 1.00, -100.00, 10.00, 0.00, 1.00, 0.05, 1.50, 0.00, 0.00,
        {0.00, 0.00, 0.00, 0.00}, {1.50, 1.50, 1.50, 1.50}, 0.00};
    static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {1.0, 0.0, 0.3, 1.0, 1.0, true, -1, +1};
    static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {true, 1.0, 0, 10, 35, 10000, 50000};
    static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::crazyflie_openmv, PARAMETERS_SPEC>;
    static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)0.01};
    static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, reward_function, {}, {}, termination};
    static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
    static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
    static constexpr T STATE_LIMIT_POSITION_X = 100000;
    static constexpr T STATE_LIMIT_POSITION_Y = 100000;
    static constexpr T STATE_LIMIT_POSITION_Z = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
};
struct WorldSpec : E::hyperdrone::Specification<TF, TI, DYN> {
    static constexpr TI INSTANCES_PER_ENVIRONMENT = 1;
    static constexpr TI CAM_WIDTH = 16;
    static constexpr TI CAM_HEIGHT = 16;
    static constexpr TI HISTORY_LENGTH = 1;
};
struct WorldSpecGate : WorldSpec {
    static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;
};
using BaseTF = E::hyperdrone::World<WorldSpec>;
using WorldTF = E::hyperdrone::tasks::target_frame::World<E::hyperdrone::tasks::target_frame::Specification<BaseTF>>;
using BaseMG = E::hyperdrone::World<WorldSpecGate>;
using WorldMG = E::hyperdrone::tasks::moving_gate::World<E::hyperdrone::tasks::moving_gate::Specification<BaseMG>>;
static const std::string kGateAsset = "87d2b8b1d518445872d34906320a76656731902f";

template <typename WORLD, typename OBS, bool HAS_GATE = false>
class HyperdroneEnv : public ControlEnv {
    static constexpr TI N = WORLD::INSTANCES;
    static_assert(N == 1, "un lote de 1 instancia");
    static constexpr TI OBS_DIM = OBS::DIM;
    static constexpr TI ACT_DIM = WORLD::ACTION_DIM;

    DEVICE device;
    RNG rng;
    WORLD* world;
    typename WORLD::SharedContext* shared;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, N>>> params;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, N>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, N>>> reset_mask, terminated_flags;
    rlt::Tensor<rlt::tensor::Specification<TF, TI, rlt::tensor::Shape<TI, N, ACT_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<TF, TI, rlt::tensor::Shape<TI, N, OBS_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<TF, TI, rlt::tensor::Shape<TI, N>>> rewards;
    std::string name_;
    TI num_scenes_;

public:
    HyperdroneEnv(std::string name, const std::vector<std::string>& hashes, const std::string& gate_hash = "")
        : world(new WORLD{}), shared(new typename WORLD::SharedContext{}), name_(std::move(name)),
          num_scenes_(hashes.size()) {
        rlt::malloc(device, rng);
        rlt::init(device, rng, 0);
        rlt::malloc(device, shared->library);
        rlt::malloc(device, *world);
        if constexpr (HAS_GATE) world->gate_asset_path = l2fv::scenePath(gate_hash);
        std::vector<std::string> paths;
        for (const auto& h : hashes) paths.push_back(l2fv::scenePath(h));
        rlt::rendering::datasets::procthor::GLB dataset{{}, paths};
        typename decltype(dataset)::Corpus corpus;
        rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
        rlt::init(device, *world, *shared, dataset, corpus, 0, num_scenes_, 0);
        rlt::malloc(device, params);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, terminated_flags);
        rlt::malloc(device, actions);
        rlt::malloc(device, observations);
        rlt::malloc(device, rewards);
    }
    ~HyperdroneEnv() override {
        rlt::free(device, params);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, reset_mask);
        rlt::free(device, terminated_flags);
        rlt::free(device, actions);
        rlt::free(device, observations);
        rlt::free(device, rewards);
        rlt::free(device, *world);
        rlt::free(device, shared->library);
        delete world;
        delete shared;
    }
    HyperdroneEnv(const HyperdroneEnv&) = delete;
    HyperdroneEnv& operator=(const HyperdroneEnv&) = delete;

    std::string name() const override { return name_; }
    int obsDim() const override { return static_cast<int>(OBS_DIM); }
    int actionDim() const override { return static_cast<int>(ACT_DIM); }
    bool nonNegativeObs() const override { return true; }
    int maxSteps() const override { return static_cast<int>(DYN::EPISODE_STEP_LIMIT); }

    void reset(uint64_t seed) override {
        rlt::select_scene(device, *world, static_cast<TI>(seed % num_scenes_));
        rlt::init(device, rng, static_cast<typename DEVICE::index_t>(seed));
        rlt::set_all(device, reset_mask, true);
        rlt::sample_initial_parameters(device, *world, params, reset_mask, rng);
        rlt::sample_initial_state(device, *world, params, states, reset_mask, rng);
    }
    std::vector<double> rawObserve() override {
        rlt::render(device, *world, params, states, reset_mask);
        rlt::set_all(device, reset_mask, false);
        rlt::observe(device, *world, params, states, OBS{}, observations, rng);
        std::vector<double> o(OBS_DIM);
        for (TI i = 0; i < OBS_DIM; i++) o[i] = rlt::get(device, observations, 0, i);
        return o;
    }
    std::vector<double> observe() override {
        auto o = rawObserve();
        for (double& x : o) x = std::clamp(x, 0.0, 1.0);   // RGB ∈ [0, 1]
        return o;
    }
    StepResult step(const std::vector<double>& action) override {
        for (TI j = 0; j < ACT_DIM; j++)
            rlt::set(device, actions, static_cast<TF>(std::clamp(action.at(j), -1.0, 1.0)), 0, j);
        rlt::step(device, *world, params, states, actions, next_states, rng);
        rlt::reward(device, *world, params, states, actions, next_states, rewards, rng);
        rlt::copy(device, device, next_states, states);
        rlt::terminated(device, *world, params, states, terminated_flags, rng);
        return {static_cast<double>(rlt::get(device, rewards, 0)), rlt::get(device, terminated_flags, 0)};
    }
};
}  // namespace hd
#endif

using Factory = std::function<std::unique_ptr<ControlEnv>(const EnvOptions&)>;

static const std::map<std::string, Factory>& registry() {
    static const std::map<std::string, Factory> r = {
        {"pendulum", [](const EnvOptions& o) {
             return make<E::Pendulum<PendulumSpec>>("pendulum", std::vector<ObsDim>{unit(), unit(), sym(8)}, 200, 1, o.torque_scale);
         }},
        // Solo el ángulo (cos, sin): la velocidad hay que inferirla → requiere memoria
        {"pendulum-position", [](const EnvOptions& o) {
             return make<E::PendulumPosition<PendulumSpec>>("pendulum-position", std::vector<ObsDim>{unit(), unit()}, 200, 1, o.torque_scale);
         }},
        // Solo la velocidad: la posición hay que integrarla → requiere memoria
        {"pendulum-velocity", [](const EnvOptions& o) {
             return make<E::PendulumVelocity<PendulumSpec>>("pendulum-velocity", std::vector<ObsDim>{sym(8)}, 200, 1, o.torque_scale);
         }},
        // La acción se invierte al azar por episodio y la observación lo indica (4.ª dimensión)
        {"pendulum-multitask", [](const EnvOptions& o) {
             return make<E::PendulumMultiTask<PendulumSpec>>("pendulum-multitask", std::vector<ObsDim>{unit(), unit(), sym(8), unit()}, 200, 1, o.torque_scale);
         }},
        // Igual, pero la inversión NO se observa (solo la última acción): hay que inferirla → memoria
        {"pendulum-meta", [](const EnvOptions& o) {
             return make<E::PendulumMeta<PendulumSpec>>("pendulum-meta", std::vector<ObsDim>{unit(), unit(), sym(8), unit()}, 200, 1, o.torque_scale);
         }},
        // Estilo gym: −1 por paso, termina al alcanzar la altura objetivo; torque ±5 (torque_scale 0,2 → ±1)
        {"acrobot", [](const EnvOptions& o) {
             return make<E::Acrobot<AcrobotSpec>>("acrobot", std::vector<ObsDim>{unit(), unit(), unit(), unit(), sym(4 * M_PI), sym(9 * M_PI)}, 500, 1, o.torque_scale);
         }},
        // Subir y equilibrar invertido; costo moldeado; sin término; inicio aleatorio
        {"acrobot-swingup", [](const EnvOptions& o) {
             return make<E::AcrobotSwingup<AcrobotSwingupSpec>>("acrobot-swingup", std::vector<ObsDim>{unit(), unit(), unit(), unit(), sym(4 * M_PI), sym(9 * M_PI)}, 500, 1, o.torque_scale);
         }},
        // Quedarse en el origen con orientación 0; termina fuera de ±1 m; sin límite propio → 500 pasos (5 s)
        {"car", [](const EnvOptions&) {
             return make<E::Car<CarSpec>>("car", carObs(), 500);
         }},
        // Recorrer la pista lo más rápido posible (recompensa vx); termina al salir; 1000 pasos (10 s)
        {"car-track", [](const EnvOptions&) {
             auto o = carObs();
             for (int i = 0; i < 3; i++) o.push_back(range(0, 1));     // rayos: distancia libre normalizada
             return make<E::CarTrack<CarTrackSpec>>("car-track", o, 1000);
         }},
        // Visitar 2 banderas en orden; sus posiciones solo se observan en el paso 0 → memoria
        {"flag", [](const EnvOptions&) {
             std::vector<ObsDim> o{range(0, 10), range(0, 10), sym(5), sym(5), unit(), unit(), unit()};
             for (int i = 0; i < 4; i++) o.push_back(range(0, 10));   // después del paso 0 vale −1 → satura en −1
             return make<E::Flag<FlagSpec>>("flag", o, 50);
         }},
        // Emitir (n.º de unos en los últimos 10 pasos)/10 viendo un bit por paso → memoria
        {"memory", [](const EnvOptions&) {
             return make<E::Memory<MemorySpec>>("memory", std::vector<ObsDim>{unit()}, 20);
         }},
        // Llevar un punto a un objetivo en [−1, 1]²; termina al llegar; 40 pasos
        {"reacher", [](const EnvOptions&) {
             return make<E::Reacher<ReacherSpec>>("reacher", std::vector<ObsDim>{unit(), unit(), unit(), unit()}, 40);
         }},
        // Reacher observado por imagen 16×16×3 (768 entradas)
        {"reacher-visual", [](const EnvOptions&) { return makeImage<E::ReacherVisual<ReacherVisualSpec>>("reacher-visual"); }},
        // Por imagen y con memoria (objetivos secuenciales); hard = 2 objetivos, 80 pasos
        {"reacher-visual-memory", [](const EnvOptions&) {
             return makeImage<E::ReacherVisualMemory<ReacherVisualSpec>>("reacher-visual-memory"); }},
        {"reacher-visual-memory-hard", [](const EnvOptions&) {
             return makeImage<E::ReacherVisualMemory<ReacherVisualHardSpec>>("reacher-visual-memory-hard"); }},
        // Cuadricóptero (zoo): estabilizar en el origen desde inicios de hasta 90°; 500 pasos a 100 Hz
        {"l2f", [](const EnvOptions&) { return makeL2f("l2f"); }},
        // Variante pequeña del zoo: seguimiento de trayectoria, 250 pasos a 50 Hz
        {"l2f-tiny", [](const EnvOptions&) { return makeL2f("l2f-tiny"); }},
#ifdef TESIS_HAVE_MUJOCO
        // Locomoción: avanzar en x sin caerse (gymnasium Ant-v4)
        {"ant", [](const EnvOptions&) { return make<E::mujoco::Ant<AntSpec>>("ant", antObs(), 1000); }},
#endif
#ifdef TESIS_HAVE_RAYTRACING
        // Cuadricóptero observado solo por cámara 16×16 dentro de casas ProcTHOR (con --test-split: casas de test)
        {"l2f-visual", [](const EnvOptions& o) {
             return std::make_unique<l2fv::L2fVisualEnv>("l2f-visual", o.test_split ? l2fv::kTestScenes : l2fv::kTrainScenes);
         }},
        // Servo-control visual: volver a la pose de una imagen objetivo, viendo cámara + objetivo (1536 entradas)
        {"hyperdrone-target-frame", [](const EnvOptions& o) {
             return std::make_unique<hd::HyperdroneEnv<hd::WorldTF, typename hd::WorldTF::Observation>>(
                 "hyperdrone-target-frame", o.test_split ? l2fv::kTestScenes : l2fv::kTrainScenes);
         }},
        // Pasar por una puerta que oscila, viendo solo la cámara (768 entradas)
        {"hyperdrone-moving-gate", [](const EnvOptions& o) {
             return std::make_unique<hd::HyperdroneEnv<hd::WorldMG, typename hd::BaseMG::Observation, true>>(
                 "hyperdrone-moving-gate", o.test_split ? l2fv::kTestScenes : l2fv::kTrainScenes, hd::kGateAsset);
         }},
#endif
        // 5 agentes cruzan un cuello de botella; política compartida (la misma red por agente)
        {"bottleneck", [](const EnvOptions&) {
             return make<E::multi_agent::Bottleneck<BottleneckSpec>>("bottleneck", bottleneckObs(), 200,
                                                                     static_cast<int>(BottleneckParams::N_AGENTS));
         }},
    };
    return r;
}

std::vector<std::string> envNames() {
    std::vector<std::string> n;
    for (const auto& [k, _] : registry()) n.push_back(k);
    return n;
}

std::unique_ptr<ControlEnv> makeEnv(const std::string& name, const EnvOptions& opt) {
    auto it = registry().find(name);
    if (it == registry().end()) throw std::invalid_argument("makeEnv: entorno desconocido '" + name + "'");
    return it->second(opt);
}

}  // namespace envs
