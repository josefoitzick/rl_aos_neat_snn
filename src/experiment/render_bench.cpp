// Costo de renderizar l2f_visual en CPU (paso 5 de la selección de entornos, docs/entornos_rltools.md).
// Carga una escena ProcTHOR (.glb), crea el entorno con una cámara (NUM_ENVS = 1) y mide el tiempo por
// observación para cámaras de 16, 32 y 64 px. Ejecutable aparte: RLtools no admite dos unidades de
// compilación con sus encabezados en el mismo binario.
#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>
#include <rl_tools/rendering/datasets/annotations/operations_cpu.h>
#include <rl/zoo/l2f_visual/environment.h>

#include <chrono>
#include <cstdio>
#include <string>

namespace rlt = rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using RNG = rlt::devices::random::CPU::ENGINE<>;

template <TI CAM>
void bench(const std::string& scene, int frames) {
    using ENV = typename rlt::rl::zoo::l2f_visual::ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI, 1, CAM, CAM>::ENVIRONMENT;
    using RENDERER_SPEC = typename ENV::SPEC::RENDERER_SPEC;
    DEVICE device;
    RNG rng;
    rlt::init(device, rng, 1);

    auto t0 = std::chrono::steady_clock::now();
    auto* library = new rlt::rendering::raytracing::AssetLibrary<RENDERER_SPEC>{};
    auto* renderer = new rlt::rendering::raytracing::Renderer<RENDERER_SPEC>{};
    auto* annotations = new rlt::rendering::datasets::annotations::FreeSpace<typename ENV::SPEC::ANNOTATIONS_SPEC>{};
    rlt::malloc(device, *library);
    rlt::malloc(device, *renderer, *library);
    rlt::rendering::Bundle<T> bundle;
    rlt::load<typename RENDERER_SPEC::SHADING, RENDERER_SPEC::HAS_RGB>(device, bundle, scene.c_str());
    auto scene_id = rlt::insert(device, *library, bundle);
    rlt::init(device, *renderer, *library, scene_id);
    rlt::generate_probe_directions(device, *renderer);
    rlt::rendering::datasets::annotations::FreeSpaceParameters<T, TI> fsp{};
    rlt::rendering::datasets::annotations::annotate(device, *annotations, bundle.metadata, *renderer, fsp);
    const double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    ENV env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    env.renderer = renderer;
    env.annotations = annotations;
    env.use_target_mode = true;
    typename ENV::Parameters params;
    rlt::initial_parameters(device, env, params);
    typename ENV::State state, next;
    rlt::sample_initial_state(device, env, params, state, rng);
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::Observation::DIM, false>> obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM, false>> action;
    rlt::set_all(device, action, (T)0);

    double obs_s = 0, step_s = 0, mean = 0;
    for (int k = 0; k < frames; k++) {
        auto a = std::chrono::steady_clock::now();
        rlt::observe(device, env, params, state, typename ENV::Observation{}, obs, rng);
        auto b = std::chrono::steady_clock::now();
        rlt::step(device, env, params, state, action, next, rng);
        state = next;
        if (rlt::terminated(device, env, params, state, rng)) rlt::sample_initial_state(device, env, params, state, rng);
        auto c = std::chrono::steady_clock::now();
        obs_s += std::chrono::duration<double>(b - a).count();
        step_s += std::chrono::duration<double>(c - b).count();
        for (TI i = 0; i < ENV::Observation::DIM; i++) mean += rlt::get(obs, 0, i);
    }
    std::printf("cámara %3d×%-3d  obs %6d  preparación %.2f s  render %.3f ms/obs  dinámica %.4f ms/paso  "
                "píxel medio %.3f  posiciones libres %d\n",
                (int)CAM, (int)CAM, (int)ENV::Observation::DIM, setup_s, 1e3 * obs_s / frames, 1e3 * step_s / frames,
                mean / frames / ENV::Observation::DIM, (int)annotations->num_positions);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "uso: render_bench <escena.glb> [cuadros]\n"); return 1; }
    const int frames = argc > 2 ? std::stoi(argv[2]) : 200;
    bench<16>(argv[1], frames);
    bench<32>(argv[1], frames);
    bench<64>(argv[1], frames);
    return 0;
}
