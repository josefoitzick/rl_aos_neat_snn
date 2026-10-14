cmake_minimum_required(VERSION 3.16)
project(RlAosNeatSnn VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

# Todos los ejecutables en build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CTest)

# RLTools (header-only), submódulo en external/rl-tools
set(RL_TOOLS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/rl-tools/include
    CACHE PATH "Directorio de cabeceras de RLTools")

# Simulador SNN
add_subdirectory(snn)

# ── Módulos de la tesis ──
# (src/aos, src/envs, src/eval se agregan a medida que tengan código)

find_package(OpenMP REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

# NEAT: genoma, marcas históricas, especiación, reproducción, selector de operadores
add_library(neat
    src/neat/genome.cpp
    src/neat/innovation.cpp
    src/neat/population.cpp
    src/neat/selector.cpp
)
target_include_directories(neat PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(neat PUBLIC snn_core)
target_compile_options(neat PRIVATE -Wall -Wextra)

# AOS: selectores de operadores (Adaptive Pursuit; luego PPO)
add_library(aos
    src/aos/adaptive_pursuit.cpp
    src/aos/factory.cpp
    src/aos/features.cpp
)
target_link_libraries(aos PUBLIC neat)
target_compile_options(aos PRIVATE -Wall -Wextra)

# Entornos de control (RLtools) y evaluación de genomas; requieren external/rl-tools
if(TARGET rl_tools)
  add_library(envs src/envs/rltools_envs.cpp)
  target_include_directories(envs PUBLIC ${CMAKE_SOURCE_DIR}/src)
  # Configuraciones del zoo de RLtools (src/rl/zoo/...) para l2f
  target_include_directories(envs SYSTEM PRIVATE ${RL_TOOLS_INCLUDE_DIR}/../src)
  # MuJoCo (opcional, para ant): 2.3.5, la versión que usa RLtools, compilado en ~/opt/mujoco-2.3.5.
  # Se importa directo: su mujocoConfig.cmake exige OpenGL (solo para renderizar), que mullo no tiene,
  # y libmujoco no depende de él.
  set(MUJOCO_ROOT $ENV{HOME}/opt/mujoco-2.3.5 CACHE PATH "Instalación de MuJoCo 2.3.5")
  find_library(MUJOCO_LIBRARY mujoco PATHS ${MUJOCO_ROOT}/lib NO_DEFAULT_PATH)
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h PATHS ${MUJOCO_ROOT}/include NO_DEFAULT_PATH)
  if(MUJOCO_LIBRARY AND MUJOCO_INCLUDE_DIR)
    target_link_libraries(envs PRIVATE ${MUJOCO_LIBRARY})
    target_include_directories(envs SYSTEM PRIVATE ${MUJOCO_INCLUDE_DIR})
    target_compile_definitions(envs PRIVATE TESIS_HAVE_MUJOCO)
    message(STATUS "MuJoCo: ${MUJOCO_LIBRARY} (entorno ant habilitado)")
  else()
    message(STATUS "MuJoCo no encontrado en ${MUJOCO_ROOT}: entorno ant deshabilitado")
  endif()
  target_link_libraries(envs PRIVATE rl_tools)
  target_compile_options(envs PRIVATE -Wall -Wextra)

  add_library(eval src/eval/evaluator.cpp)
  target_link_libraries(eval PUBLIC neat envs OpenMP::OpenMP_CXX)
  target_compile_options(eval PRIVATE -Wall -Wextra)

  add_executable(neat_control src/experiment/neat_control.cpp)
  target_link_libraries(neat_control PRIVATE eval aos CLI11::CLI11)

  # Raytracing de RLtools en CPU (backend GENERIC) para los entornos visuales l2f_visual y hyperdrone.
  # Requiere assimp (lector de .glb) y stb, compilados en ~/opt sin root (docs/entornos_rltools.md).
  set(ASSIMP_ROOT $ENV{HOME}/opt/assimp-5.4.3 CACHE PATH "Instalación de assimp")
  set(STB_INCLUDE_DIR $ENV{HOME}/opt/src/stb CACHE PATH "Encabezados de stb (commit fijado por RLtools)")
  find_package(assimp QUIET PATHS ${ASSIMP_ROOT} NO_DEFAULT_PATH)
  if(assimp_FOUND AND EXISTS ${STB_INCLUDE_DIR}/stb_image.h)
    message(STATUS "Raytracing CPU: assimp ${assimp_VERSION} + stb (l2f-visual y render_bench habilitados)")
    target_link_libraries(envs PRIVATE assimp::assimp)
    target_include_directories(envs SYSTEM PRIVATE ${STB_INCLUDE_DIR})
    target_compile_definitions(envs PRIVATE TESIS_HAVE_RAYTRACING
        RL_TOOLS_RENDERING_ENABLE_RAYTRACING RL_TOOLS_RENDERING_RAYTRACING_BACKEND_GENERIC)
    # Costo de renderizar l2f_visual en CPU (escenas ProcTHOR)
    add_executable(render_bench src/experiment/render_bench.cpp)
    target_link_libraries(render_bench PRIVATE rl_tools assimp::assimp)
    target_include_directories(render_bench SYSTEM PRIVATE ${RL_TOOLS_INCLUDE_DIR}/../src ${STB_INCLUDE_DIR})
    target_compile_definitions(render_bench PRIVATE
        RL_TOOLS_RENDERING_ENABLE_RAYTRACING RL_TOOLS_RENDERING_RAYTRACING_BACKEND_GENERIC)
  else()
    message(STATUS "Raytracing CPU deshabilitado: falta assimp en ${ASSIMP_ROOT} o stb en ${STB_INCLUDE_DIR}")
  endif()

  # Sonda de entornos: referencias, cotas de observación y costo de la SNN (docs/entornos_rltools.md)
  add_executable(env_probe src/experiment/env_probe.cpp)
  target_link_libraries(env_probe PRIVATE eval CLI11::CLI11)

  # Referencia RL: política del zoo de RLtools (cargada con dlopen) evaluada con los episodios finales de NEAT
  # (docs/diseno_neat_ap.md §15.4). Las políticas se compilan con scripts/compilar_politica_rl.sh.
  add_executable(rl_ref_eval src/experiment/rl_ref_eval.cpp)
  target_link_libraries(rl_ref_eval PRIVATE eval CLI11::CLI11 ${CMAKE_DL_LIBS})

  # NEAT como entorno del agente PPO (NeatRun / NeatPool, docs/diseno_ppo.md §3.2–3.3) y su verificación V1
  add_library(aos_entorno src/aos/neat_run.cpp)
  target_link_libraries(aos_entorno PUBLIC eval aos)
  target_compile_options(aos_entorno PRIVATE -Wall -Wextra)
  add_executable(aos_neat_check src/experiment/aos_neat_check.cpp)
  target_link_libraries(aos_neat_check PRIVATE aos_entorno CLI11::CLI11)

  # Paso 3 del agente PPO: entrenamiento de PPO (RLtools) sobre NEAT (NeatPool), con checkpoints (docs/diseno_ppo.md §3–4)
  add_executable(aos_ppo_train src/experiment/aos_ppo_train.cpp)
  target_link_libraries(aos_ppo_train PRIVATE aos_entorno rl_tools CLI11::CLI11 OpenMP::OpenMP_CXX)
  target_compile_options(aos_ppo_train PRIVATE -Wall -Wextra)

  # Paso 1 del agente PPO: el PPO de RLtools sobre un AOS sintético, con el lote propio (docs/diseno_ppo.md §5)
  add_executable(aos_toy_ppo src/experiment/aos_toy_ppo.cpp)
  target_include_directories(aos_toy_ppo PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(aos_toy_ppo PRIVATE rl_tools CLI11::CLI11 OpenMP::OpenMP_CXX)
  target_compile_options(aos_toy_ppo PRIVATE -Wall -Wextra)
endif()

# Verificación NEAT + SNN en XOR (como la §4.2 de Stanley & Miikkulainen 2002)
add_executable(xor_snn src/experiment/xor_snn.cpp)
target_link_libraries(xor_snn PRIVATE neat aos OpenMP::OpenMP_CXX CLI11::CLI11)

if(BUILD_TESTING)
  find_package(Catch2 3 REQUIRED)
  add_executable(tesis_tests
      tests/smoke_snn_tests.cpp
      tests/neat_tests.cpp
      tests/aos_tests.cpp
      tests/operadores_tests.cpp
      tests/ppo_tests.cpp
  )
  target_link_libraries(tesis_tests PRIVATE snn_engine neat aos Catch2::Catch2WithMain)
  if(TARGET eval)
    target_sources(tesis_tests PRIVATE tests/eval_tests.cpp tests/neat_run_tests.cpp)
    target_link_libraries(tesis_tests PRIVATE eval aos_entorno)
  endif()
  include(Catch)
  catch_discover_tests(tesis_tests)
endif()
