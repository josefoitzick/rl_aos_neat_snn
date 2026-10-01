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
)
target_link_libraries(aos PUBLIC neat)
target_compile_options(aos PRIVATE -Wall -Wextra)

# Entornos de control (RLtools) y evaluación de genomas; requieren external/rl-tools
if(TARGET rl_tools)
  add_library(envs src/envs/rltools_envs.cpp)
  target_include_directories(envs PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(envs PRIVATE rl_tools)
  target_compile_options(envs PRIVATE -Wall -Wextra)

  add_library(eval src/eval/evaluator.cpp)
  target_link_libraries(eval PUBLIC neat envs OpenMP::OpenMP_CXX)
  target_compile_options(eval PRIVATE -Wall -Wextra)

  add_executable(neat_control src/experiment/neat_control.cpp)
  target_link_libraries(neat_control PRIVATE eval aos CLI11::CLI11)
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
  )
  target_link_libraries(tesis_tests PRIVATE snn_engine neat aos Catch2::Catch2WithMain)
  if(TARGET eval)
    target_sources(tesis_tests PRIVATE tests/eval_tests.cpp)
    target_link_libraries(tesis_tests PRIVATE eval)
  endif()
  include(Catch)
  catch_discover_tests(tesis_tests)
endif()
