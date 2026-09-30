cmake_minimum_required(VERSION 3.10)
project(SNNSimulator VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(CMAKE_CROSSCOMPILING AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv8-a+crc")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include(CTest)
enable_testing()

find_package(CLI11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# RLTools (header-only). En el repo de la tesis vive como submódulo en
# external/rl-tools; el CMakeLists raíz fija esta variable antes de incluir snn/.
set(RL_TOOLS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../external/rl-tools/include
    CACHE PATH "Directorio de cabeceras de RLTools")
if(EXISTS "${RL_TOOLS_INCLUDE_DIR}/rl_tools")
  set(SNN_HAVE_RL_TOOLS ON)
  add_library(rl_tools INTERFACE)
  target_include_directories(rl_tools SYSTEM INTERFACE ${RL_TOOLS_INCLUDE_DIR})
else()
  set(SNN_HAVE_RL_TOOLS OFF)
  message(WARNING "RLTools no encontrado en ${RL_TOOLS_INCLUDE_DIR}: se omiten snn_pendulum y snn_acrobot")
endif()

# Core del simulador
add_library(snn_core
    src/core/neuron.cpp
    src/core/network.cpp
)

target_include_directories(snn_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Codificador del simulador
add_library(snn_encoding
    src/encoding/encoder.cpp
    src/encoding/poissonEncoder.cpp
    src/encoding/rateEncoder.cpp
    src/encoding/rlEncoder.cpp
    src/encoding/ttfsEncoder.cpp
)

target_include_directories(snn_encoding PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Motor + decodificador
add_library(snn_engine
    src/core/simulator.cpp
    src/decoding/decoder.cpp
)

target_include_directories(snn_engine PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(snn_engine
    PUBLIC
        snn_core
        snn_encoding
        OpenMP::OpenMP_CXX
)

# Simulador principal
add_executable(snn_simulator
    src/core/main.cpp
)
target_link_libraries(snn_simulator 
    PRIVATE
        snn_engine
        CLI11::CLI11
)

# Tests — gated behind BUILD_TESTING (from include(CTest) above, default ON)
# so environments without Catch2 3 (e.g. the ZCU104/PYNQ board, which only
# ships Catch2 2.13.8 via apt) can still build the core libraries with
# -DBUILD_TESTING=OFF, without needing Catch2 at all.
if(BUILD_TESTING)
  find_package(Catch2 3 REQUIRED)

  add_executable(snn_tests
      tests/core/networkTests.cpp
      tests/core/neuronTests.cpp
      tests/core/izhikevichTests.cpp
      tests/core/weightsRestTests.cpp
  )
  target_link_libraries(snn_tests
      PRIVATE
          snn_engine
          Catch2::Catch2WithMain
  )

  include(Catch)
  catch_discover_tests(snn_tests)
endif()

# Inyección de corriente directa en neuronas individuales
add_executable(snn_current_injection
    src/core/current_injection_main.cpp
)
target_link_libraries(snn_current_injection
    PRIVATE
        snn_core
        CLI11::CLI11
)

if(SNN_HAVE_RL_TOOLS)
# Evaluación del péndulo con RLTools
add_executable(snn_pendulum
    src/rl/pendulum_eval.cpp
)
target_link_libraries(snn_pendulum
    PRIVATE
        snn_engine
        rl_tools
)

# Evaluación del acrobot con RLTools
add_executable(snn_acrobot
    src/rl/acrobot_eval.cpp
)
target_link_libraries(snn_acrobot
    PRIVATE
        snn_engine
        rl_tools
)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")

