# rl_aos_neat_snn

Código de una tesis (memoria de título) sobre **entrenamiento de redes neuronales de espigas mediante
neuroevolución con selección adaptativa de operadores**.

NEAT evoluciona redes de espigas (Izhikevich) para tareas de control. Un agente PPO (RLtools) selecciona
en cada generación cómo aplicar los operadores de mutación, a partir del estado de la población (AOS
basado en estado). Se compara contra probabilidades estáticas y contra Adaptive Pursuit.

## Estructura

```
snn/              Simulador SNN en C++ (basado en snn-simulator; ver snn/README.md)
src/neat/         Genoma, innovaciones, especiación, población y catálogo de operadores de mutación
src/aos/          Selectores de operadores: estático | Adaptive Pursuit | PPO (misma interfaz)
src/envs/         Entornos de control de RLtools + codificación/decodificación hacia la SNN
src/eval/         Evaluación genoma → red → episodios → fitness (paralela y con semillas)
src/experiment/   Ejecutables (NEAT en control, sondas de entornos, evaluación de políticas RL de referencia)
src/ref_rl/       Configuraciones de los agentes RL de referencia
configs/          Configuraciones de experimentos
tests/            Tests (Catch2)
scripts/          Lanzadores, estudio de hiperparámetros (Optuna) y análisis
analysis/         Análisis estadístico y gráficos (Python)
external/         Dependencias como submódulos (rl-tools)
```

## Compilar

Requiere CMake ≥ 3.16, un compilador C++17 con OpenMP, CLI11 y Catch2 3.

```bash
git submodule update --init          # trae external/rl-tools
cmake -S . -B build
cmake --build build -j 16
ctest --test-dir build
```

Los ejecutables quedan en `build/bin/`. Los resultados de experimentos se guardan fuera del repo.

### Dependencias opcionales (entornos extra)

CMake las detecta solo; sin ellas, esos entornos no se registran y el resto compila igual.

| Dependencia | Habilita | Ubicación por defecto |
|---|---|---|
| MuJoCo 2.3.5 (la versión que usa RLtools) | `ant` | `~/opt/mujoco-2.3.5` (`-DMUJOCO_ROOT=...`) |
| assimp 5.x + stb (commit fijado por RLtools) | `l2f-visual`, `hyperdrone-*`, `render_bench` (raytracing en CPU) | `~/opt/assimp-5.4.3`, `~/opt/src/stb` |

Los entornos visuales necesitan las escenas ProcTHOR del dataset público `rl-tools/conta` (Hugging Face) en
`~/.cache/rl_tools/conta` (o `CONTA_ROOT` / `CONTA_CACHE`).

### Ejecutables

| Ejecutable | Para qué |
|---|---|
| `neat_control` | NEAT + SNN en un entorno de control (`--env`, `--selector static\|ap`, `--resample-train`, `--keep-state`, …) |
| `env_probe` | Sonda de entornos: dimensiones, retornos de referencia, cotas de observación y costo de la SNN |
| `rl_ref_eval` | Evalúa una política RL exportada en los mismos episodios que NEAT |
| `render_bench` | Costo de renderizar los entornos visuales en CPU |
| `xor_snn` | Verificación de NEAT + SNN en XOR |

### Catálogo de operadores

Por defecto, `neat_control` usa los 3 operadores de NEAT (mutar pesos, agregar conexión, agregar nodo). Con
`--catalog` se elige otro catálogo entre 15 operadores (por ejemplo, pesos finos, operador dirigido por la
población, sesgo y dinámica de neuronas, ganancia de entrada, duplicar neurona, borrar conexión o neurona), y con
`--p` sus probabilidades. `--credito-rango` registra por operador el percentil de sus hijos en cada generación.

```bash
build/bin/neat_control --env acrobot-swingup --resample-train \
    --catalog mutate_weights,add_connection,add_node,weights_fine,delete_connection \
    --p 0.6,0.1,0.05,0.2,0.05 --credito-rango --out /tmp/prueba
```
