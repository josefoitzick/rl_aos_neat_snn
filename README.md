# rl_aos_neat_snn

Código de la memoria de título **"Entrenamiento de Redes Neuronales de Espigas mediante Neuroevolución
con Selección Adaptativa de Operadores"**. Autor: José Foitzick Maripil. Profesor guía: Javier Baladron Pezoa.
Universidad de Santiago de Chile, 2026.

NEAT evoluciona redes de espigas (Izhikevich) para tareas de control. Un agente PPO (RLtools) selecciona
en cada generación cómo aplicar los operadores de mutación, a partir del estado de la población (AOS
basado en estado). Se compara contra probabilidades estáticas y contra Adaptive Pursuit.

## Estructura

```
snn/              Simulador SNN en C++ (basado en snn-simulator de dilgeto; ver snn/README.md)
src/neat/         Genoma, innovaciones, especiación, población, operadores de mutación
src/aos/          Selectores de operadores: estático | Adaptive Pursuit | PPO (misma interfaz)
src/envs/         Entornos de control de RLtools + codificación/decodificación hacia la SNN
src/eval/         Evaluación genoma → red → episodios → fitness (paralela y con semillas)
src/experiment/   Ejecutable principal, configuración y registro por generación
configs/          Configuraciones de experimentos
tests/            Tests (Catch2)
scripts/          Lanzadores
analysis/         Análisis estadístico y gráficos (Python)
docs/             Propuesta, notas de diseño y análisis del simulador
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
