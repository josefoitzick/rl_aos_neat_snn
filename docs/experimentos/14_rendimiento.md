# 14 · Rendimiento: simulador y paralelización

| | |
|---|---|
| Semana | S04 (2026-10-14) |
| Estado | aplicado en el repo (A, B y C) y verificado el 2026-10-14; binarios congelados en `~/tesis/runs/fase_ppo_bin` |
| Capítulo de la tesis | Anexo: implementación y costo computacional |

## Pregunta

¿Se puede iterar más rápido sin cambiar ningún resultado? Se buscaron dos cosas:
- cuellos de botella del simulador SNN;
- hebras ociosas en las corridas de NEAT y en el entrenador de PPO.

## Diseño

**Medición sobre los procesos en curso**, sin tocarlos:
- uso de CPU por proceso y por hebra (`ps -L`);
- carga del nodo.

**Perfil del simulador con callgrind:** cuenta instrucciones, así que no le afecta la carga. Usa un microbenchmark con redes de
la forma de los fenotipos evolucionados (política reactiva, reinicio por paso de control).

**Arreglo A (simulador)** (`snn/src/core/neuron.cpp`):
- los factores de decaimiento exp(−dt/τ) se calculan una vez, porque dt y τ son fijos;
- el ruido no se sortea cuando vale 0, que es el caso en todos los experimentos.

Las operaciones aritméticas son las mismas, en el mismo orden.

**Prototipo B (paralelización)** (`src/eval/evaluator.cpp`, `neat_control`):
- las tareas paralelas pasan de "un genoma" a "un episodio de un genoma";
- la validación del campeón (10 episodios por generación) y el test final (100) se reparten entre hebras;
- la media se suma en el mismo orden que antes.

**Verificación:** corridas cortas con el binario congelado `operadores_bin` (sha `ae239e87…`) y con las copias modificadas, en 7
configuraciones:
- acrobot-swingup: estático con P₀; estático con K = 10 uniforme; AP con K = 10; P en dos fases;
- bottleneck con K = 10;
- l2f con P₀;
- pendulum-meta con K = 10.

Se exigen `generations.csv` (sin la columna de segundos) y `summary.txt` idénticos. Prototipos en
`~/tesis/runs/optim_verif/`.

## Resultados

**Uso de CPU de los procesos en curso:**

| Proceso | Hebras | Uso de las hebras de trabajo |
|---|---|---|
| Prueba de margen (población 70–130) | 8 | 85 % (hebra principal 96 %) |
| bottleneck 300k (población 70) | 24 | **66 %** (hebra principal 89 %) |
| Entrenador de PPO (16 corridas) | 150 | 87 % |

Causa: la validación del campeón corre en serie (10 episodios contra los 700 de la población). Es ~1,4 % del trabajo, y por
Amdahl cuesta 10 % con 8 hebras y 26 % con 24.

**Perfil del simulador** (instrucciones; red de 44 neuronas y 400 sinapsis):
- original: generador aleatorio 30 %, exponenciales 21 %;
- con el arreglo A: 2,4× menos instrucciones. Lo restante es el paso de la neurona (~55 %) y la propagación de espigas (~40 %).

**Arreglo A, de punta a punta** (3 hebras por corrida, los dos binarios a la vez):

| Configuración | ¿Idéntico? | Aceleración |
|---|---|---|
| acrobot-swingup, estático P₀ | sí | 2,48× |
| acrobot-swingup, K = 10 uniforme | sí | 2,49× |
| acrobot-swingup, AP K = 10 | sí | 2,56× |
| acrobot-swingup, P en dos fases | sí | 2,41× |
| bottleneck, K = 10 | sí | 2,40× |
| l2f, P₀ | sí | 2,54× |
| pendulum-meta, K = 10 | sí | 2,48× |

Son las primeras generaciones, con redes chicas. Con redes grandes, la propagación de espigas pesa más: el microbenchmark da
1,7–2,7×.

**Prototipo B** (A + episodios repartidos + validación y test final en paralelo):
- Resultados idénticos al binario congelado en las 7 configuraciones.
- Con 3 hebras, la aceleración total frente al congelado es 2,59–2,95×.
- Pasan los 88 tests.
- **Con 24 hebras**, A sola frente a A + B a la vez (bottleneck #96, 30 generaciones):

  | Versión | Uso de CPU | Tiempo |
  |---|---|---|
  | A sola | 62 % | 43,1 s |
  | A + B | **96 %** | **25,6 s** (1,68× más rápido) |

  Las salidas son idénticas.

**Modelo del entrenador de PPO**, por generación, en unidades de episodio. Calibrado con el 87 % medido con 150 hebras y
N = 16:

| Hebras | Código actual, N = 16 | Código actual, N = 32 | Con B completo (validación y Φ también repartidas), N = 16 |
|---|---|---|---|
| 150 | 85 % | 92 % | 98 % |
| 300 | 74 % | 85 % | 96 % |
| 576 | 60 % | 75 % | 92 % |

## Conclusión y límites

- El arreglo A acelera todas las corridas ~2–2,5× sin cambiar ningún resultado. Las corridas ya hechas siguen siendo
  comparables, y se pueden reutilizar.
- B elimina las hebras ociosas: el uso pasa del 62–66 % al 96 % con 24 hebras. Así se pueden dar más hebras a las corridas
  largas (bottleneck) y acortar la ruta crítica. En el entrenador de PPO falta repartir también la validación y Φ entre las
  corridas del pool.
- Lo que **no** conviene: `-march=native` en todo el código, `float` o `-ffast-math`. Cambian el redondeo (por ejemplo, FMA en
  la dinámica de los entornos) y romperían la comparabilidad con lo ya corrido.
- **Límite:** la verificación cubre las primeras generaciones de 7 configuraciones. La adopción se verificaría además con los
  tests, la V1 contra los binarios congelados y la V4 (reanudar el entrenamiento de PPO).

## Adopción en el repo (2026-10-14)

Se decidió aplicar las tres mejoras antes del entrenamiento de PPO y de la comparación final.

| Mejora | Archivos | Qué cambia |
|---|---|---|
| A | `snn/include/core/neuron.hpp`, `snn/src/core/neuron.cpp` | Decaimiento en caché; sin sorteo de ruido con noise = 0 |
| B | `src/eval/evaluator.{hpp,cpp}`, `src/experiment/neat_control.cpp` | Tareas por episodio en `evaluatePopulation(s)`; validación y test final con `evaluateGenomeParallel` |
| C | `src/aos/neat_run.{hpp,cpp}`, `src/eval/evaluator.{hpp,cpp}` | `eval::evaluateGenomes` mide varios genomas en lote. `NeatPool` mide así la validación de los campeones, Φ y el test final de todas sus corridas, con los episodios repartidos entre hebras. `NeatRun` expone esos trabajos (`validationJob`, `phiJob`, `finalJob`) y recibe los valores (`closeGeneration(v)`, `closeWindow(J)`); el uso secuencial no cambia |

**Verificación** (`~/tesis/runs/optim_verif/verificar_repo.sh`):
1. 88/88 tests;
2. V1 (`NeatRun`/`NeatPool` = `neat_control`; `neat_control` nuevo = `operadores_bin`; control negativo);
3. las 7 configuraciones contra el binario congelado;
4. V4 (reanudar el entrenamiento de PPO es exacto);
5. entrenador nuevo = entrenador anterior en el mismo entrenamiento de V4.

## Datos y reproducción

- `~/tesis/runs/optim_verif/`:
  - `micro/` (microbenchmark, perfiles callgrind);
  - `repo_a` y `build_a` (arreglo A), `repo_b` y `build_b` (A + B);
  - `verificar_a.sh`, `verificar_b.sh` y sus resúmenes en `ver_a/resumen.txt` y `ver_b/resumen.txt`.

## Documentos relacionados

- [11](11_neat_como_entorno.md) (verificaciones bit a bit del entorno de PPO) · `docs/diseno_ppo.md` §4.2 (costo del entrenamiento).

## Resultado de la verificación y binarios congelados

`verificar_repo.sh` sin fallas:
- 88/88 tests;
- V1: 9 casos y la comparación con el binario congelado, más el control negativo;
- las 7 configuraciones, idénticas y 2,5–2,7× más rápidas;
- V4: reanudar es exacto;
- el entrenador nuevo es idéntico al anterior en `ppo.csv`, `corridas.csv`, `decisiones.csv`, el actor y la validación.

Binarios congelados en `~/tesis/runs/fase_ppo_bin` para el entrenamiento de PPO y la comparación final:

| Binario | sha |
|---|---|
| `neat_control` | `52e237c8…` |
| `aos_ppo_train` | `7b2e0538…` |
| `aos_neat_check` | `f3493257…` |

El código está identificado por el commit base y el diff sin commitear (`fuente_sin_commit.patch`).
