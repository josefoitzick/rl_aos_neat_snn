# Inventario de entornos de RLtools (2026-10-03)

Base para la selección de entornos (`diseno_neat_ap.md` §15.2). Fuentes revisadas:
- RLtools rama `main` @ `b32d998` (2026-03-23) y rama `render` @ `6bb2fda` (2026-09-15). `render` es `main` + 1115 commits y trae entornos que `main` no tiene (reacher, hyperdrone, l2f_visual).
- el *zoo* (`src/rl/zoo/`), que es la configuración "oficial" con la que RLtools publica benchmarks.

**[D] 2026-10-03: versión única de RLtools = `render` @ `6bb2fda`** (`external/rl-tools` en ese commit, *detached*).
En pendulum y acrobot el cambio respecto de `main` es solo de tipos, pero `uniform_real_distribution` normaliza
distinto: los estados iniciales cambian levemente, así que las corridas anteriores (calibración, reunión 2)
no se reproducen bit a bit con esta versión. Quedan como exploratorias; el barrido vuelve a correr pendulum
y acrobot como anclas.

Codificación (SMALL, diferencial): **entradas SNN = 2 × obs** (1 × obs si la observación es no negativa,
como en imágenes), **salidas SNN = 2 × acción**, 50 ms de SNN por paso de control.

## 1. Tabla general

| Entorno (variante) | Rama | Zoo | Obs | Acción | Pasos máx. | Recompensa / término | Memoria | Entradas → salidas SNN |
|---|---|---|---|---|---|---|---|---|
| **pendulum-v1** | main | ✓ | 3 (cos θ, sin θ, θ̇) | 1 | 200 | −(θ² + 0,1θ̇² + 0,001u²); sin término | No | 6 → 2 |
| pendulum-position-v1 | main | ✓ | 2 (solo ángulo) | 1 | 200 | Igual que pendulum; falta la velocidad | **Sí** | 4 → 2 |
| pendulum-velocity | main | — | 1 (solo θ̇) | 1 | 200 | Igual; falta la posición | **Sí** | 2 → 2 |
| pendulum-multitask-v1 | main | ✓ | 4 | 1 | 200 | La acción se invierte al azar por episodio; la 4.ª observación indica cuál | No | 8 → 2 |
| pendulum-meta | main | — | 4 | 1 | 200 | Igual, pero la inversión **no** se observa (4.ª = última acción): hay que inferirla | **Sí** | 8 → 2 |
| **acrobot** (el que usamos) | main | — | 6 | 1 | 500 | −1 por paso; termina al alcanzar la altura meta (estilo gym). DT 0,2, torque ±5 (lo bajamos a ±1) | No | 12 → 2 |
| **acrobot-swingup-v0** | main | ✓ | 6 | 1 | 500 | Costo moldeado: subir **y equilibrar** invertido; sin término. Inicio aleatorio; DT 0,02, torque ±5 | No | 12 → 2 |
| car | main | — | 6 | 2 (acelerador, dirección) | **sin límite definido** | exp(−5·costo) por quedarse cerca del origen; termina fuera de ±1 m | No | 12 → 4 |
| car-track | main | — | 9 (6 + 3 rayos) | 2 | sin límite definido | Velocidad vx; termina al salir de la pista | No | 18 → 4 |
| **flag** | main | ✓ | 11 | 2 (aceleración xy) | 50 (zoo) / 200 | +1000 por bandera visitada (2 en orden); −100 y fin al tocar el borde | **Sí**: las banderas solo se ven en el paso 0 | 22 → 4 |
| **memory** | main | — | 1 (bit de entrada) | 1 | 20 | −10·\|u − (unos en los últimos 10 pasos)/10\| | **Sí, es el objetivo** | 2 → 2 |
| **reacher-v0** | render | ✓ | 4 (posición, objetivo) | 2 (velocidad xy) | 40 | −distancia; termina al llegar (r < 0,1) | No | 8 → 4 |
| reacher-visual-v0 | render | ✓ | 16×16×3 = 768 (imagen) | 2 | 40 | Igual que reacher | No | 768 → 4 (píxeles ≥ 0) |
| reacher-visual-memory(-hard)-v0 | render | ✓ | 768 (imagen) | 2 | 40 / 80 (2 objetivos) | Igual; objetivos secuenciales | **Sí** | 768 → 4 |
| **l2f** (cuadricóptero Crazyflie) | main | ✓ | 18 + 64 de historial de acciones = 82 (variante *tiny*: historial 2) | 4 (rotores) | 500 (5 s a 100 Hz) | Costo cuadrático de posición, orientación, velocidad y acción; termina al alejarse 1 m o con velocidades extremas | No (el historial suple) | 164 → 8 (*tiny*: obs 29 → 58 → 8; 250 pasos a 50 Hz, seguimiento de trayectoria) |
| l2f_visual | render | ✓ | Cámara 64×64 | 4 | 500 | Igual que l2f | — | Miles → 8 |
| hyperdrone | render | — | Cámaras con *raytracing*; multiagente | 4·N | — | Tareas: *moving gate*, *target frame*, localización visual-inercial | — | Miles |
| bottleneck-v0 (multiagente) | main | ✓ | 5 agentes × 12 (incluye lidar de 5 rayos) = 60 | 5 × 2 = 10 | 200 | Cruzar un cuello de botella sin chocar | No | 24 → 4 por agente (política compartida) o 120 → 20 conjunta |
| ant-v4 (MuJoCo) | main | ✓ | 27 | 8 | 1000 | Locomoción estilo gymnasium | No | 54 → 16 |
| *batch* | render | — | — | — | — | No es un entorno: envoltorio para correr N instancias en paralelo | — | — |

En **negrita**, los que hoy veo como candidatos de primera línea.

## 2. Clasificación para el barrido (etapa A)

**A1. Usables directamente** (vector de estado, un agente, solo encabezados): pendulum-v1, acrobot,
acrobot-swingup-v0, car y car-track, reacher-v0, l2f (*tiny* y completo), flag, memory, variantes de pendulum.

**A2. Requieren adaptación:**
- **bottleneck:** multiagente. La opción natural es una política compartida, con la misma red evaluada por cada agente sobre su observación local.
- **car:** no define un límite de pasos, hay que fijarlo nosotros.
- **pendulum-multitask:** revisar qué varía entre tareas.

**A3. Requieren infraestructura extra. [D] 2026-10-03: se prueban igual, no se descarta ningún entorno sin probarlo.**
- **Observación por imagen** (reacher-visual*, l2f_visual, hyperdrone): de 768 a miles de entradas.
  - Para reacher-visual, los píxeles son ≥ 0, así que basta 1 neurona de entrada por píxel (768 en 16×16×3), no la codificación diferencial.
  - Hay que medir el costo del genoma inicial (totalmente conectado a las entradas) y de la SNN.
  - hyperdrone renderiza con *raytracing*: medir su costo en CPU.
- **ant-v4:** instalar la biblioteca de MuJoCo dentro del home (sin root) y medir el costo de 1000 pasos × 50 ms de SNN.

Criterio del usuario: la tesis trata de AOS **con estado**, así que interesan justamente los problemas complejos. Un entorno solo se excluye por un motivo medido (costo inviable en este hardware, o fracaso total de NEAT estático) y el motivo queda reportado.

## 3. [!] Hallazgo: la política actual no tiene memoria entre pasos

`src/eval/evaluator.cpp` llama a `net.fastReset()` en **cada paso de control**. NEAT sí crea conexiones
recurrentes (`Genome::mutateAddConnection`), pero el estado de la red (potenciales y sinapsis) se borra
antes de cada observación. Hoy la SNN es una **política reactiva**.

Consecuencias:
- **Los entornos que exigen memoria quedan sin solución por diseño:** memory, flag, pendulum-position y pendulum-velocity, reacher-visual-memory. Si se incluyen en el barrido así, aparecerán como "imposibles" por un motivo del evaluador, no del entorno.
- **Opción:** agregar al evaluador un modo `--keep-state`, que reinicia solo al comienzo del episodio. Con eso la dinámica temporal de la SNN y las conexiones recurrentes sirven como memoria, y es un argumento natural a favor de las SNN.
- **Costo:** se pierde la independencia entre pasos, y en entornos sin memoria podría desestabilizar (actividad residual).

**[D] 2026-10-03: implementado** como opción `--keep-state` del evaluador (`EvalConfig::keep_state`; `snn/` sin
cambios). Sin la opción, el resultado es idéntico al evaluador anterior (test de regresión en `tests/eval_tests.cpp`).
Por defecto se mantiene la política reactiva; en los entornos de memoria se usa `--keep-state`.

## 4. Lista para el barrido (etapa A): todos los entornos

| Grupo | Entornos | Para qué |
|---|---|---|
| Control clásico (referencia) | pendulum-v1, acrobot ±1, acrobot ±5 | Anclan contra los resultados ya obtenidos |
| Control más difícil | acrobot-swingup-v0, car, car-track, reacher-v0, pendulum-multitask | Dificultad moderada |
| Alta dimensión | l2f-tiny, l2f, bottleneck (política compartida), ant-v4 | Mide si NEAT + SNN escala |
| Memoria | memory, flag, pendulum-position, pendulum-velocity, pendulum-meta | Explota la dinámica temporal de la SNN (requiere el modo con estado, §3) |
| Visuales | reacher-visual, reacher-visual-memory(-hard) | Imagen 16×16 sin escenas 3D |
| Visuales con escenas 3D (extensión B, §6) | l2f-visual, hyperdrone-target-frame, hyperdrone-moving-gate (con memoria y reactivos) | Límite superior de complejidad |

**Estado (2026-10-03):**
- Etapa A: 27 variantes × 5 semillas en `~/tesis/runs/barrido_etapa_a` (tmux `barrido_a`).
- Extensión B: 6 variantes × 5 semillas en `~/tesis/runs/barrido_etapa_b` (tmux `barrido_b`, arranca al terminar A).
- Excluido con motivo: visual_inertial_localization (no es una tarea de control).

## 5. Implementación y sonda (2026-10-03)

**Código:**
- `src/envs/rltools_adapter.hpp`: adaptador genérico.
  - Normalización afín por dimensión, `(x − centro)/escala`, recortada a ±1, con envoltura para los ángulos.
  - Multiagente con política compartida.
  - Observación no negativa con 1 entrada por dimensión.
- `src/envs/rltools_envs.cpp`: registro de los entornos. Usa una sola unidad de compilación, porque RLtools define funciones no-*inline* en sus encabezados.
- `src/experiment/env_probe.cpp`: la sonda. Resultados en `~/tesis/runs/env_probe_v2*/` (`probe.csv`, `obs_stats.csv`).

**Detalles por entorno:**
- **car:** `sample_initial_state` de RLtools no compila (bug: llama a `initial_state` sin `parameters`). Se replica su intención en el adaptador. Límite de 500 pasos para car y 1000 para car-track, como los entrenadores de RLtools en `src/rl/environments/car/`. car-track usa la pista por defecto compilada; la versión de CPU carga un `.bmp` relativo al directorio de trabajo y se evita.
- **flag:** las banderas se sortean en `sample_initial_parameters`, que el adaptador llama al comienzo de cada episodio.
- **bottleneck:** el lidar mide la distancia hasta el obstáculo más cercano, muros incluidos; RLtools no usa `LIDAR_RANGE`. Cota = diagonal del área (28,3). Un agente muerto observa lidar 0, que queda en −1.
- **l2f y l2f-tiny:** fábricas del zoo tal cual (`ENVIRONMENT_FACTORY` y `ENVIRONMENT_TINY_FACTORY`).
- **ant:**
  - MuJoCo 2.3.5, la versión que fija RLtools, compilado desde el código fuente en `~/opt/mujoco-2.3.5` (sin root).
  - CMake lo importa directo, porque su `mujocoConfig.cmake` exige OpenGL (solo para renderizar), que mullo no tiene.
  - Es opcional: sin MuJoCo, `ant` no se registra.
- **Cotas sin límite analítico** (velocidades de car y de ant): máximo de |x| bajo acciones aleatorias, redondeado hacia arriba.

**Sonda:** 20 episodios por entorno, con acción aleatoria U(−1, 1) y con acción nula. Costo de la SNN = genoma mínimo de NEAT (totalmente conectado), con política reactiva.

| Entorno | Agentes | Entradas → salidas SNN | Pasos máx. | Retorno aleatorio | Retorno acción nula | SNN ms/episodio* | s/generación** |
|---|---|---|---|---|---|---|---|
| pendulum | 1 | 6 → 2 | 200 | -1347.0 | -1301.0 | 1.8 | 0.2 |
| pendulum-position | 1 | 4 → 2 | 200 | -1347.0 | -1301.0 | 1.3 | 0.1 |
| pendulum-velocity | 1 | 2 → 2 | 200 | -1347.0 | -1301.0 | 0.8 | 0.1 |
| pendulum-multitask | 1 | 8 → 2 | 200 | -1326.4 | -1301.0 | 2.3 | 0.2 |
| pendulum-meta | 1 | 8 → 2 | 200 | -1326.4 | -1301.0 | 2.2 | 0.2 |
| acrobot (±5) | 1 | 12 → 2 | 500 | -135.6 | -500.0 | 7.9 | 0.7 |
| acrobot-swingup | 1 | 12 → 2 | 500 | -18.0 | -18.1 | 7.8 | 0.7 |
| car | 1 | 12 → 4 | 500 | 83.5 | 153.4 | 5.0 | 0.5 |
| car-track | 1 | 18 → 4 | 1000 | -8.0 | 0.0 | 23.7 | 2.2 |
| reacher | 1 | 8 → 4 | 40 | -35.0 | -35.7 | 0.5 | 0.0 |
| flag | 1 | 22 → 4 | 50 | 230.0 | 50.0 | 0.2 | 0.0 |
| memory | 1 | 2 → 2 | 20 | -120.7 | -78.9 | 0.1 | 0.0 |
| l2f-tiny | 1 | 58 → 8 | 250 | -0.4 | 0.8 | 1.6 | 0.1 |
| l2f | 1 | 164 → 8 | 500 | 2.2 | 3.5 | 4.3 | 0.4 |
| bottleneck | 5 | 24 → 4 | 200 | -4.9 | 0.0 | 30.9 | 2.9 |
| ant | 1 | 54 → 16 | 1000 | -61.0 | 999.2 | 273.0 | 25.6 |
| reacher-visual | 1 | 768 → 4 | 40 | -35.0 | -35.7 | 19.2 | 1.8 |
| reacher-visual-memory | 1 | 768 → 4 | 40 | -35.0 | -35.7 | 32.4 | 3.0 |
| reacher-visual-memory-hard | 1 | 768 → 4 | 80 | -154.6 | -153.4 | 64.7 | 6.1 |

\* Un hilo. El largo real del episodio depende de la política (los que terminan pronto, como l2f, salen baratos).
\** Estimación con población 150 × E = 10 episodios en 16 hilos. La red crece durante la evolución, así que el costo real será mayor.

**Lectura preliminar:**
- **Referencia del umbral normalizado (§9.3):** `f_ref = max(aleatoria, nula)`. En ant, la acción nula da ~1000, porque quedarse quieto cobra la recompensa de "sano" en cada paso. Ese es el piso que una política tiene que superar.
- **acrobot ±5** se resuelve casi al azar (−136 contra −500 sin acción): confirma que es trivial (§14.2).
- **Costo:** todo es viable en mullo. ant es el más caro (~26 s/generación con la red mínima, ~1,5 h para 200 generaciones con 16 hilos).

**Pendiente (paso 5):** l2f_visual y hyperdrone.
- Necesitan las escenas 3D de ProcTHOR (`.glb`, que se descargan del dataset `rl-tools/conta` en Hugging Face) y el *raytracer* de RLtools.
- Sus entrenadores oficiales son CUDA, aunque existe `src/rl/environments/l2f_visual/training.cpp` para CPU.
- La integración es aparte: cargar la escena, el renderizador y las anotaciones de espacio libre.

## 6. Entornos visuales con escenas 3D: l2f-visual (paso B, 2026-10-03)

**Dependencias, compiladas en el home sin root:**
- assimp 5.4.3, solo con el lector glTF, en `~/opt/assimp-5.4.3`;
- stb en el commit fijado por RLtools (`f1c79c0`), en `~/opt/src/stb`;
- backend de raytracing `GENERIC` (CPU).

CMake los detecta solos; si faltan, `l2f-visual` no se registra y el resto compila igual.

**Escenas:** dataset `rl-tools/conta` (Hugging Face, 25 archivos, 395 MB, SHA-1 verificado) en `~/.cache/rl_tools/conta`.
El catastro (`render_bench` sobre las 23 escenas glTF) mostró que solo hay **3 casas ProcTHOR distintas**, cada una
en dos versiones (misma geometría); el resto son objetos sueltos (1–5 mallas, ~1 m):

| Casa | Mallas | Tamaño | Versiones (SHA-1) | Uso |
|---|---|---|---|---|
| 0 (ProcTHOR-Train-0) | 95 | 8 × 6 m | `a8fda3e9` (RLtools), `28d17959` | Entrenamiento (`a8fda3e9`) |
| 1 (ProcTHOR-Train-1) | 655 | 16 × 18 m | `7f1c9129` (RLtools), `75d1d67a` | Entrenamiento (`7f1c9129`) |
| 2 | 124 | 9,4 × 5,7 m | `44b5c137`, `c6e7892a` | **Test** (ambas versiones): casa no vista |

**Costo de renderizar en CPU** (`render_bench`, 1 hilo): 0,02–0,09 ms por imagen a 16×16, 0,28 ms a 32×32 y 1,1 ms a 64×64.
El cuello de botella es la SNN, no el render. NEAT parte totalmente conectado: 64×64×3 = 12 288 entradas × 8 salidas
≈ 98 000 conexiones (~100 h por corrida), así que es inviable.

**[D] Cámara 16×16 RGB** (768 entradas, igual que reacher-visual).

**Adaptador** (`l2fv::L2fVisualEnv` en `src/envs/rltools_envs.cpp`):
- Fábrica del zoo (`l2f_visual::ENVIRONMENT_FACTORY`) con `NUM_ENVS = 1`.
- "Modo objetivo", como el entrenador de CPU de RLtools: dinámica y recompensa de l2f (estabilizar desde hasta 90°), con la casa trasladada para que el dron quede en una posición libre, la primera de las anotaciones.
- Cada instancia (una por hilo) carga las escenas de su partición con su propio renderizador.
- La casa de cada episodio es `semilla mod 2`, así que los números aleatorios son comunes a todos los genomas.
- `EnvOptions::test_split`: `neat_control` mide al campeón (test por generación y test final) en las escenas de test. Los demás entornos ignoran la opción.

**Sonda** (`~/tesis/runs/env_probe_l2fv`):

| Entorno | Entradas → salidas | Pasos máx. | Aleatoria | Nula | µs/paso entorno | SNN ms/episodio |
|---|---|---|---|---|---|---|
| l2f-visual | 768 → 8 | 500 | 2,78 | 4,52 | 120 (incluye render) | 40 |

Los episodios aleatorios duran ~46 pasos, porque el dron se cae y el episodio termina. Una política que lo estabilice
alarga los episodios hasta 500 pasos, y el costo por generación crece en esa proporción.

**Prueba de humo:** `neat_control` con `--keep-state`, 2 generaciones; corre, y el test en la casa 2 funciona (6 s por generación con E = 2 y 4 hilos).

### 6.1 hyperdrone (paso B4, 2026-10-03)

Tres tareas en `include/rl_tools/rl/environments/hyperdrone/tasks/`:

| Tarea | De qué trata | Decisión |
|---|---|---|
| **target_frame** | Servo-control visual: el dron ve su cámara y una imagen objetivo tomada desde la pose meta; debe volver a esa pose | **Integrada**: `hyperdrone-target-frame` |
| **moving_gate** | Pasar volando por una "puerta" que oscila (+10 al pasar; fin si choca) | **Integrada con sustituto**: `hyperdrone-moving-gate` |
| visual_inertial_localization | Un autopiloto vuela una ruta y se evalúa la estimación de pose (benchmark tipo OpenVINS) | **Excluida**: no es una tarea de control |

**Adaptador** (`hd::HyperdroneEnv`): hyperdrone trabaja por lotes (tensores de instancias con máscara de reinicio), y aquí se usa un lote de 1 instancia expuesto como `ControlEnv`.
- **Por paso:** `render` → `observe` → `step` / `reward` / `terminated`.
- **Escena:** `select_scene(semilla mod 2)`, con las mismas casas y particiones que l2f-visual.
- **Cámara:** 16×16 RGB. target-frame observa cámara + imagen objetivo (1536 entradas); moving-gate, solo la cámara (768).

**MDP:** se toma de `src/rl/environments/l2f_visual/training_hyperdrone_config.h`, el entrenador oficial de hyperdrone:
- Crazyflie con cámara (`crazyflie_openmv`).
- Recompensa cuadrática con −100 al terminar.
- Terminación a 1 m o con velocidades extremas.
- Inicio en la pose meta con inclinación ≤ 0,3 rad.

**[!] Los ejemplos de video (`demo_common.h`) no sirven:** su recompensa tiene todos los pesos en 0 y nunca terminan. La primera versión del adaptador los usaba y daba recompensa 0; se corrigió.

**Diferencias con el entrenador oficial:**
- 500 pasos (5 s, como l2f) en vez de 10 000.
- Cámara 16×16.
- Observación solo de imagen: el oficial agrega orientación, giro e historial de acciones, y apila 10 cuadros; aquí la memoria la da `--keep-state`.
- Sin aleatorización de dominio.

**Puerta sustituta (moving-gate):** el ejemplo oficial dibuja la puerta con `x500.glb`, de `tests/data` de RLtools, que no es público (el dataset `rl-tools/test-data` de Hugging Face pesa 2,5 KB). Los 17 objetos sueltos de `rl-tools/conta` son **modelos de drones** con la convención `body` / `prop_0..3` de hyperdrone; se usa `87d2b8b1` (5 mallas, ~1 m). Pasar o chocar se calcula geométricamente (radio de apertura 0,4 m) y no depende del modelo 3D, que solo cambia la apariencia.

**Sonda** (`~/tesis/runs/env_probe_hyperdrone`):

| Entorno | Entradas → salidas | Aleatoria | Nula | µs/paso entorno | SNN ms/episodio | Pasos medios (aleatoria) |
|---|---|---|---|---|---|---|
| hyperdrone-target-frame | 1536 → 8 | −205,5 | −169,3 | 364 | 161 | 51 |
| hyperdrone-moving-gate | 768 → 8 | −189,0 | −151,0 | 276 | 70 | 50 |

**Prueba de humo** (`neat_control --keep-state`, 2 generaciones, E = 2, 4 hilos): ~12 s por generación en ambas, con el test en la casa 2 funcionando.
Con E = 10 serían ~60 s por generación en 4 hilos. El costo crece si NEAT logra episodios más largos.

### 6.2 Equivalencia de binarios (paso B5, 2026-10-03)

`scripts/equivalencia_binarios.sh` corre el binario congelado del barrido A (sha256 `dfd3c0c1…`) y el binario
con l2f-visual y hyperdrone (`1a5d7d99…`) con la misma configuración corta. Compara `generations.csv` (sin la
columna de tiempo) y `summary.txt` (sin la duración) byte a byte.

**Resultado: IDÉNTICOS en los 10 casos.** Cubren todos los caminos de código: pendulum, acrobot ±1, memory y flag
con `--keep-state`, bottleneck (multiagente), reacher-visual (imagen), car (estado inicial propio), car-track,
l2f-tiny (zoo) y ant (MuJoCo).

**Control negativo:** la misma comparación entre semillas distintas detecta la diferencia. Por lo tanto, A y B
forman un solo experimento.

Nota: la primera versión del script daba un falso "idéntico" (no creaba las carpetas y comparaba archivos ausentes).
Ahora falla explícitamente si falta un resultado, e incluye el control negativo.

## 7. Resultados de los barridos A y B (2026-10-04)

165 corridas: 33 variantes × 5 semillas × 500 generaciones, NEAT estático, sin errores ni corridas omitidas.
- A: 21:14 → 05:58, binario dfd3c0c1.
- B: 05:58 → 14:06, binario 1a5d7d99. Es equivalente a A (§6.2).

Métricas generadas con `scripts/analisis_barrido_ab.py`:
- **Trivial:** máximo entre los retornos aleatorio y de acción cero del probe (§5).
- **z:** (test − trivial) / sd entre semillas.
- **Brecha:** (train − test) / (train − trivial). 0 = generaliza todo; ≥ 1 = se pierde todo lo aprendido.
- **g90:** generación en que el máximo acumulado de best_train alcanza el 90 % de su mejora.

| Variante | obs/act/agentes | Trivial | Test ± sd | Train | z vs trivial | Brecha | g90 | Mejora 2ª mitad | Ocultas / conexiones | h/corrida |
|---|---|---|---|---|---|---|---|---|---|---|
| ant | 27/8/1 | 999.2 | 1139 ± 1.2e+02 | 1221 | 1.2 | 0.37 | 328 | 0.26 | 14 / 910 | 7.6 |
| reacher-visual-memory-hard_memoria | 768/2/1 | -153.4 | -169.4 ± 7.8 | -63.58 | -2 | 1.2 | 129 | 0.09 | 20 / 3135 | 6.3 |
| reacher-visual-memory-hard_reactivo | 768/2/1 | -153.4 | -152.5 ± 4.9 | -74 | 0.18 | 0.99 | 291 | 0.11 | 18 / 3130 | 6.3 |
| reacher-visual-memory_memoria | 768/2/1 | -35 | -44.66 ± 2.7 | -4.015 | -3.6 | 1.3 | 79 | 0.02 | 11 / 3109 | 1.9 |
| reacher-visual-memory_reactivo | 768/2/1 | -35 | -40.56 ± 2.1 | -10.08 | -2.7 | 1.2 | 146 | 0.07 | 10 / 3106 | 2.2 |
| bottleneck | 12/2/5 | 0 | 15.26 ± 9.1 | 28.82 | 1.7 | 0.47 | 329 | 0.18 | 13 / 139 | 4.0 |
| car-track | 9/2/1 | 0 | 599.7 ± 52 | 846.7 | 12 | 0.29 | 139 | 0.04 | 9 / 105 | 2.8 |
| reacher-visual | 768/2/1 | -35 | -35.81 ± 6.3 | -3.272 | -0.13 | 1 | 57 | 0.01 | 11 / 3114 | 2.4 |
| acrobot-swingup | 6/1/1 | -17.95 | -13.36 ± 1.2 | -11.23 | 3.8 | 0.32 | 151 | 0.05 | 15 / 68 | 1.6 |
| acrobot_t5 | 6/1/1 | – | -17.91 ± 0.8 | -15.98 | – | – | 187 | 0.08 | 13 / 60 | 0.1 |
| acrobot_t1 | 6/1/1 | -135.6 | -100.6 ± 3.6 | -84.46 | 9.6 | 0.32 | 71 | 0.02 | 17 / 71 | 0.6 |
| car | 6/2/1 | 153.4 | 398.8 ± 23 | 443.1 | 11 | 0.15 | 16 | 0.00 | 10 / 73 | 1.5 |
| l2f | 82/4/1 | 3.457 | 4.791 ± 1.7 | 12.7 | 0.76 | 0.86 | 208 | 0.11 | 8 / 1338 | 0.9 |
| l2f-tiny | 29/4/1 | 0.8269 | 1.781 ± 0.39 | 4.838 | 2.4 | 0.76 | 211 | 0.09 | 13 / 510 | 0.1 |
| pendulum | 3/1/1 | -1301 | -260.3 ± 61 | -187 | 17 | 0.066 | 157 | 0.10 | 16 / 62 | 0.4 |
| pendulum-multitask | 4/1/1 | -1301 | -722.3 ± 1.9e+02 | -440.5 | 3 | 0.33 | 348 | 0.20 | 18 / 72 | 0.5 |
| reacher | 4/2/1 | -35 | -13.57 ± 4.6 | -3.109 | 4.7 | 0.33 | 15 | 0.00 | 11 / 68 | 0.0 |
| memory_memoria | 1/1/1 | -78.9 | -40.06 ± 6.3 | -38.28 | 6.2 | 0.044 | 242 | 0.17 | 19 / 68 | 0.0 |
| memory_reactivo | 1/1/1 | -78.9 | -51.65 ± 1.2 | -51.8 | 22 | -0.0055 | 0 | 0.00 | 0 / 4 | 0.0 |
| flag_memoria | 11/2/1 | 230 | 225.2 ± 1.8e+02 | 1168 | -0.027 | 1 | 282 | 0.14 | 12 / 129 | 0.1 |
| flag_reactivo | 11/2/1 | 230 | 315.4 ± 1.2e+02 | 1258 | 0.73 | 0.92 | 262 | 0.12 | 10 / 118 | 0.1 |
| pendulum-position_memoria | 2/1/1 | -1301 | -934.8 ± 51 | -647.5 | 7.1 | 0.44 | 270 | 0.14 | 17 / 57 | 0.4 |
| pendulum-position_reactivo | 2/1/1 | -1301 | -1174 ± 86 | -871.8 | 1.5 | 0.7 | 241 | 0.10 | 18 / 59 | 0.4 |
| pendulum-velocity_memoria | 1/1/1 | -1301 | -1052 ± 59 | -947.6 | 4.2 | 0.3 | 285 | 0.19 | 17 / 56 | 0.4 |
| pendulum-velocity_reactivo | 1/1/1 | -1301 | -930.4 ± 31 | -864.9 | 12 | 0.15 | 129 | 0.03 | 13 / 37 | 0.3 |
| pendulum-meta_memoria | 4/1/1 | -1301 | -1120 ± 66 | -930.2 | 2.8 | 0.51 | 210 | 0.14 | 15 / 61 | 0.4 |
| pendulum-meta_reactivo | 4/1/1 | -1301 | -875.8 ± 1.3e+02 | -593.6 | 3.4 | 0.4 | 248 | 0.18 | 17 / 66 | 0.4 |
| hyperdrone-target-frame_memoria | 1536/4/1 | -169.3 | -165.6 ± 5.6 | -152.8 | 0.68 | 0.77 | 137 | 0.03 | 10 / 12324 | 7.8 |
| hyperdrone-target-frame_reactivo | 1536/4/1 | -169.3 | -174.4 ± 4.6 | -118 | -1.1 | 1.1 | 116 | 0.04 | 15 / 12339 | 7.2 |
| hyperdrone-moving-gate_memoria | 768/4/1 | -151 | -145.3 ± 20 | -107.7 | 0.29 | 0.87 | 152 | 0.05 | 14 / 6195 | 3.8 |
| hyperdrone-moving-gate_reactivo | 768/4/1 | -151 | -114.6 ± 4.5 | -105.2 | 8 | 0.2 | 73 | 0.02 | 12 / 6183 | 1.8 |
| l2f-visual_memoria | 768/4/1 | 4.518 | 3.916 ± 0.48 | 6.443 | -1.2 | 1.3 | 217 | 0.06 | 10 / 6180 | 4.6 |
| l2f-visual_reactivo | 768/4/1 | 4.518 | 4.132 ± 0.6 | 9.931 | -0.65 | 1.1 | 131 | 0.02 | 7 / 6167 | 4.2 |

acrobot_t5 no tiene trivial: el probe se hizo con torque ±1.

### 7.1 Lectura

**Grupos:**
- **NEAT resuelve o se estanca temprano (anclas):** pendulum, acrobot_t1, car, car-track, reacher (g90 ≤ 160; reacher y car ≈ 15).
- **NEAT aprende con dificultad (candidatos):**
  - pendulum-multitask, pendulum-meta (reactivo), pendulum-position y pendulum-velocity (memoria), memory (memoria): g90 de 210 a 350, z de 3 a 7;
  - bottleneck: ruidoso, g90 329;
  - acrobot-swingup;
  - ant: z = 1.2 respecto de la acción cero (999 por sobrevivir), pero sigue mejorando (26 % de la mejora en la 2ª mitad);
  - hyperdrone-moving-gate reactivo: el único visual que aprende (z = 8, brecha 0.2);
  - l2f y l2f-tiny: brecha alta.
- **NEAT no supera lo trivial:**
  - reacher-visual (las 4 variantes), l2f-visual y hyperdrone-target-frame: el problema parece ser la representación (256 píxeles como entradas directas, 3000 a 12000 conexiones), no los operadores;
  - flag: sobreajuste total (train ≈ 1200, test ≈ trivial).

**Memoria (`keep_state`):**
- **Imprescindible en memory:** sin estado la red no aprende nada (0 ocultas, 4 conexiones).
- **Ayuda en pendulum-position** y, levemente, en hyperdrone-target-frame.
- **Empeora en pendulum-velocity, pendulum-meta, flag, moving-gate y los reacher visuales.**

**Sobreajuste por diseño.** Cada corrida entrena con los mismos 10 episodios (semillas del stream 0). Las brechas ≥ 0.8 en flag, l2f y los visuales lo muestran. Ver `diseno_neat_ap.md` §15.4, paso 5.

**Operadores (con la P de Stanley fija):** add_node tiene la mayor tasa de mejora en casi todas las ventanas. El mejor operador cambia de 3 a 6 veces solo en entornos ruidosos (flag, pendulum-multitask, las variantes de pendulum con memoria). Estas corridas **no alcanzan** para medir la no estacionariedad: eso requiere sondas de P fija (fase posterior).

**Siguiente paso:** referencia RL por entorno (`diseno_neat_ap.md` §15.4).

## 8. Referencia RL (fase de comparación, 2026-10-04; en curso)

Plan en `diseno_neat_ap.md` §15.4. Objetivo: score = (NEAT − trivial) / (RL − trivial) por entorno.

### 8.1 Tubería

| Paso | Herramienta | Qué hace |
|---|---|---|
| Entrenadores del zoo | `scripts/ref_rl/compilar_zoo.sh` | 16 binarios (entorno × algoritmo) en `~/tesis/runs/ref_rl/bin`, sha256 en `SHA256SUMS` |
| Entrenadores propios | `scripts/ref_rl/compilar_propias.sh` + `src/ref_rl/*.h` | Copia generada de `zoo.cpp` con un gancho (`scripts/ref_rl/generar_zoo_tesis.py`, solo agrega `#ifdef`); 13 binarios en `bin_propias` |
| Entrenamiento | `scripts/ref_rl/lanzar_zoo.sh`, `lanzar_ref.sh` | 5 semillas por binario, 1 hilo por corrida, tmux, candado, `.running`, reanudable |
| Política → biblioteca | `scripts/ref_rl/compilar_politica.sh` + `src/experiment/rl_policy_shim.cpp` | `checkpoint.h` → `.so` con solo `rlp_*` visibles (no choca con nuestra unidad de RLtools) |
| Evaluación | `build/bin/rl_ref_eval`, `scripts/ref_rl/evaluar_zoo.sh` | Política en NUESTRO entorno con los MISMOS episodios finales de NEAT (`episodeSeeds(s, 100, 2)`, `test_split`, s = 1..5) |
| Resumen | `scripts/ref_rl/resumen_ref.py` | Tabla y `~/tesis/runs/ref_rl/score.json` |

Configuraciones propias (`src/ref_rl`):
- Hiperparámetros de la configuración del zoo (o de RLtools) más cercana, sin ajustar.
- Solo se cambian el entorno y el horizonte, que pasa a ser el nuestro.
- Entornos cubiertos:
  - acrobot ±5 y ±1 (SAC de acrobot-swingup);
  - pendulum-multitask (SAC-GRU, TD3 y PPO de pendulum-multitask-v1);
  - pendulum-velocity (SAC-GRU; TD3 y PPO de pendulum-position);
  - pendulum-position con SAC-GRU;
  - memory (SAC-GRU, con secuencias de 20);
  - l2f completo (`l2f/sac.h`, que el zoo no compila);
  - car y car-track (PPO de `src/rl/environments/car`).

### 8.2 Verificaciones

**Equivalencia del entorno.** `rl_ref_eval` reproduce bit a bit un bucle independiente escrito directamente con RLtools en double: pendulum, política SAC, semillas de NEAT 1 → −160.9120 en ambos.

**El retorno que reporta el zoo no es comparable:**
- su aritmética es float (en pendulum, ≈ 5 % de diferencia con los mismos episodios);
- su evaluación es la del penúltimo checkpoint;
- en l2f-tiny usa un horizonte de 500 pasos, mientras que el entorno y NEAT usan 250 (5 s a 50 Hz).

Se usa siempre nuestra evaluación. Si en algo sesga, el RL sale algo peor y el score de NEAT algo **optimista**, que es la dirección conservadora.

**Controles:**
- *Positivo:* la biblioteca reproduce los ejemplos de entrada y salida del checkpoint.
- *Negativo:* las políticas `zero` y `random` reproducen el trivial, y las dimensiones distintas se rechazan.
- *Óptimo de memory:* `src/ref_rl/memory_oracle.cpp`, con memoria perfecta de la ventana, da exactamente 0 en los 500 episodios.

**Bug de RLtools.** La exportación de actores TD3 escribe `td3_sampling::Configuration<float, …>`. Se parchea en una copia del checkpoint (`compilar_politica.sh`); la acción no cambia.

**car sin TD3.** El runner off-policy exige `ENVIRONMENT::EPISODE_STEP_LIMIT`, que Car no define.

**pendulum-multitask del zoo.** Usa la clase `PendulumMeta`: su referencia corresponde a nuestro **pendulum-meta**, y su SAC ya es recurrente (GRU).

### 8.3 Scores (v2, 2026-10-04 18:15)

Se regeneran con:
```
PARALELO=8 bash scripts/ref_rl/evaluar_zoo.sh
python3 scripts/ref_rl/resumen_ref.py
```
Falta pendulum-position SAC del zoo (≈ 02:00 del 2026-10-05). Las otras 3 referencias de pendulum-position ya están.

Lectura de la tabla:
- **NEAT** es el test final con 500 generaciones (barridos A y B).
- **± del score** es la sd de NEAT entre semillas, dividida por (RL − trivial).
- **óptimo analítico** es el oráculo de memory (`src/ref_rl/memory_oracle.cpp`).

| Variante NEAT | NEAT (500 gen) | RL (mejor entrenador) | Trivial | Score | Grupo |
|---|---|---|---|---|---|
| acrobot_t5 | -17.91 | -17 (tesis-acrobot-t5_sac) | -189.4 | 0.99 ± 0.00 | ancla |
| acrobot_t1 | -100.6 | -73.54 (tesis-acrobot-t1_sac) | -500 | 0.94 ± 0.01 | ancla |
| car-track | 599.7 | 641.4 (tesis-car-track_ppo) | 0 | 0.93 ± 0.08 | ancla |
| pendulum | -260.3 | -153 (pendulum-v1_sac) | -1252 | 0.90 ± 0.06 | ancla |
| car | 398.8 | 423.7 (tesis-car_ppo) | 183.3 | 0.90 ± 0.10 | ancla |
| reacher | -13.57 | -3.32 (reacher-v0_ppo) | -40.79 | 0.73 ± 0.12 | NEAT lucha |
| pendulum-velocity_reactivo | -930.4 | -705.7 (tesis-pendulum-velocity_sac) | -1252 | 0.59 ± 0.06 | NEAT lucha |
| memory_memoria | -40.06 | 0 (óptimo analítico) | -79.68 | 0.50 ± 0.08 | NEAT lucha |
| pendulum-multitask | -722.3 | -155.6 (tesis-pendulum-multitask_sac) | -1249 | 0.48 ± 0.18 | NEAT lucha |
| acrobot-swingup | -13.36 | -6.656 (acrobot-swingup-v0_sac) | -18.03 | 0.41 ± 0.11 | NEAT lucha |
| pendulum-meta_reactivo | -875.8 | -329.4 (pendulum-multitask-v1_sac) | -1249 | 0.41 ± 0.14 | NEAT lucha |
| pendulum-velocity_memoria | -1052 | -705.7 (tesis-pendulum-velocity_sac) | -1252 | 0.37 ± 0.11 | NEAT lucha |
| memory_reactivo | -51.65 | 0 (óptimo analítico) | -79.68 | 0.35 ± 0.02 | NEAT lucha |
| pendulum-position_memoria | -934.8 | -169.7 (pendulum-position-v1_td3) | -1252 | 0.29 ± 0.05 | NEAT lucha |
| bottleneck | 15.26 | 61.74 (bottleneck-v0_ppo) | 0 | 0.25 ± 0.15 | NEAT lucha |
| pendulum-meta_memoria | -1120 | -329.4 (pendulum-multitask-v1_sac) | -1249 | 0.14 ± 0.07 | fuera de alcance |
| pendulum-position_reactivo | -1174 | -169.7 (pendulum-position-v1_td3) | -1252 | 0.07 ± 0.08 | fuera de alcance |
| ant | 1139 | 6536 (ant-v4_ppo) | 1000 | 0.03 ± 0.02 | fuera de alcance |
| l2f | 4.791 | 30.55 (tesis-l2f-full_sac) | 4.28 | 0.02 ± 0.07 | fuera de alcance |
| l2f-tiny | 1.781 | 134.9 (l2f_td3) | 0.8496 | 0.01 ± 0.00 | fuera de alcance |

**Lectura:**
- **Anclas:** pendulum, car, car-track y acrobot ±1/±5.
- **NEAT lucha:**
  - pendulum-multitask (multitarea);
  - memory, pendulum-velocity y pendulum-position (memoria);
  - pendulum-meta (meta);
  - acrobot-swingup;
  - bottleneck (multiagente);
  - reacher (0,73, casi ancla).
- **Fuera de alcance:** ant (RL camina: 6536 contra 1139 de NEAT, con 1000 de trivial), l2f y l2f-tiny.
- **La memoria de la SNN (`keep_state`) empeora** en los péndulos de observación parcial: pendulum-meta da 0,14 con memoria contra 0,41 reactivo, y pendulum-velocity 0,37 contra 0,59. RL con GRU sí la aprovecha.

El barrido largo (§15.4.1 de `diseno_neat_ap.md`) dirá si con 1000–2000 generaciones algún entorno cambia de grupo, y qué efecto tiene la rotación de episodios.

### 8.4 Detalles técnicos adicionales

- **v2 (OpenBLAS, `~/opt/conda-blas`):** SAC ≈ 15× más rápido que el backend genérico de un hilo. v1 quedó interrumpida (carpetas `zoo/`, `zoo_propias/`).
- **bottleneck:** el zoo exporta un `multi_agent_wrapper`. `rl_policy_shim.cpp` evalúa su red interna (la política compartida) por agente. El autotest se hace con el envoltorio completo. Verificado: las demás políticas dan resultados idénticos antes y después del cambio.
- **car:** solo PPO (el entrenador de RLtools para CarTrack). TD3 requiere un `EPISODE_STEP_LIMIT` que Car no define.
- **`evaluar_zoo.sh` con `PARALELO`** (12 por omisión), para respetar el tope de hilos del nodo.
