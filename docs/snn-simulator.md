# snn-simulator: análisis en profundidad

> Escrito por Claude el 2026-09-26. Se leyó el código completo (~3,4k líneas), se compiló en mullo, se
> corrieron los tests y se hizo una medición de velocidad.
> **Incorporado al repo como [`snn/`](../snn/)** (2026-09-26), con autorización del autor (*dilgeto*), ya como código propio de la tesis.
> Origen: https://github.com/dilgeto/snn-simulator, commit `0633a1c` (2026-09-23, "aarch64"). Su historial git no se conservó. Licencia original: **GPL v3** (se mantiene `snn/LICENSE`).
> Al incorporarlo se eliminaron `build_cmake/` (binarios commiteados), `.claude/` y el submódulo vacío `external/rl-tools`; `patrones_disparo.png` y `voltajes.csv` se movieron a `snn/docs/`.
> Las rutas de este documento se refieren a la estructura original (hoy bajo `snn/`).
> Convención: **[!]** = bug o riesgo · **[→T]** = implicación para la tesis.

---

## 1. Qué es y de dónde viene

- Simulador de SNN en **C++17**, escrito por un compañero (*dilgeto*) para **su** tesis (repo iniciado en febrero de 2026, 36 commits).
- Está orientado a **WANN** (*Weight Agnostic Neural Networks*): las sinapsis **no tienen peso propio**. Toda la red usa **un único peso compartido** que se pasa al simular, y la evolución actúa sobre la topología, la polaridad y el tipo de neurona.
- **Replica en C++ el pipeline de nuestro paper.** Los encoders `SMALL` y `LARGE` están documentados en el código como "traducción directa del encoding de ANNarchy-NEAT (*bastianloyola*)", que es la base del paper Izhikevich. La corriente total también se define "como ANNarchy".
- Mención a la placa **ZCU104/PYNQ** y soporte **aarch64**: el compañero apunta a hardware embebido o FPGA.
- **Es solo el simulador (el fenotipo).** No hay NEAT: no hay genoma, innovaciones, especiación, cruce ni ciclo evolutivo. Las demos de RL usan redes fijas hechas a mano.

## 2. Estructura

```
include/core/      neuron.hpp  synapse.hpp  network.hpp  simulator.hpp
include/encoding/  encoder (base) · poisson · rate · ttfs · rlEncoder
include/decoding/  decoder.hpp  rlDecoder.hpp   ← [!] dos headers para la misma clase (ver §9)
src/core/          neuron.cpp network.cpp simulator.cpp main.cpp current_injection_main.cpp
src/encoding/      ...cpp
src/decoding/      decoder.cpp (compilado) · rlDecoder.cpp (NO compilado por CMake)
src/rl/            pendulum_eval.cpp  acrobot_eval.cpp   (demos con entornos de RLtools)
tests/core/        neuronTests · networkTests · izhikevichTests   (Catch2 v3)
scripts/           plot_current_injection.py
external/rl-tools  ← [!] submódulo git SIN .gitmodules → carpeta vacía (commit fijado b32d998)
```

**Targets CMake:**

| Target | Qué es | Depende de |
|---|---|---|
| `snn_core` | Neurona y red | — |
| `snn_encoding` | Encoders | — |
| `snn_engine` | Simulador y decoder | snn_core, snn_encoding, **OpenMP** |
| `snn_simulator` | Demo de población en paralelo (CLI) | CLI11 |
| `snn_current_injection` | Curvas V(t) por tipo de neurona → CSV | CLI11 |
| `snn_pendulum`, `snn_acrobot` | Demos RL | **RLtools** (header-only) |
| `snn_tests` | Tests unitarios | Catch2 3 (`-DBUILD_TESTING=OFF` lo omite) |

Flags de release: `-O3 -march=native`.

## 3. Modelo de neurona (`IzhikevichNeuron`)

Integración de Euler con **dos sub-pasos de dt/2** por paso (dt = 1 ms por defecto):

```
g_exc *= exp(-dt/tau_exc)        tau_exc = 5 ms  (AMPA)
g_inh *= exp(-dt/tau_inh)        tau_inh = 10 ms (GABA)
repetir 2 veces:
   I  = g_exc - g_inh + i_offset + noise·U(-0.5, 0.5)
   v += (0.04v² + 5v + 140 - u + I)·dt/2      (clamp de seguridad ±1e4)
   u += a(bv - u)·dt/2                         (clamp ±1e6)
si v ≥ 30:  spike; v = c; u += d
```

- Las **entradas sinápticas son conductancias con decaimiento exponencial**: un spike presináptico suma `w` a `g_exc` o a `g_inh` de la neurona postsináptica.
- Estado inicial y `reset()`: `v = c`, `u = b·c`.
- **25 tipos predefinidos** (`NeuronType`): las 6 clases corticales (RS, FS, IB, CH, LTS, RZ) y los 20 patrones de Izhikevich (2004). Algunos pares comparten (a,b,c,d) exactos: SPIKE_LATENCY = TONIC_SPIKING, INTEGRATOR = CLASS1, THRESHOLD_VARIABILITY = REBOUND_SPIKE. También hay un constructor con (a,b,c,d) libres.
- **Correspondencia con los modelos del paper**: los 8 existen tal cual.

| Paper | (a, b, c, d) | `NeuronType` |
|---|---|---|
| IZ-A | 0.02, 0.2, -65, 6 | `TONIC_SPIKING` |
| IZ-B | 0.02, 0.25, -65, 6 | `PHASIC_SPIKING` |
| IZ-C | 0.02, 0.2, -50, 2 | `TONIC_BURSTING` |
| IZ-D | 0.02, 0.25, -55, 0.05 | `PHASIC_BURSTING` |
| IZ-E | 0.02, 0.2, -55, 4 | `MIXED_MODE` |
| IZ-F | 0.01, 0.2, -65, 8 | `SPIKE_FREQUENCY_ADAPTATION` |
| IZ-G | 0.02, -0.1, -55, 6 | `CLASS1_EXCITABLE` |
| IZ-H | 0.2, 0.26, -65, 0 | `CLASS2_EXCITABLE` |

**Diferencias con el ANNarchy del paper**, a considerar si se quiere reproducir sus resultados:

| Aspecto | ANNarchy (paper) | snn-simulator |
|---|---|---|
| Estado inicial | v = -65, u = -14 | v = c, u = b·c (depende del tipo) |
| τ de conductancias | τ_I = 10 ms para exc e inh | τ_exc = 5 ms, τ_inh = 10 ms |
| Integración | Euler, 1 paso | Euler, 2 sub-pasos de dt/2 |
| Entrada de observación | Corriente `I` constante durante 50 ms | `setInputCurrents` **suma a g_exc en cada ms** (ver §6) |
| Pesos | Por conexión | **Un peso compartido** |

## 4. Red (`Network`)

- Vectores `neurons` y `synapses`, con listas de adyacencia `post_synapses[i]` y `pre_synapses[i]`, y listas de rol `input`/`hidden`/`output`.
- **Sinapsis** = (pre, post, excitatoria/inhibitoria, habilitada). **No tiene peso.** `getCurrent(w)` devuelve el peso compartido.
- **`step(shared_weight)`**:
  1. Los spikes del paso anterior se propagan como conductancia (retardo sináptico de 1 dt).
  2. Se integra cada neurona habilitada.
  3. `t += dt`.
- **Construcción:** `addInputNeuron`, `addHiddenNeuron`, `addOutputNeuron`, `addSynapse(pre, post, exc)`.
  - `addSynapse` rechaza autoconexiones.
  - Si `allow_recurrent=false`, rechaza también las que cierran un ciclo (DFS `canReach`). Por defecto la red es **recurrente**.
- **Mutaciones reversibles (estilo WANN):**
  - `insertNeuron(syn)`: deshabilita la sinapsis y crea neurona oculta + 2 sinapsis que heredan la polaridad. Es el *add node* de NEAT.
  - `enable`/`disable` de sinapsis y neuronas.
  - Cambio de polaridad y de tipo de neurona (vía `Individual`).
- **Eliminación física:** `removeSynapse` y `removeNeuron` con cascada, que borra las ocultas que quedan aisladas. **[!] Usan swap-and-pop, así que los IDs de neuronas y sinapsis cambian tras borrar.**
- `clone()`: copia profunda por valor (`new Network(*this)`).

**[→T]** Los IDs no son estables y no hay números de innovación. `Network` debe tratarse como **fenotipo**: el genoma NEAT vive aparte y se **construye una `Network` por evaluación**.

## 5. Simulador (`Simulator`)

- Mantiene una **población de punteros a `Network`**, con o sin ownership. `Individual` es un proxy para mutar una red concreta.
- `SimulationConfig`: `dt = 1`, `encoding_duration = 50`, `evaluation_duration = 100`, `verbose`.
- `simulate(idx, currents, w, T)`:
  - hace `fastReset()` y, durante T ms, llama a `setInputCurrents(currents)` y `step(w)` en cada paso;
  - devuelve `SimulationResult` con los tiempos de spike de cada salida, el voltaje final de las salidas y el total de spikes.
- `simulateEncoded(...)`: igual, pero con un `Encoder` de trenes de spikes (cada spike de entrada inyecta conductancia 100).
- **`simulateAll` y `simulateAllEncoded`: la población en paralelo con OpenMP** (`schedule(dynamic)`, un hilo por red).
- Utilidades de WANN:
  - `evaluateMultipleWeights` y `findBestWeight`: barren el peso compartido.
  - `evaluateFitness`: promedio sobre casos de prueba.
- `runEpisode(...)`: bucle de RL con callback. **[!] Está roto** (§9).

## 6. Codificación y decodificación

**Encoders de trenes de spikes** (`Encoder::encode(value∈[0,1], duración, dt)`):

| Encoder | Tipo | Nota |
|---|---|---|
| `PoissonEncoder` | Estocástico | Con refractario; **[!] RNG compartido** |
| `RateEncoder` | Estocástico | **[!] RNG compartido** |
| `TTFSEncoder` | Determinista | Time-to-first-spike, lineal o logarítmico |

**`RLEncoder`** (para observaciones de RL), con 6 tipos:

| Tipo | Qué hace | Salida | Origen |
|---|---|---|---|
| `SMALL` | 2 neuronas por variable [neg, pos], corriente = \|x\|·w | Vector de corrientes → `setInputCurrents` | Paper (acrobot/mountain) |
| `LARGE` | N neuronas por variable; solo la del intervalo activo recibe `current_val` | Corrientes | Paper (cartpole2/3) |
| `INTERVAL` | Como LARGE pero con trenes de spikes precalculados a `max_rate` | Spikes | — |
| `POPULATION` | Campos receptivos gaussianos (σ = 0.3) | Spikes | — |
| `RATE` | Tasa determinista | Spikes | — |
| `TEMPORAL` | Tiempo ∝ (1 - x) | Spikes | — |

**[!] Semántica de "corriente":** `setInputCurrents` hace `g_exc += I` en **cada** ms. Con τ_exc = 5 ms,
g_exc converge a I/(1 - e^{-0.2}) ≈ **5,5·I**. La corriente efectiva es unas 5,5 veces la nominal y
tarda algunos ms en subir. La alternativa exacta a "corriente constante" sería `setIOffset`, que
existe en la neurona y se usa en `current_injection_main.cpp`.

**Decoders** (`RLDecoder`):
- **Discretos:** `SPIKE_COUNT`, `FIRST_SPIKE`, `WINNER_TAKES_ALL` (tasa), `VOTING` (sub-ventanas) y `GROUP_SPIKE_COUNT` (grupos de N neuronas por acción; equivale al decoder de cartpole3 del paper).
- **Continuos:** `SPIKE_COUNT`, `FIRST_SPIKE`, `RATE` (tasa/100 Hz, recortada a 1).

**[!] Empates y silencio → acción 0.** El paper elegía una acción aleatoria cuando no había spikes.
Aquí hay un sesgo determinista hacia la acción 0.

## 7. RL con RLtools

- **RLtools** es header-only. Las demos usan sus entornos **Pendulum** y **Acrobot**, no gymnasium.
- **Demos** (`pendulum_eval`, `acrobot_eval`): una red fija (entradas → 2 o 4 ocultas → 1 salida, todo RS), un barrido del peso compartido {5, 10, 20} y **un episodio**.
  - Observación → corriente en [0, 20].
  - 50 ms de simulación por paso.
  - Decoder RATE → acción continua en [-1, 1].
- **Acrobot de RLtools ≠ Acrobot-v1 de gymnasium** (verificado en el código de RLtools, commit b32d998):

| | RLtools Acrobot | gymnasium Acrobot-v1 (paper) |
|---|---|---|
| Acción | **Continua** [-1, 1] → torque en **[-5, +5]** | Discreta {-1, 0, +1} |
| Recompensa | -1 por paso | -1 por paso |
| Término | Punta sobre la altura 1,0 | Igual |
| Límite | 500 pasos, DT = 0,2 | 500 pasos |
| Estado inicial | U(-0,1, 0,1) en ángulos y velocidades | Igual |

  **[→T]** Con torque continuo de ±5, el Acrobot de RLtools es probablemente **más fácil** que el del paper. Los fitness **no son comparables** con el paper.
- **[→T] El PPO de RLtools solo soporta acciones continuas** (política gaussiana con `log_std`; no hay rama categórica). Para el meta-controlador, la acción del MDP debería ser **continua**, por ejemplo un vector de probabilidades o intensidades por operador. Ver `comprension.md` §4.

## 8. Estado verificado en mullo (2026-09-26)

- **Dependencias:** cmake 3.28, g++ 13.3, CLI11, Catch2 3 y OpenMP ya están en el sistema. Falta **RLtools** (el submódulo vacío). Sin él no compilan `snn_pendulum` ni `snn_acrobot`.
- **Compilación** (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`; targets `snn_tests`, `snn_simulator`, `snn_current_injection`): **sin warnings** con `-Wall -Wextra`.
- **Tests:** **19/20 pasan.** Falla `FS: sin adaptación significativa de frecuencia` (razón de ISI 2,86 > 1,5). Probablemente es el transitorio inicial por arrancar en v = c; hay que revisarlo.
- **Determinismo:** la simulación secuencial y la paralela dan conteos de spikes idénticos (con `noise = 0`).
- **Velocidad:**
  - Una red de 5 neuronas y 6 sinapsis simula **25 000 ms (= un episodio Acrobot de 500 pasos × 50 ms) en ~4,2 ms** con 1 hilo, unos 0,17 µs por ms simulado.
  - 64 redes × 1000 ms: speedup de 6,8× con 16 hilos. La carga es pequeña; con redes grandes o muchos episodios debería escalar mejor.
  - Comparado con ANNarchy, que compilaba C++ por genoma, es un cambio de órdenes de magnitud.

## 9. Bugs y riesgos encontrados [!]

1. **`Simulator::runEpisode`:** el callback recibe el estado como `const std::vector<double>&`, así que **no puede actualizarlo**. Cada paso del episodio vuelve a codificar el estado **inicial**. Ninguna demo lo usa, pero es la API "de RL" y no sirve tal cual.
2. **Condición de carrera en `simulateAllEncoded`:** los hilos OpenMP comparten el `encoder`, y `PoissonEncoder`/`RateEncoder` usan un `mutable std::mt19937` interno. Es comportamiento indefinido y quita reproducibilidad. Con encoders deterministas (TTFS, RLEncoder) no hay problema.
3. **Reproducibilidad:** los RNG (ruido de la neurona y encoders) se siembran con `std::random_device`. El ruido de la neurona es `thread_local` y no tiene API para sembrarlo. **[→T]** La propuesta exige **mismas semillas** en todas las configuraciones, así que hay que agregar el sembrado.
4. **`decoder.hpp` y `rlDecoder.hpp` definen la misma clase `RLDecoder`** con el **mismo include guard** y contenido distinto (`rlDecoder.hpp` tiene una sobrecarga extra). CMake solo compila `decoder.cpp`. Si una unidad de compilación incluye `rlDecoder.hpp` y usa esa sobrecarga, falla al enlazar; si mezcla ambos headers, es una violación de ODR.
5. **Submódulo `external/rl-tools` sin `.gitmodules`:** un `git clone --recursive` no lo trae. Hay que obtener RLtools en el commit `b32d9985…` e indicarlo con `-DRL_TOOLS_INCLUDE_DIR=...`.
6. **Binarios versionados:** `build_cmake/snn_tests` (y otros artefactos de build) están commiteados.
7. **Semántica de entrada como conductancia acumulada** (§6) y **sesgo a la acción 0** (§6): no son bugs, pero cambian la dinámica respecto del paper.
8. **`Simulator sim;` no compila (constructor ambiguo):** `Simulator(const SimulationConfig& = {})` y `Simulator(double dt = 1.0)` son ambos constructores por defecto. Hay que usar `Simulator sim{SimulationConfig{}}` o quitar un default. Lo detectó el test de humo del repo.
9. **Cobertura de tests:** no hay tests de encoders, decoders, `Simulator`, mutaciones ni eliminación con swap-and-pop, que es la parte más delicada.
10. **Estado inicial `v = c`, `u = b·c`:** TONIC_BURSTING (IZ-C) y CHATTERING arrancan **en el equilibrio inestable** (−50 mV) y disparan con una perturbación mínima; RESONATOR y REBOUND_BURST disparan espontáneamente; BISTABILITY, ACCOMMODATION e INHIBITION_INDUCED_* disparan sin entrada. Como la red se reinicia en cada paso del entorno, se repite en cada ventana. Detalle y propuesta en `diseno_neat_ap.md` §3.

> **Actualización 2026-09-26 (M0):** resueltos los puntos 1 (runEpisode), 3 (semillas), 4 (decoder duplicado),
> 5 (submódulo eliminado), 6 (binarios), 8 (constructor ambiguo) y 10 (estado inicial), más pesos por sinapsis y
> entrada como corriente constante. Queda el 2 (encoders estocásticos compartidos entre hilos, documentado en
> `simulator.hpp`) y la cobertura de tests de encoders y decoders. Detalle en `diseno_neat_ap.md` §14.

## 10. Qué falta para usarlo en la tesis (brechas)

| Necesidad de la tesis | ¿Lo tiene? | Qué hacer |
|---|---|---|
| Pesos **por sinapsis** (NEAT evoluciona pesos; la propuesta incluye operadores **paramétricos** de "modificación de pesos") | ❌ Solo peso compartido | Agregar `weight` a `Synapse` y usarlo en `step()`. El peso compartido pasa a ser un caso particular (escala global) |
| Genoma NEAT (innovaciones, cruce, especiación, compatibilidad) | ❌ | Implementarlo: portar el NEAT C++ del paper (sin Python embebido) o escribirlo limpio. El genoma genera la `Network` |
| Operadores de mutación | ⚠️ Parcial: insertar neurona/sinapsis, deshabilitar, polaridad y tipo | Definirlos en el genoma (fuente de verdad) y construir el fenotipo |
| Evaluación en paralelo de la población | ✅ OpenMP | Paralelizar por genoma (episodios completos por hilo) |
| Entornos de control | ⚠️ Pendulum y Acrobot de RLtools (continuos) | Decidir los entornos (Fase 3). ¿Portar CartPole, Acrobot-v1 y MountainCar discretos? ¿Usar los de RLtools? ¿Algo más difícil? |
| Bucle de episodio RL | ❌ `runEpisode` roto | Escribir un evaluador `episodio(genoma, entorno, semilla) → retorno` |
| Semillas y reproducibilidad | ❌ | Sembrado explícito por corrida y por evaluación, sin `random_device` |
| Registro por generación | ❌ | Lo aporta el lado NEAT o AOS |
| Agente PPO para AOS | ⚠️ RLtools disponible pero sin integrar | Integrar RLtools PPO. Acción **continua** (§7) |

## 11. Cómo lo usaríamos (propuesta inicial, por discutir)

```
Genoma NEAT (nuestro)  ──build──▶  Network (snn-simulator)  ──episodios──▶  fitness
     ▲   nodos: tipo Izhikevich        con pesos por sinapsis        Entorno C++ + RLEncoder SMALL/LARGE
     │   conexiones: innov, peso,                                     + RLDecoder
     │   exc/inh, enabled
     │
  operadores de mutación  ◀── probabilidades o intensidades ◀── Agente PPO (RLtools) ◀── estado de la población
```

- **snn-simulator como biblioteca** (`snn_core`, `snn_encoding`, `snn_engine`), no como aplicación.
- Opciones de integración:
  - **Resuelto (2026-09-26):** el autor autorizó incorporarlo como código propio. Vive en `snn/` y se modifica directamente, sin fork.
- **Licencia GPL v3:** el código de la tesis que lo incluya queda bajo GPL v3.

## 12. Preguntas abiertas

1. ~~Fork o coordinación~~ → incorporado como código propio en `snn/` (autorización del autor).
2. ~~Pesos por sinapsis o compartido~~ → **pesos por sinapsis**: se evolucionan pesos y topología.
3. ¿Entradas como corriente constante (`i_offset`, igual que el paper) o como conductancia acumulada (actual)?
4. ~~Entornos~~ → **entornos de RLtools** (coherente con la propuesta, §5.2).
