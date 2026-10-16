# Comprensión de la tesis — notas de trabajo

> Documento de referencia rápida, escrito por Claude el 2026-09-26 a partir de
> [`propuesta.md`](propuesta.md) (fuente de verdad) y de una revisión del código del paper previo
> (`~/archivo_paper_2026/Neuronas_Izhikevich`).
> Convención: **[P]** = lo dice la propuesta · **[C]** = observado en el código previo ·
> **[?]** = abierto, por decidir · **[!]** = riesgo o advertencia.
> Actualizar este archivo cuando se tomen decisiones (sección 9).

---

## 1. La tesis en 30 segundos

Se entrenan **SNN** (redes neuronales de espigas) para **tareas de control** con **NEAT**. Hoy NEAT aplica
sus mutaciones (cambiar un peso, agregar neurona, agregar conexión…) con **probabilidades fijas**, sin
mirar cómo va la evolución. La tesis reemplaza eso por un **agente de RL (PPO, biblioteca RLtools, C++)**
que en **cada generación** observa métricas de la población (**estado**), decide cómo aplicar los
operadores de mutación (**acción**) y recibe una **recompensa** según cómo mejoró la población. Esto se
formaliza como un **MDP**.

Se compara contra **2 líneas base** con el mismo catálogo de operadores:
1. **Estática**: probabilidades fijas, sintonizadas.
2. **Adaptive Pursuit** (Thierens 2005): AOS **sin estado**, el mecanismo de SNAP-NEAT.

La comparación responde dos preguntas. Contra la estática: ¿sirve adaptar? Contra Adaptive Pursuit:
¿sirve usar el *estado*?

**Pregunta de investigación [P]:** ¿Cómo incorporar AOS *basado en estado* dentro de NEAT para
**(a) aumentar el fitness final de la mejor red** y **(b) reducir las generaciones hasta converger**
al entrenar SNN?

**Aporte / nicho [P]:** intersección de la taxonomía AOS de Pei et al. (2025) (familia *basada en
estado*) con la de hibridación EA+RL de P. Li et al. (2024) (RL asiste al EA → selección dinámica de
operadores), aplicada a **neuroevolución topológica de SNN**. Ni SNAP-NEAT ni AGENT formalizan un MDP
ni aprenden una política.

---

## 2. Arquitectura [P]

Tres módulos, todo el núcleo en **C++**:

| Módulo | Qué hace | Nota |
|---|---|---|
| **Motor NEAT** | Población, especiación, catálogo de operadores paramétricos (pesos) y topológicos (nodos, sinapsis) | Base: NEAT C++ del equipo [C] |
| **Simulador SNN** | Simula la dinámica de espigas y asigna el fitness de cada genoma | **Nuevo, en C++** (el paper usaba ANNarchy/Python) |
| **Meta-controlador** | Agente PPO (RLtools) que sustituye las probabilidades estáticas por una política π(a\|s) | Decide **entre generaciones**, no controla la tarea |

Ciclo por generación (mi lectura de la propuesta; el orden exacto de los pasos está por decidir):

```
          ┌──────────────────────────────────────────────────────────────┐
          │ generación t                                                  │
          │  1. Simulador SNN evalúa a todos los genomas → fitness         │
          │  2. Extraer estado s_t (métricas de la población)              │
          │  3. Recompensa r_t por la acción a_{t-1} (mejora vs t-1)       │
          │  4. Agente PPO: a_t ~ π(·|s_t)   (guardar transición)          │
          │  5. NEAT: eliminar / reproducir / MUTAR usando a_t / especiar  │
          └───────────────────────────────┬──────────────────────────────┘
                                          ▼  t+1
  Un episodio del MDP = una corrida de NEAT (≈ N generaciones) → cada corrida entrega pocas transiciones.
```

---

## 3. Qué está FIJADO vs qué está ABIERTO

| Tema | Fijado por la propuesta [P] | Abierto [?] |
|---|---|---|
| Algoritmo RL | PPO de RLtools, en C++. **Su PPO solo soporta acciones continuas (gaussianas)**, verificado 2026-09-26 | Arquitectura de la red de la política, hiperparámetros (se sintonizan en Fase 4) |
| Granularidad | Decisión **entre generaciones** | ¿Una acción global por generación, por especie o por genoma? |
| Estado | "Métricas poblacionales" (glosario: diversidad genética, nivel de estancamiento) | Lista concreta de features y su normalización |
| Acciones | Operadores de mutación **topológicos y paramétricos** de NEAT | **Continua**, por la restricción de RLtools: vector de probabilidades o intensidades por operador. ¿Se agregan operadores (p. ej., eliminar nodo o conexión como AGENT)? |
| Recompensa | "Incentivar convergencia, penalizar desaprendizaje" (Gantt 1.5) | Fórmula concreta y ventana de crédito |
| Aprendizaje | "La política se ajusta **durante la ejecución**" (§2.3) | Online vs preentrenada: **se decide sobre la marcha, eligiendo lo que maximice resultados** (usuario, 2026-09-26) |
| Líneas base | Estática (desde valores de Stanley 2002, sintonizada en Fase 4) + Adaptive Pursuit | Parámetros de AP (p_min, α, β) |
| Entornos | Control simulado, complejidad **incremental**, dinámica no lineal, horizonte largo | **Cuáles**: se eligen con un barrido de NEAT estático sobre **todos** los entornos de RLtools (22 registrados; `entornos_rltools.md`). En curso desde 2026-10-03 |
| Codificación | Se definen "esquemas de codificación de espigas" (Fase 3) | ¿Se reusa la del paper? (ver §6) |
| Simulador SNN | **snn-simulator**, incorporado en `snn/` (C++, de *dilgeto*), ver [`snn-simulator.md`](snn-simulator.md) | Agregar pesos por sinapsis, sembrado, fixes |
| Modelo de neurona | No especificado | Izhikevich (el simulador trae 25 tipos, incluidos los IZ-A..H del paper). ¿Cuál o cuáles? |
| Réplicas | **11** corridas por configuración y entorno, **mismas semillas** y **mismo presupuesto de evaluaciones de fitness** | Tamaño de población y presupuesto concreto |

---

## 4. Diseño del MDP (Fase 1): candidatos para discutir

> **2026-10-13:** propuesta detallada del agente PPO (MDP, integración con RLtools, entrenamiento y verificación) en
> [`diseno_ppo.md`](diseno_ppo.md), **pendiente de revisión**. Lo que sigue son las ideas iniciales.

Esto no está en la propuesta. Son ideas iniciales para la Fase 1 y hay que validarlas con la literatura
(Pei 2025; P. Li 2024; Karafotias 2014).

**Estado s_t** (normalizado y acotado):
- Fitness: mejor, media y desviación estándar, normalizados por la escala del entorno. Mejora relativa del mejor en una ventana de k generaciones.
- Estancamiento: generaciones desde la última mejora del mejor.
- Diversidad: número de especies, distancia de compatibilidad media, entropía de tamaños de especie.
- Complejidad: nodos y conexiones medias y del mejor.
- Progreso: fracción del presupuesto consumido (t / T).
- Historial de operadores: tasa de éxito reciente de cada operador (le da al agente la información que usa AP).

**Acción a_t:**
- *Opción A, continua:* vector de probabilidades por operador (salida gaussiana → sigmoid/softmax). Encaja con PPO para control continuo y con el código actual, que ya usa probabilidades por operador (§6.2).
- *Opción B, discreta:* elegir qué operador (o combinación) aplicar esta generación. **Descartada con RLtools:** su PPO solo implementa política gaussiana (verificado en el commit b32d998). Usarla exigiría implementar PPO categórico. → **Preferir la Opción A.**

**Recompensa r_t:**
- Base: Δ(mejor fitness) normalizado entre t-1 y t.
- Penalización por "desaprendizaje": caída del mejor o de la media.
- **[!] Crédito retardado:** SNAP-NEAT advierte que una mutación estructural rinde recién después de que sus pesos se ajustan. Evaluar el operador en el paso inmediato subestima los topológicos. Considerar ventanas o recompensa acumulada (γ alto) o crédito diferido.

**[!] Eficiencia de muestra:** si el episodio es una corrida de ~50–200 generaciones, PPO recibe muy pocas
transiciones por corrida. Opciones:
- (a) paralelizar muchas corridas como entornos paralelos, lo que encaja con PPO y con los 288 núcleos de mullo;
- (b) preentrenar la política en corridas previas;
- (c) decisiones más finas (por especie) para tener más muestras.

Si se preentrena, **el costo de entrenamiento debe reportarse** para que la comparación con las líneas
base sea justa (mismo presupuesto de evaluaciones).

---

## 5. Protocolo de evaluación [P] (§3.4): checklist

- [ ] 3 configuraciones: **AOS-RL** vs **Estática** vs **Adaptive Pursuit**, con el mismo catálogo de operadores.
- [ ] Por entorno: **11** corridas independientes, **mismas semillas**, **mismo presupuesto de evaluaciones de fitness**.
- [ ] Analizar **por entorno**, nunca agregar entre entornos.
- [ ] **Eficacia**: fitness final de la mejor red → **Mann-Whitney U**.
- [ ] **Complejidad**: número de neuronas y conexiones de la mejor red → **Mann-Whitney U**.
- [ ] **Estabilidad**: dispersión del fitness entre réplicas → **Brown-Forsythe**.
- [ ] **Eficiencia**: generaciones hasta un umbral → **log-rank** sobre curvas de supervivencia; las corridas que no llegan quedan censuradas, no se descartan.
  - Umbral: el criterio de éxito del entorno si existe; si no, **95 % del mejor fitness de las corridas preliminares de la Fase 4**. **Se fija ANTES de las réplicas finales.**
- [ ] Comparaciones por pares (AOS-RL vs cada línea base), **α = 0,05**, corrección de **Holm** por entorno.
- [ ] Tamaño de efecto **Â12 de Vargha-Delaney** (P(AOS-RL supera a la línea base)).
- [ ] Con n = 11 solo se detectan **efectos grandes**; la propuesta lo acepta explícitamente.
- Herramientas: Python con NumPy, Pandas, SciPy y **lifelines** (supervivencia), más Matplotlib y Seaborn.
- **[!] Implicación de registro:** cada corrida debe guardar por generación el fitness (mejor, media, desv. estándar), la complejidad, las acciones del agente y el uso de operadores. Sin la curva por generación no se puede calcular "generaciones hasta el umbral".

---

## 6. Relación con el trabajo previo (paper Izhikevich)

- Repo formal publicado: https://github.com/josefoitzick/neuroevolution-izhikevich
- Repo de experimentación en mullo: `~/archivo_paper_2026/Neuronas_Izhikevich` (índice en `~/archivo_paper_2026/README.md`).

### 6.1 Qué hacía el paper [C]
- NEAT en C++ (`NEAT/src`, unas 2,4k líneas) que **embebe Python** y llama a `annarchy.py` para evaluar cada genoma con ANNarchy.
- 8 variantes de neurona Izhikevich (IZ-A..H: tonic spiking, phasic bursting, etc.) sobre CartPole, Acrobot y MountainCar (gymnasium) más clasificación (iris, wine, bc).
- **Codificación**: cada dimensión de la observación → **2 neuronas de entrada** (parte negativa y positiva), inyectando corriente `I = |x|·w`.
- **Decodificación**: se simulan **50 ms** por paso del entorno; la acción es el **argmax del conteo de espigas** de las neuronas de salida (acción aleatoria si no hay espigas).
- Fitness = retorno medio en **100 episodios**. Población de 50 genomas y 50 generaciones.
- Paralelismo: `fork()` + pipes, con `process_max` procesos por generación.

### 6.2 El punto donde se engancha la tesis [C]
`Genome::mutation()` (`NEAT/src/genome.cpp:260`) aplica **4 operadores como Bernoulli independientes**:

| Operador | Parámetro de probabilidad | Qué hace |
|---|---|---|
| Mutar peso | `probabilityWeightMutated` | ±`learningRate` (paso **fijo**) sobre **una** conexión habilitada |
| Agregar nodo | `probabilityAddNode{Small,Large}` | Divide una conexión habilitada; tope `n_max` nodos |
| Agregar conexión | `probabilityAddLink{Small,Large}` | Par aleatorio; si ya existe deshabilitada, la re-habilita; si no, peso en `weightsRange` con signo aleatorio |
| Mutar peso de entrada | `probabilityInputWeightMutated` | Perturba `inputWeights` |

- Las probabilidades *Large* se usan cuando la red tiene **menos** de `largeSize` nodos. El nombre es contraintuitivo; hay que confirmarlo.
- **Contadores de uso por generación ya existen** (`parameters.mutacionPeso`, `agregarNodos`, …, escritos en `operadores.txt`). Sirven para la asignación de crédito y para las features de estado.
- En `Population::evolution()` el orden por generación es: `evaluate → eliminate → mutations → reproduce → speciation`.
  - `mutations()` muta **in situ** a los sobrevivientes de cada especie, salvo al mejor.
  - Los hijos de cruce **no** se mutan.
  - Solo los hijos de especies con un único genoma se generan por copia + mutación.

  Esto difiere del NEAT canónico, donde se mutan los hijos, y conviene decidir si se mantiene.

### 6.3 Qué cambia en la tesis
- **Simulador SNN en C++ (snn-simulator)** en vez de ANNarchy. Replica los encoders del paper; ver `snn-simulator.md`. Esto elimina el Python embebido, la compilación por genoma y la caché `annarchy/`, que en el paper llegó a **2,3 TB**.
- **[!] Consecuencia:** los entornos de gymnasium (Python) ya no están disponibles directamente. Hay que reimplementarlos en C++ (CartPole, Acrobot y MountainCar son pocas decenas de líneas cada uno) o usar los que traiga RLtools (**verificar cuáles**).
- **[!] CartPole** probablemente cae fuera del alcance ("los problemas elementales convergen en muy pocas generaciones", §3.3). Sirve como smoke test.

### 6.4 Hallazgos en el código previo (revisar antes de reutilizarlo) [C][!]
1. **Genomas sin evaluar cuando `numberGenomes % process_max ≠ 0`.**
   - Causa: `Population::evaluate()` (`population.cpp:279`) reparte con `genomes_per_process = nGenomes / max_processes` (división entera). El resto de los genomas **nunca se evalúa** y queda con **fitness 0**.
   - Por qué importa: en Acrobot y MountainCar el retorno siempre es negativo, así que esos genomas sin evaluar pasan a ser **"los mejores"**.
   - Evidencia (2026-09-26), contando las generaciones cuyo "Best Genome" tiene fitness `0`:
     - `corridas/mountain_par` (process_max=32, 50 genomas): **1 544 / 4 312 (36 %)**.
     - `corridas/reopt_rl` acrobot: **10 084 / 19 600 (51 %)**. Mountain: **7 628 / 19 600 (39 %)**. Ambos con process_max=32.
     - Campañas con process_max=50 o 25 (acrobot_par, *_oldp_par, cartpole_par): **0**.
   - **Fix:** usar división hacia arriba y asignar el resto de los genomas.
2. **`Population::reproduce()` (rama de hijos por mutación):** toma `index` dentro del rango de la especie pero lee `genomes[index]` (la población global) en vez de `species[i]->genomes[index]`.
3. **Fitness ruidoso:** en acrobot y mountaincar los pesos de entrada se **re-muestrean al azar en cada evaluación** (`np.random.uniform(min,max)`), y si no hay espigas la acción es aleatoria. El mismo genoma da fitness distintos entre evaluaciones. **[!] Para AOS esto es ruido directo en la recompensa del agente.**

---

## 7. Plan de trabajo [P]: 17 semanas, 620 h

| Fase | Horas | Semanas | Entregable clave |
|---|---|---|---|
| 1. Formalización MDP | 60 | S1–S5 | Estado, acción y recompensa definidos (1.2–1.5) |
| 2. Desarrollo e integración | 190 | S1–S12 | Entorno C++ + RLtools (S1–2), NEAT adaptado (S2–8), SNN C++ (S3–9), PPO (S5–10), flujo NEAT↔agente verificado (S10–12) |
| 3. Entornos y codificación | 60 | S4–S11 | Entornos (S4–6), codificación (S5–8), validación (S7–9), tests de fitness (S9–11) |
| 4. Experimentación y tuning | 150 | S10–S17 | Tuning NEAT (S10–13), cuellos de botella (S10–13), preliminares agente (S12–14), tuning PPO (S13–15), **umbral por entorno** (S14–17) |
| 5. Evaluación comparativa | 80 | S7–S17 | Scripts estadísticos (S7–10), líneas base (S11–14), réplicas AOS (S15–16), análisis (S16–17) |
| 6. Redacción | 80 | S1–S17 | Capítulos (S1–3), redacción continua (S4–16), revisión final (S16–17) |

**Ruta crítica:** 2.1 → 2.2/2.4 → 3.3 → 3.4 → 2.5 → 4.1 → 4.3 → 5.1 → 5.5. El agente se prueba recién
en la **S12**, lo que deja poco margen si PPO no aprende. Conviene tener un **prototipo mínimo del lazo
completo lo antes posible**, con un simulador simple, un entorno simple y un agente simple.

**S1 = semana del 2026-09-21** → S17 ≈ semana del 2027-01-11. La Gantt es **referencia**, no un calendario rígido.

---

## 8. Riesgos principales [!]

1. **PPO con pocas muestras** (§4): el agente no aprende en el horizonte de una corrida. Mitigación: corridas en paralelo como entornos vectorizados, preentrenamiento o granularidad más fina.
2. **Recompensa ruidosa** (§6.4-3) y **crédito retardado** de las mutaciones estructurales.
3. **Costo de simulación SNN** × 11 réplicas × 3 configuraciones × entornos × tuning. Hay que medir temprano el tiempo por evaluación en C++.
4. **Justicia de la comparación:** mismo presupuesto de evaluaciones y mismas semillas; el costo de entrenar al agente debe declararse.
5. **Entornos demasiado fáciles**: si convergen rápido, no hay horizonte para que el agente aprenda (§3.3).
6. **RLtools**: curva de aprendizaje (header-only, C++ con plantillas pesadas) y soporte de acciones discretas por verificar.
7. **Reutilizar el código del paper sin corregir los bugs de §6.4**.

---

## 9. Decisiones tomadas (bitácora)

> Registrar aquí cada decisión con fecha y motivo. Formato: `AAAA-MM-DD — decisión — por qué`.

- 2026-09-26 — Estructura: `~/tesis/` para todo lo de la tesis; `~/archivo_paper_2026/` para el paper previo (solo referencia).
- 2026-09-26 — **S1 del plan = 2026-09-21**. La Carta Gantt es **referencia, no un calendario rígido**.
- 2026-09-26 — La tesis tendrá un **repo nuevo y vacío** (lo crea el usuario). No se parte de los repos del paper.
- 2026-09-26 — Simulador SNN: **snn-simulator** (C++, de *dilgeto*, GPL v3). Análisis en `snn-simulator.md`.
- 2026-09-26 — Online vs preentrenamiento del agente: se decide sobre la marcha, eligiendo lo que maximice resultados.
- 2026-09-26 — Acción del MDP: **continua** (el PPO de RLtools solo soporta acciones gaussianas).
- 2026-09-26 — **Pesos por sinapsis**: se evolucionan pesos y topología. El peso compartido era propio de la tesis del compañero (WANN).
- 2026-09-26 — snn-simulator se incorpora como **código propio** en `snn/`, con autorización de *dilgeto*; se eliminó su `.git`. Sin fork.
- 2026-09-26 — Entornos de control: **los de RLtools**.
- 2026-09-26 — Líneas base primero: NEAT estándar (estático) y NEAT + Adaptive Pursuit; el AOS con estado viene después.
- 2026-09-26 — Esquema **categórico** (un operador por hijo ~ P), común a las 3 configuraciones.
- 2026-09-26 — Catálogo inicial = **NEAT clásico** (mutar pesos, agregar conexión, agregar nodo), verificado en Stanley 2002 §3.1 y §4.1. P estático = (0,909 ; 0,057 ; 0,034).
- 2026-09-26 — Neurona base: **IZ-A (TONIC_SPIKING)**. Reposo estable al iniciar; entrada como corriente constante.
- 2026-09-26 — NEAT se escribe **desde cero** (no se porta el del paper). Nada de WANN.
- 2026-09-26 — RLtools clonado por Claude en `external/rl-tools` (b32d998); el usuario lo registra como submódulo al hacer su commit.
- 2026-09-26 — Acrobot de RLtools con torque ±5 es trivial (la población inicial lo resuelve); usar `torque_scale` 0,2 (±1, como gymnasium).
- 2026-09-26 — Organización: **un solo repo** (`rl_aos_neat_snn`, público) modular por bibliotecas CMake; RLtools como submódulo en `external/rl-tools`; resultados en `~/tesis/runs/` (fuera de git); PDF firmado en `~/tesis/privado/` (fuera de git).
- 2026-10-03 — Reunión 2 con el profesor guía (`diseno_neat_ap.md` §15):
  - la **mutación de pesos es parte del catálogo** del AOS;
  - el **crédito se explora sistemáticamente**, guiado por la taxonomía de Pei et al. (2025);
  - la **elección de entornos es la tarea prioritaria** y será una sección de los primeros capítulos;
  - el umbral normalizado y los desvíos del NEAT original no se abordaron.
- 2026-10-03 — Fase inicial con los **3 operadores de mutación de Stanley**. El cruce sigue fijo (tasas del paper); la especiación es mecanismo de la población y puede ir al estado del agente, no a la acción. El usuario explora operadores nuevos la semana del 2026-10-05 (`diseno_neat_ap.md` §15.3).
- 2026-10-03 — Metodología de selección de entornos: **solo con NEAT estático** (sin AOS, para no sesgar la elección hacia un método comparado). No se descarta ningún entorno sin probarlo: solo se excluye por un motivo medido, y queda reportado.
- 2026-10-03 — Etapa A del barrido:
  - **5 semillas × 500 generaciones**, con especiación dinámica (10 especies objetivo) y reasignación de pesos 0,02;
  - entornos de memoria con `--keep-state` y, como control, con política reactiva.
- 2026-10-03 — RLtools fijado en la rama **`render` @ `6bb2fda`**, que trae reacher y hyperdrone. Su RNG cambió respecto de `main`, así que las corridas anteriores quedan como exploratorias.
- 2026-10-03 — Opción `--keep-state` en el evaluador: la SNN se reinicia solo al comienzo del episodio, lo que da una política con memoria. Por defecto se mantiene la política reactiva (sin cambios, verificado con un test de regresión).
- 2026-10-03 — Entornos visuales:
  - cámara **16×16** (64×64 es inviable para NEAT con topología inicial completa);
  - escenas: entrenamiento en las **casas 0 y 1** (las de RLtools), test en la **casa 2** (no vista); el dataset público solo trae 3 casas;
  - hyperdrone usa el MDP del entrenador oficial;
  - moving-gate usa un dron público como modelo de la puerta (el original no es público);
  - visual_inertial_localization se excluye: no es una tarea de control.
- 2026-10-03 — Recursos de mullo: el barrido usa como máximo **160 hilos**, para dejar ~128 núcleos a otro usuario . Barridos con candado, marca atómica por corrida y binario congelado. A y B quedan certificados como un solo experimento por la prueba de equivalencia de binarios.

- 2026-10-04 — Barridos A y B terminados (165 corridas). Filtro: los visuales (salvo moving-gate reactivo) y flag no superan el retorno trivial (`entornos_rltools.md` §7).
- 2026-10-04 — **Criterio de selección de entornos:** score = (NEAT − trivial) / (RL − trivial), con RL de RLtools evaluado en nuestro entorno con los mismos episodios finales que NEAT (`diseno_neat_ap.md` §15.4; `entornos_rltools.md` §8).
  - ≥ 0,9: ancla;
  - 0,2–0,8: "NEAT lucha", objeto de estudio del AOS;
  - < 0,2: fuera de alcance.

  El análisis de estado (sondas de P) va después, solo sobre los elegidos.
- 2026-10-04 — Referencias RL v2: OpenBLAS y 10 semillas. moving-gate queda sin referencia: RLtools solo entrena hyperdrone en GPU.
- 2026-10-04 — `neat_control --resample-train`: los episodios de entrenamiento rotan en cada generación, comunes a la población, y el campeón final se elige por validación. Se mide junto al protocolo fijo antes de decidir (pregunta abierta 9). Sin la opción, el binario es idéntico al anterior.
- 2026-10-04 — Barrido largo: 2000 generaciones (ant 1000), protocolos fijo y rotación. Desde las 17:30 el tope es de **256 hilos** (entra otro usuario), así que primero corren las semillas 1–5; las 6–10 quedan encadenadas para después.
- 2026-10-11 — **El MDP se diseña transferible entre entornos**, sin salir del alcance de la propuesta (agente por entorno contra estático y AP). Requisitos:
  - acción: P sobre el catálogo común;
  - estado: sin unidades del entorno (fitness normalizado con el score trivial→RL o por rangos, diversidad, estancamiento, complejidad relativa, fracción del presupuesto y **tasa de éxito reciente de cada operador**, para que la política descubra en la corrida las preferencias de cada entorno);
  - recompensa: mejora en score normalizado.

  Extensión posible, si el agente por entorno funciona: entrenar en acrobot-swingup + l2f (baratos) y probar sin reentrenar en bottleneck (caro).
- 2026-10-11 — **Líneas base confirmadas: NEAT estático, AP (sin estado) y PPO (con estado).** AP encaja con el esquema "un operador por hijo según P" (los bandits clásicos eligen en secuencia y por argmax, y chocan con la evaluación por lotes de NEAT) y es el mecanismo de SNAP-NEAT. Un bandit por lotes (Thompson o softmax sobre UCB) queda como segunda línea base sin estado **solo si sobra tiempo**: la prioridad es el catálogo y la mejora con estado.

- 2026-10-14 — **Plan revisado tras la etapa 1 de la sonda** (efectos de los operadores casi todos dentro del ruido):
  - **Dos escenarios de comparación:** E1 "sin ajuste previo" (todos parten de P uniforme sobre el catálogo; motivación de la propuesta, §2.3) y E2 "con ajuste previo" (P₀ de Optuna, como dice §3.4). E2 se reporta aunque dé nulo.
  - **Proponer al profesor la comparación pareada por semilla** (Wilcoxon de rangos con signo; números aleatorios comunes), en vez de Mann-Whitney. La correlación intra-semilla medida es 0,72 (acrobot) y 0,79 (l2f); detecta efectos ~2× más chicos. `analysis/compare.py --pareado`.
  - **Catálogo K = 10:** Stanley + input_gain, weights_fine, neuron_dynamics, neuron_bias, delete_simple_node + neuron_family y delete_connection (efecto dependiente del entorno: crean margen para el AOS).
  - **l2f se reemplaza por pendulum-meta reactivo** (sin `--keep-state`), por la regla de respaldo (score < 0,2). Entornos: bottleneck, acrobot-swingup y pendulum-meta.
  - **Prueba de margen** antes de entrenar PPO (`~/tesis/runs/margen_aos/`): uniforme, AP desde uniforme y P en dos fases; reglas fijadas antes de mirar.
  - **Evaluación final de PPO con `aos_ppo_train --evaluar`** (idéntico bit a bit a `neat_control`), en lugar de integrarlo a `neat_control`. Entrenar y evaluar con el mismo binario congelado: entre binarios distintos, el redondeo de la red puede cambiar trayectorias.
- 2026-10-14 — **Documentación para la escritura:** `docs/README.md` (índice y mapa a capítulos), `docs/bitacora/` (una página por semana desde S01 = 2026-09-21) y `docs/experimentos/` (13 fichas). Se actualiza al cerrar cada sesión, con redacción neutra para poder subirla al repo público.

## 9b. Diseño de las líneas base
Ver [`diseno_neat_ap.md`](diseno_neat_ap.md): NEAT estático y Adaptive Pursuit con un esquema categórico común (un operador por hijo ~ P), catálogo O1–O6, `OperatorSelector`, CRN y conjunto de test, **umbral normalizado para fitness negativo** (el "95 %" de §3.4 falla con recompensas negativas).

## 10. Preguntas abiertas para el usuario o el profesor guía

1. ~~Fecha de la S1~~ → 2026-09-21.
2. ~~Base de código~~ → repo nuevo; simulador = snn-simulator.
3. ¿Qué tipo(s) de neurona Izhikevich? ¿Fijo o evolucionable por nodo (el simulador permite cambiar el tipo)?
4. Entornos → RLtools. Regla fijada el 2026-10-04: score contra una referencia RL (`diseno_neat_ap.md` §15.4).
   - Provisoriamente: anclas pendulum, car, car-track y acrobot; "NEAT lucha" en pendulum-multitask, memory, acrobot-swingup, pendulum-meta, pendulum-velocity, pendulum-position y bottleneck.
   - La selección final sale del barrido largo y se presenta en la reunión del 2026-10-06.
5. ~~Online vs preentrenado~~ → sobre la marcha.
6. ~~Discreta vs continua~~ → continua (restricción de RLtools).
7. ¿Se agregan operadores al catálogo (eliminar nodo o conexión, cambiar polaridad o tipo de neurona; snn-simulator ya los soporta)? → Por ahora no (§9, 2026-10-03). El usuario los explora la semana del 2026-10-05; también se le planteará al profesor.
8. ~~Pesos~~ → por sinapsis.
9. ~~Fork~~ → código propio en `snn/`.
10. ¿Protocolo de entrenamiento: episodios fijos (como el preprint) o rotación por generación (`--resample-train`)? Se decide con el barrido largo; se plantea en la reunión del 2026-10-06.
11. La memoria de la SNN (`--keep-state`) empeora el desempeño en varios péndulos de observación parcial, mientras que RL con GRU sí la aprovecha. ¿Se estudia, o se trabaja con la versión reactiva?
