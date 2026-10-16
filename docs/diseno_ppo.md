# Diseño del agente PPO (AOS basado en estado)

> Documento de trabajo (2026-10-13). Propuesta de diseño del MDP (Fase 1) y de la integración con RLtools (Fase 2.3–2.5),
> **pendiente de revisión**. Las decisiones abiertas están en §9.
> Relacionados: [`propuesta.md`](propuesta.md) (fuente de verdad), [`comprension.md`](comprension.md) §4 y §9,
> [`diseno_neat_ap.md`](diseno_neat_ap.md) (líneas base), [`operadores_candidatos.md`](operadores_candidatos.md) (catálogo).
> Convención: **[P]** propuesta · **[V]** verificado en el código o en nuestros datos · **[D]** decisión propuesta, a confirmar ·
> **[?]** abierto · **[!]** riesgo.

---

## 0. Resumen

| Elemento | Propuesta |
|---|---|
| Episodio | Una corrida completa de NEAT: semilla s, presupuesto B evaluaciones |
| Reloj de decisión | T = 50 decisiones por corrida, una cada 2 % del presupuesto (k ≈ 7–21 generaciones por decisión, según el entorno). Las métricas se extraen en **cada** generación |
| Acción | a ∈ ℝ^K gaussiana → **P = softmax(log P₀ + c·tanh(a/c))**, con P₀ = P estática ajustada y c = 3. Con a = 0 se obtiene exactamente la línea base estática |
| Estado | 14 rasgos globales (presupuesto, progreso, estancamiento, diversidad, complejidad) + 2 por operador (crédito por rango y tasa de fallo). Sin unidades del entorno |
| Recompensa | r_t = ΔΦ + (w/T)·Φ, con Φ = score del campeón vigente medido en episodios propios (stream 3). Retorno = mejora final + w · área bajo la curva |
| Entrenamiento | Fuera de línea y por entorno, en semillas 1001+. Según el paso 1 (§5.1), harían falta **≥ 400 corridas de NEAT, idealmente ~800** (el plan inicial era ~190). La política se congela y se evalúa en las semillas 1–11 |
| Integración | Entorno por lotes propio para RLtools (`NeatBatch`), con 16 corridas de NEAT en paralelo. El runner, GAE y `train` de RLtools se usan sin modificar |

**La meta:** con 11 réplicas, la prueba de Mann-Whitney detecta una mejora de ≈ 0,05–0,07 de score (§1.3). Es lo mismo que
ganó Optuna sobre NEAT por defecto. PPO tiene que sumar eso **encima** de NEAT ajustado.

---

## 1. Punto de partida

### 1.1 Lo que fija la propuesta [P]

- Agente **PPO de RLtools**, en C++ (§3.3).
- "Un agente autónomo extrae métricas de la población **en cada generación** (el estado), decide la aplicación de los
  operadores de variación (la acción) y evalúa el rendimiento resultante (la recompensa)" (§2.3, enfoque 3).
- Opera "a nivel de la selección de operadores **entre generaciones**", no controla la tarea (§3.2).
- Recompensa: "**incentivar convergencia, penalizar desaprendizaje**" (Gantt, tarea 1.5).
- Evaluación (§3.4):
  - 11 réplicas con el mismo presupuesto;
  - métricas: fitness final (Mann-Whitney), generaciones hasta un umbral (log-rank), complejidad y estabilidad (Brown-Forsythe);
  - corrección de Holm y tamaño de efecto Â12.

### 1.2 Decisiones previas que condicionan este diseño [V]

- **Esquema categórico**, común a estático, AP y PPO: un operador por hijo, según P (2026-09-26). En `population.cpp`
  **todo hijo no élite** recibe exactamente un operador, sea clon o hijo de cruce, así que P gobierna toda la variación.
- **Acción continua:** el PPO de RLtools solo trae política gaussiana.
- **Catálogo común** a los tres entornos: 15 operadores implementados, de los que se congelarán ~6–8 tras la sonda.
- **MDP transferible** (2026-10-11):
  - estado sin unidades del entorno;
  - acción = P sobre el catálogo común;
  - tasa de éxito por operador en el estado.
- **Base NEAT por entorno:** configuración ajustada con Optuna (#96, #49, #65), con rotación de episodios. El presupuesto
  común queda pendiente de las corridas de 300 mil evaluaciones.
- **Online o fuera de línea:** "se decide sobre la marcha, eligiendo lo que maximice resultados" (2026-09-26). La
  recomendación está en §9.1.

### 1.3 Lo que dicen nuestros datos [V]

Datos de NEAT ajustado: 75 mil evaluaciones, rotación, semillas 1–10 (`~/tesis/runs/eval_optuna/`).

| | bottleneck (#96) | acrobot-swingup (#49) | l2f (#65) |
|---|---|---|---|
| Población × generaciones | 70 × 1071 | 130 × 577 | 210 × 357 |
| Costo por corrida | ≈ 38 hilo-h | ≈ 10 hilo-h | ≈ 6 hilo-h |
| Test final crudo, media ± DE | 20,6 ± 2,8 | −11,6 ± 0,43 | 8,07 ± 1,14 |
| Score, media ± DE | 0,33 ± 0,045 | 0,57 ± 0,038 | 0,15 ± 0,043 |
| Mejora mínima detectable con 11 réplicas (crudo) | ≈ +3,6 a +4,2 | ≈ +0,6 | ≈ +1,5 a +1,7 |
| Cambios de campeón por corrida (validación) | 26 (20–37) | 23 (12–35) | 14 (7–30) |

Consecuencias:

1. **La meta:** el poder de Mann-Whitney se simuló con n = 11 por grupo y datos normales.
   - Con α = 0,05 alcanza 0,8 para una diferencia de 1,3 DE.
   - Con α = 0,025 (primer paso de Holm con dos comparaciones) necesita 1,5 DE.
   - Con la DE de cada entorno (≈ 0,04 de score), eso equivale a **≈ 0,05–0,07 de score**, del mismo orden que la ganancia de
     Optuna sobre NEAT por defecto (+0,06 a +0,09).
2. **Con los 3 operadores de Stanley no hay margen:**
   - en la sonda de P, P solo importó en bottleneck;
   - cambiar P a mitad de la corrida nunca superó al mejor P fijo.

   Si hay margen, lo abre el catálogo ampliado, y eso se verifica antes de entrenar (§6).
3. **El costo lo domina bottleneck**, y el presupuesto común de NEAT multiplica el costo de entrenar PPO (§4.2).

### 1.4 Lo que tomamos de la literatura

- **Pei et al. (2025):**
  - en AOS, PPO aparece solo con entrenamiento **fuera de línea** ([182], [197], [202]–[204]);
  - la guía §VI-C recomienda entrenar fuera de línea cuando hay recursos y problemas de entrenamiento representativos;
  - estado sugerido: rasgos del proceso (último operador, último crédito, estancamiento, presupuesto) y de la población.
- **Schuchardt, Golkov y Cremers (2019), *Learning to Evolve*:**
  - PPO entrenado fuera de línea sobre varias instancias, evaluado con la acción media determinista;
  - para episodios de duración fija, **V(s_T) = 0** y la **fracción de tiempo restante** en el estado, siguiendo a Pardo
    et al. (2018);
  - recompensa telescópica: α·log(f_max(s_{t+1}) / f_max(s_t));
  - el control a nivel de **población** se aprendió bien; el control por individuo casi nunca superó a la base;
  - entre 21 agentes entrenados igual, el resultado varió mucho, y elegir por validación ayuda.
- **DAC** (Biedenkapp et al., 2020; Adriaensen et al., 2022):
  - formulación como MDP contextual, con recompensa = −costo del paso;
  - Shala et al. (2020) usaron como recompensa el área bajo la curva del mejor-hasta-ahora, que es densa y premia el
    desempeño en todo momento;
  - además partieron de una heurística experta (CSA) como "profesor".
- **Sharma et al. (2019), DE-DDQN:** entrenamiento fuera de línea con partición entrenamiento/prueba de instancias, y
  comparación de varias definiciones de recompensa.
- **Andrychowicz et al. (2021)**, estudio de >250 mil agentes on-policy:
  - inicializar la **última capa de la política con pesos 100× menores**, para que la acción parta centrada en 0;
  - activación tanh;
  - actor y crítico separados, pudiendo el crítico ser más ancho;
  - normalizar observaciones;
  - GAE con λ ≈ 0,9 y sin recorte de la pérdida del valor;
  - Adam con lr = 3·10⁻⁴;
  - γ es de los hiperparámetros más influyentes;
  - repetir la acción varios pasos ("frame skip").
- **Ng, Harada y Russell (1999):** el *shaping* basado en potencial, F = γΦ(s′) − Φ(s), no cambia la política óptima.
- **MetaBox (Ma et al., 2023):**
  - algunos métodos MetaBBO-RL (p. ej., LDE) quedan por debajo de optimizadores clásicos bien ajustados (JDE21, MadDE) en
    parte de las pruebas;
  - propone protocolos con partición entrenamiento/prueba y métricas de generalización y transferencia (MGD, MTE).

---

## 2. El MDP

### 2.1 Episodio y reloj de decisión [D]

- **Episodio:** una corrida de NEAT con semilla s y G generaciones (B = G · población evaluaciones).
- **Decisiones:** T por corrida, en fracciones fijas del presupuesto.
  - La decisión t (t = 0 … T−1) se toma tras evaluar la generación g_t = ⌊t·(G−1)/T⌋.
  - Su P rige la reproducción de las generaciones g_t+1 … g_{t+1}.
  - El episodio termina en g_T = G−1.
- **T = 50**, con estas k = (G−1)/T generaciones por decisión a 75 mil evaluaciones:

| bottleneck | acrobot-swingup | l2f |
|---|---|---|
| k ≈ 21 | k ≈ 12 | k ≈ 7 |

- **Las métricas se extraen en cada generación** [P] y se agregan en la ventana de la decisión.

Por qué no decidir en cada generación:

1. **Señal.** El efecto de un P que rige una sola generación es diminuto frente al ruido entre corridas (DE ≈ 0,04). Repetir
   la acción k generaciones (el "frame skip" de Andrychowicz et al.) hace medible cada decisión.
2. **Transferencia.** El mismo horizonte en los tres entornos (T = 50) y un reloj relativo, como el (T−t)/T de *Learning to
   Evolve*, permiten aplicar una política en otro entorno o con otro presupuesto.
3. **Costo.** El costo de PPO es despreciable en ambos casos. Lo que limita es la cantidad de corridas de NEAT, no la de
   decisiones.

Decidir en cada generación (k = 1) queda como ablación si sobra tiempo (§9.2).

```
generación: 0 ─ 1 … g1 ─ g1+1 … g2 ─ … ─ g49+1 … G−1
            ↑ s0, a0     ↑ s1, a1        ↑ s49, a49  ↑ s50 (terminal)
            P(a0) rige 1…g1;  r0 = Φ(g1) − Φ(0) + (w/T)·Φ(g1)
```

### 2.2 Acción [D]

a_t ∈ ℝ^K es gaussiana. Su media sale del actor (MLP) y su desviación estándar es aprendida e independiente del estado
(`mlp_unconditional_stddev` de RLtools). El entorno la convierte en P:

```
P_t = softmax( log P₀ + c · tanh(a_t / c) ),   c = 3
```

- **P₀ = P estática ajustada del catálogo final** (la de la línea base, tras el reajuste del catálogo), con un piso de 0,002
  renormalizado para que ningún operador quede inalcanzable.
  - Con la última capa del actor inicializada 100× menor, **la política inicial es la línea base estática** y PPO solo
    aprende desviaciones.
  - Es el mismo principio de partir de una heurística experta de Shala et al. (2020).
- **Desviación acotada:** el cociente P_k/P_j puede alejarse a lo más e^{±2c} de su valor en P₀. No hay colapso a un solo
  operador; ±c es la "región de confianza" de la mezcla.
- Cerca de 0, c·tanh(a/c) ≈ a. Con la desviación estándar inicial de 0,5, la exploración mueve cada logit ±0,5 (factor 1,6).
- La log-probabilidad de PPO se calcula sobre a, antes de la transformación. La transformación es parte del entorno y no
  sesga el gradiente.
- **En la evaluación**, se usa la acción media (determinista), como en *Learning to Evolve*.
- K es una constante de compilación (`ACTION_DIM`) igual al tamaño del catálogo final.

Descartado:

| Alternativa | Motivo |
|---|---|
| Acción discreta | RLtools no la soporta, y el esquema por hijo necesita una distribución igual |
| Una P por especie o por individuo | Más dimensiones. En *Learning to Evolve* el control por individuo casi nunca se aprendió |
| softmax(a) sin P₀ | Parte de P uniforme, que en bottleneck fue claramente peor (0,13 contra 0,27) |

### 2.3 Estado [D]

σ(J) = (J − J_trivial) / (J_RL − J_trivial) es el score de siempre.

| # | Rasgo | Definición | Rango | Para qué |
|---|---|---|---|---|
| 1 | τ | evaluaciones usadas / B | [0, 1] | Horizonte fijo: sin τ el MDP no es markoviano (Pardo et al., 2018) |
| 2 | V* | σ de la mejor validación hasta ahora (la del campeón vigente) | ~[0, 1] | Nivel alcanzado |
| 3 | V_gen | σ media, en la ventana, de la validación del campeón de cada generación | ~[0, 1] | Nivel actual de la población |
| 4 | V* − V_gen | brecha | ≥ 0 | Regresión de la población (desaprendizaje) |
| 5 | ΔV* | cambio de V* en la última ventana | ≥ 0 | Progreso reciente |
| 6 | estancamiento | τ − τ de la última mejora de V* | [0, 1] | Estancamiento (Pei) |
| 7 | dispersión | DE del fitness de entrenamiento de la última generación / (J_RL − J_trivial) | ≥ 0 | Diversidad de desempeño |
| 8 | especies | n.º de especies / especies objetivo | ~1 | Especiación |
| 9 | especie mayor | tamaño / población | [0, 1] | Convergencia |
| 10 | umbral δt | log(δt / δt inicial) | ℝ | Diversidad genética (la especiación dinámica ajusta δt) |
| 11 | conexiones | log(conexiones medias / conexiones iniciales) | ≥ 0 | Complejidad |
| 12 | ocultas | log(1 + ocultas medias) | ≥ 0 | Complejidad |
| 13 | conexiones del campeón | log(conexiones / iniciales) | ≥ 0 | Complejidad del mejor |
| 14 | ocultas del campeón | log(1 + ocultas) | ≥ 0 | Complejidad del mejor |
| 15 … 14+K | crédito_k | media móvil (η = 0,5 por ventana) del percentil de los hijos del operador k en su generación, menos 0,5 | [−0,5, 0,5] | Lo que usa AP, ahora condicionado al estado (crédito por rango, Pei) |
| 15+K … 14+2K | fallo_k | fracción de aplicaciones del operador k que no pudieron aplicarse | [0, 1] | P. ej., duplicar o borrar neurona sin ocultas |

- **Dimensión:** 14 + 2K. Con K = 8 son 30 rasgos; con K = 15, 44.
- **Todo está disponible en el despliegue sin evaluaciones extra:** `neat_control` ya valida al campeón de cada generación.
  PPO usa el mismo presupuesto que estático y AP.
- **Crítico con información privilegiada** (`ObservationPrivileged` de RLtools):
  - recibe además Φ, el score del campeón en el conjunto de recompensa (§2.4);
  - solo se usa al entrenar y reduce la varianza del valor.
- **Normalización:**
  - los rasgos ya son fracciones o logaritmos, sin unidades del entorno;
  - se suma la capa `standardize` de RLtools, con media y DE estimadas del primer lote de entrenamiento y luego fijas;
  - esas estadísticas viajan con el actor exportado.
- **Fuera de la versión 1:**
  - actividad de la SNN (neuronas silenciosas): requiere instrumentar el evaluador; candidata para la versión 2, según el
    diagnóstico de interfaz;
  - rasgos de paisaje de fitness (Pei): caros;
  - retornos crudos: tienen unidades del entorno.

### 2.4 Recompensa [D]

**Potencial.** Φ_t = σ(J_R(c_t)), donde:
- c_t es el campeón vigente (mejor validación hasta la decisión t);
- J_R es su retorno medio en M_R = 50 episodios de un **stream propio (3)**, fijo para la corrida, que no toca entrenamiento
  (1000 + g), validación (1) ni test final (2);
- solo se recalcula cuando cambia el campeón: 14–26 veces por corrida (máx. 37), ≈ 0,2 % de evaluaciones extra;
- solo existe al entrenar.

**Recompensa por decisión:**

```
r_t = (Φ_{t+1} − Φ_t) + (w/T) · Φ_{t+1},   w = 0,5
Retorno (γ = 1) = (Φ_T − Φ_0) + w · (1/T) Σ_{t=1..T} Φ_t
```

- **Eficacia:** el primer término suma la mejora final del campeón.
- **Eficiencia:** el segundo es el área bajo la curva, como la recompensa de Shala et al. (2020). Premia llegar antes, lo que
  corresponde a "generaciones hasta el umbral".
- **Desaprendizaje:**
  - ΔΦ < 0 cuando el nuevo campeón, mejor en validación, es peor en el conjunto de recompensa. Así se castiga el sobreajuste
    a la validación.
  - Opcionalmente, *shaping* con Φ_gen = V_gen y Φ_gen(terminal) = 0: castiga que la población retroceda **sin cambiar la
    política óptima** (Ng et al., 1999). Queda apagado por defecto y se enciende si el crítico no aprende.
- **Escala:** recompensas × 10 para que el retorno sea O(1).
- El **test final** (stream 2, 100 episodios) se mide al terminar cada corrida de entrenamiento solo para monitorear. Nunca
  entra al entrenamiento.

Descartado:

| Alternativa | Motivo |
|---|---|
| Mejor validación hasta ahora (lo que ve NEAT) | Es el máximo de medidas de 10 episodios a lo largo de cientos de generaciones: tiene sesgo de ganador. El agente aprendería a producir candidatos de alta varianza |
| Solo el test final | Una señal por cada 50 decisiones |
| Crédito por hijo | Es la asignación de crédito de AP: un sustituto miope, no el objetivo de la tesis |
| log del cociente (*Learning to Evolve*) | Hay retornos negativos (acrobot). Las diferencias en score cumplen el mismo papel telescópico |

### 2.5 Terminación, descuento y ventaja [D]

- `terminated` cuando t + 1 = T, con V(s_T) = 0 (RLtools: `IGNORE_TERMINATION = false`).
  - No hay truncamientos: `EPISODE_STEP_LIMIT` es mayor que T.
- **γ = 0,99 y λ = 1**, elegidos en el entorno sintético (§5.1).
  - La propuesta inicial era γ = 1 y λ = 0,95. Con pocos datos, esa combinación empeora la política antes de mejorarla.
  - λ = 1 calcula la ventaja con el retorno observado completo y no depende de un crítico todavía malo. γ = 0,99 reduce la
    varianza de las recompensas lejanas.
  - Costo: con γ = 0,99 y T = 50, el final de la corrida pesa 0,6 en el retorno, una preferencia leve por ganar antes. En el
    sintético no bajó el score final.

---

## 3. NEAT como entorno de RLtools

### 3.1 Lo que exige RLtools [V]

Revisado en `external/rl-tools` (rama `render` @ 6bb2fda):

- **El runner on-policy es genérico** en el tipo de entorno por lotes (`on_policy_runner::Specification<…, BATCH_ENVIRONMENT, …>`).
  - `collect` llama, sobre el lote: `step`, `reward`, `terminated`, `observe`, `sample_initial_parameters` y
    `sample_initial_state`. Las dos últimas reciben una máscara de reinicio.
- **El único lote que trae, `batch::Independent`, recorre las instancias en secuencia.** Con NEAT evaluaría una corrida a la
  vez.
- **El núcleo del loop de PPO** (`ppo::loop::core::Config`) fija `batch::Independent` en el código.
- **El runner copia los estados por valor** (`next_states → states`), así que State debe ser POD.
- **Orden de cada paso:**
  1. `step`;
  2. `reward(s, a, s′)`;
  3. `terminated(s′)`;
  4. copia de estados;
  5. registro de la transición (reinicio si terminó);
  6. `observe(s′)` privilegiada, para el *bootstrap*;
  7. `sample_initial_state(máscara)`;
  8. observación de la fila siguiente.
- **Acciones:** gaussianas (salida IDENTITY, log σ aprendido e independiente del estado), sin acotar. El entorno las
  transforma.
- **GAE:** V = 0 al terminar y *bootstrap* al truncar.
- **`train`:**
  - baraja las transiciones y hace N_EPOCHS pasadas en minilotes;
  - aplica recorte;
  - suma entropía sobre log σ;
  - el crítico usa MSE sin recorte;
  - la tasa de aprendizaje adaptativa por KL es opcional.
- **`NORMALIZE_OBSERVATIONS` del núcleo hace una colecta completa extra** solo para estimar estadísticas, y luego reinicia
  todos los entornos. Con NEAT eso tira ~16 corridas, así que no se usa.

### 3.2 Diseño: un lote propio, `NeatBatch` [D]

**Tipos:**
- **`NeatEnv`** (tipo de instancia, solo definiciones):
  - `T`, `TI`;
  - `Parameters` vacío;
  - `State {slot, t, phi, phi_gen}` (POD);
  - `Observation` (DIM = 14 + 2K) y `ObservationPrivileged` (DIM + 1);
  - `N_AGENTS = 1`, `ACTION_DIM = K`, `EPISODE_STEP_LIMIT = T + 1`.
- **`NeatBatch`:**
  - `INSTANCES = N`;
  - un puntero a un `RunPool` con N `NeatRun`, la configuración de NEAT y del entorno, el contador de semillas, el catálogo,
    P₀, c, T, M_R y los registros.

**Operaciones** (sobrecargas en `namespace rl_tools`; las encuentra ADL por los argumentos `Tensor`):

| Función | Qué hace |
|---|---|
| `sample_initial_state(máscara)` | Por cada ranura marcada: nueva `NeatRun` con la siguiente semilla de entrenamiento y evaluación de la generación 0 (todas las ranuras en una sola región paralela). Calcula rasgos y Φ₀, y escribe `State{slot, 0, Φ₀}` |
| `step(s, a → s′)` | P_i = actionToP(a_i). Luego, k veces: reproducir cada corrida (en paralelo entre corridas, cada una con su RNG); evaluar todas las tareas (corrida, genoma, episodio) en una región OpenMP con reparto dinámico; validar al campeón de cada generación; `endGeneration`; acumular rasgos. Al cerrar la ventana calcula Φ si cambió el campeón, y escribe `s′ = {slot, t+1, Φ′}` |
| `reward(s, a, s′)` | Solo con los campos POD |
| `terminated(s′)` | s′.t == T |
| `observe(s, tipo, obs)` | Copia los rasgos guardados; la versión privilegiada agrega Φ |

**Determinismo:**
- El RNG de NEAT y las semillas de episodio de cada corrida salen de su semilla, como en `neat_control`.
- El RNG del runner solo muestrea acciones.
- Por lo tanto, una corrida con una secuencia dada de P es **idéntica bit a bit** a `neat_control` con esa misma secuencia
  (test V1).

**Paralelismo:**
- Las tareas se aplanan a (corrida, genoma, episodio). Por generación son 16·70·10 = 11 200 tareas en bottleneck y
  16·210·10 = 33 600 en l2f.
- El fitness es la media de los episodios en orden fijo, así que el resultado no depende de los hilos.

**Configuración:**
- Se deriva de `ppo::loop::core::Config` y se reemplazan `BATCH_ENVIRONMENT`, los tipos `ON_POLICY_RUNNER_*` y
  `CRITIC_COLLECTION_*`.
- El `State` del núcleo y sus funciones son genéricos en la configuración.

**Bucle propio** (replica `loop::core::step`):
1. `collect`;
2. solo en la primera iteración: estadísticas de `standardize` en modo acumulación, con ese mismo lote;
3. GAE;
4. `train`;
5. registro;
6. *checkpoint* cada 2 iteraciones.

Los *checkpoints* caen al final de las corridas, así que no hace falta serializar NEAT.

```
RLtools (runner + PPO)                 NeatBatch (nuestro)                         NEAT + SNN (src/neat, src/eval)
collect: observe ────────────────▶ rasgos guardados de cada corrida
         actor(obs) → a ~ N(μ, σ) ─▶ step: P_i = softmax(log P₀ + c·tanh(a_i/c))
                                      k veces: reproducir(P_i) en cada corrida ───▶ Population::reproduce
                                               evaluar (corrida, genoma, episodio) ─▶ evaluador, en paralelo
                                               validar campeón, endGeneration, rasgos
                                      Φ si cambió el campeón (stream 3)
         reward(s, a, s′) ◀──────── ΔΦ + (w/T)·Φ′  (campos de State)
         terminated ◀─────────────── t′ == T
         reinicio (máscara) ────────▶ nueva corrida: siguiente semilla, generación 0
GAE → train (10 épocas) → registro → checkpoint
```

### 3.3 Cambios en NEAT: mínimos y verificables [D]

1. **`GenerationReport`** gana cuatro campos:
   - validación del campeón de la generación;
   - mejor validación hasta ahora;
   - evaluaciones usadas;
   - percentil de cada hijo (crédito por rango), calculado con una función compartida con `--credito-rango`.

   Estático y AP los ignoran, así que su resultado no cambia (verificación bit a bit con el binario congelado).
2. **`NeatRun`** (`src/aos`): una corrida de NEAT que avanza por ventanas, con el mismo ciclo de `neat_control` (rotación,
   validación, test final) más el conjunto de recompensa (solo al entrenar).
3. **Evaluador de varias corridas** (`src/eval`): mismos retornos que `evaluatePopulation` (test).
4. **`FeatureExtractor` y `actionToP`** (`src/aos`): compartidos por el entrenador y el despliegue.

### 3.4 Despliegue en `neat_control` [D]

- **`PpoSelector`** implementa `OperatorSelector`:
  - `update(report)` en cada generación (rasgos);
  - `probabilities()` cambia P solo en las fronteras de decisión.
- **Política:** el actor se exporta con `save_code` de RLtools (`checkpoint.h`) y se compila como `.so` con símbolos ocultos,
  igual que `rl_ref_eval` con `rl_policy_shim.cpp`. `neat_control` lo carga con `dlopen`.
  - El `.so` lleva metadatos: catálogo, P₀, c, T, K y versión de rasgos. `neat_control` se niega a correr si no coinciden con
    la corrida.
  - Trae un autotest con los ejemplos del checkpoint.
- **Ejecución:** `neat_control --selector ppo --policy <.so>`. Un mismo binario congelado sirve para los tres métodos, sin
  evaluaciones extra (Φ no se calcula en el despliegue).

### 3.5 Archivos

| Archivo | Contenido |
|---|---|
| `src/aos/features.{hpp,cpp}` | `FeatureExtractor`: rasgos por generación → observación por decisión |
| `src/aos/action_map.hpp` | `actionToP` y fronteras de decisión |
| `src/aos/neat_run.{hpp,cpp}` | `NeatRun` |
| `src/eval/evaluator.{hpp,cpp}` | Evaluación de varias corridas en una región paralela |
| `src/aos/ppo_selector.{hpp,cpp}` | `PpoSelector` (despliegue) |
| `src/experiment/aos_ppo_train.cpp` | `NeatEnv`/`NeatBatch`, configuración derivada, bucle, registros y *checkpoints* |
| `src/experiment/aos_toy.hpp` | Entorno sintético por lotes (§5) |
| `src/experiment/ppo_policy_shim.cpp`, `scripts/compilar_politica_ppo.sh` | Actor exportado → `.so` |
| `tests/ppo_tests.cpp` | V0–V2 y equivalencia de la exportación |

---

## 4. Entrenamiento

### 4.1 Hiperparámetros iniciales [D]

| Parámetro | Valor | Por defecto en RLtools | Fuente |
|---|---|---|---|
| Corridas en paralelo (`N_ENVIRONMENTS`) | 16 | 64 | Paralelismo suficiente con tareas aplanadas |
| Pasos por corrida e iteración | 25 (media corrida) | 64 | 400 transiciones por iteración |
| Minilote (`BATCH_SIZE`) | 100 | 512 | 4 minilotes × 10 épocas |
| `N_EPOCHS` | 20 | 10 | Sintético (§5.1): aprende el doble de rápido que con 10; con 4, no aprende |
| γ | 0,99 | 0,99 | §2.5 y §5.1 |
| λ | 1,0 | 0,95 | §2.5 y §5.1: sin bache inicial |
| Recorte ε | 0,2 | 0,2 | Andrychowicz et al.: 0,25 |
| Desviación inicial de la acción | 0,5 | 0,5 | ±0,5 en logits |
| Coeficiente de entropía | 0,01 | 0,01 | |
| Adam α del actor | 3·10⁻⁴ | 10⁻³ | Andrychowicz et al. Con 10⁻⁴ (actor y crítico) no aprende (§5.1) |
| Adam α del crítico | 10⁻³ | 10⁻³ | Sintético: algo mejor al inicio y menos deriva sin señal |
| Tasa adaptativa por KL | desactivada; se decide en el sintético | desactivada | En RLtools baja α si la KL supera 2 × 0,008, pero también lo sube hasta 10⁻² si la KL es chica |
| KL aproximada | se registra igual, con log-probabilidades antiguas y nuevas | solo con la tasa adaptativa | Monitoreo (§4.6) |
| Actor | 2 ocultas × 64, `FAST_TANH` | 3 capas × 64, RELU | Andrychowicz et al.; el zoo usa `FAST_TANH` |
| Crítico | 2 ocultas × 128, `FAST_TANH` | 64, RELU | Crítico más ancho |
| Última capa del actor | pesos × 0,01 al iniciar | — | Andrychowicz et al.; arranque en P₀ |
| Observaciones | estadísticas del primer lote, luego fijas | colecta extra | §3.1 |
| T, c, w, M_R, escala de r | 50, 3, 0,5, 50, 10 | — | §2 |

### 4.2 Presupuesto de entrenamiento y costo [D]

**Presupuesto (revisado con el paso 1, §5.1):**
- En el sintético, calibrado con nuestro ruido, la configuración final capturó esta fracción de la ganancia alcanzable:

  | Corridas de entrenamiento | 192 | 384 | 768 |
  |---|---|---|---|
  | Ganancia capturada | ~33 % | ~60 % | ~85 % |

  Con 192, la mejora (+0,018) queda muy por debajo de lo detectable con 11 réplicas (§1.3).
- **R ≥ 400 corridas por entorno, idealmente ~800** (bloques de 16), más 20 de validación de checkpoints (§4.4).
- El sintético es más fácil que NEAT: sus rasgos son limpios y su política óptima es simple. Estas cifras son un **mínimo**.
- Supera lo que gastaron las líneas base en ajuste, y hay que reportarlo (§4.5):
  - estático: Optuna con 100 trials × 3 semillas = 300 corridas;
  - AP: ~50 trials × 3 = 150 corridas.

**Costo a 75 mil evaluaciones**, con el costo por corrida de §1.3 y ~400 hilos al ~85 %:

| | l2f | acrobot-swingup | bottleneck |
|---|---|---|---|
| 420 corridas | ≈ 8 h | ≈ 12 h | ≈ 2 días |
| 820 corridas | ≈ 15 h | ≈ 1 día | ≈ 4 días |

Para bottleneck hay una alternativa: entrenar con más corridas en acrobot-swingup y l2f, que son baratos, y transferir
(§7). Es la extensión que el MDP transferible permite.

**[!] A 300 mil evaluaciones**, el costo por corrida sube:
- ≈ 3× en l2f (medido en la corrida de 300 mil ya terminada);
- del orden de 10× en acrobot y 10–20× en bottleneck (estimado de las corridas de 300 mil en curso, cuyas redes siguen
  creciendo).

Entrenar PPO en bottleneck pasaría a **semanas** con todo el nodo. Esto es un insumo directo para decidir el presupuesto
común (§9.5).

### 4.3 Semillas y particiones [D]

| Uso | Semillas de NEAT |
|---|---|
| Ajuste con Optuna (hecho) | 101–103 |
| Sondas | 201–210 |
| Entrenamiento de PPO | 1001–1192, por entorno |
| Validación de checkpoints | 2001–2010 |
| Evaluación final | 1–11 (la propuesta pide 11 réplicas); nunca vistas al entrenar |

**Streams por corrida:**

| Uso | Stream |
|---|---|
| Entrenamiento (rotación) | 1000 + generación |
| Validación | 1 |
| Test final | 2 |
| Conjunto de recompensa (nuevo; solo al entrenar) | 3 |

### 4.4 Checkpoints y número de agentes [D]

- **Checkpoint cada 2 iteraciones:** actor, crítico, optimizadores, RNG y contador de semillas. Permite reanudar tras una
  interrupción.
- **Selección:**
  - se validan el último checkpoint y el de mejor retorno de entrenamiento, en las semillas 2001–2010, con la acción
    determinista;
  - en las mismas semillas corre la P₀ estática, de modo que la comparación queda pareada;
  - gana el de mejor score medio de validación.
- **Número de agentes:** *Learning to Evolve* mostró mucha varianza entre agentes entrenados igual.
  - Si el cómputo alcanza, se entrenan 2–3 agentes (semillas de PPO distintas, mismas semillas de NEAT) en acrobot-swingup y
    l2f, que son baratos. En bottleneck, uno.
  - Se reportan **todos** los agentes entrenados, no solo el elegido.

### 4.5 Equidad con las líneas base [P/D]

- Misma configuración de NEAT, catálogo, presupuesto y semillas. En el despliegue PPO no usa evaluaciones extra.
- Se reporta también el costo "meta" de cada método, en corridas y en hilo-h: el ajuste con Optuna de estático y AP, y el
  entrenamiento más la validación de PPO. `comprension.md` §4 ya lo pedía.

### 4.6 Registro

- **`decisiones.csv` por corrida:**
  - semilla, t, τ, generación y evaluaciones;
  - observación, a y P;
  - Φ, r y V*.
- **`ppo.csv` por iteración:**
  - retorno medio y score final medio (test, solo para monitorear) de las corridas cerradas;
  - P media por operador;
  - log σ;
  - KL aproximada, pérdida del crítico, varianza explicada y fracción recortada.
- Las corridas mantienen `generations.csv` y `summary.txt`, para reutilizar los análisis existentes.

---

## 5. Escalera de verificación (antes de gastar clúster)

| Paso | Qué | Criterio de aceptación | Costo |
|---|---|---|---|
| V0 | Tests unitarios: `actionToP`, rasgos con casos sintéticos, percentiles | `actionToP`: a = 0 → P₀, suma 1, cotas respetadas. El resto pasa | minutos |
| V1 | `NeatRun` contra `neat_control`, con P fija y con secuencia de P (`--p-switch`) | Idénticos bit a bit (CSV y test final) | minutos |
| V2 | Contabilidad del runner con un `NeatBatch` falso | Cada corrida tiene exactamente T pasos, ninguna semilla se repite y el reinicio ocurre al terminar | segundos |
| V3 | Entorno sintético (abajo) | 1) PPO supera a la mejor P fija del sintético con ~200 corridas × 50 decisiones. 2) Si el óptimo es estático, no empeora a P₀. 3) **Control negativo**: con recompensa barajada no aprende | minutos |
| V4 | Integración en pendulum (barato), presupuesto corto | Corre de punta a punta y reanuda. La exportación da las mismas P que el entrenador en observaciones grabadas | ~1 h |
| V5 | Piloto en acrobot-swingup, ~96 corridas | Curva de retorno registrada. En validación, el agente no queda por debajo de P₀ | ~3 h |
| V6 | Entrenamiento completo por entorno | §4 | §4.2 |

**Entorno sintético (V3).** Es un "AOS de juguete" con la misma interfaz de lote; se elige al compilar.
- **Estado oculto:** calidad q y complejidad c.
- **Operadores sintéticos de tres arquetipos:**
  - pesos: ganancia con rendimiento decreciente hasta una capacidad que crece con c;
  - crecimiento: sube c, cuesta al principio y paga después, hasta un c*;
  - poda: baja c, y conviene si c > c*;

  más operadores neutros.
- **Ruido calibrado con NEAT:** DE del final entre corridas ≈ 0,04, y brecha entre la mejor P fija y la mejor política
  adaptativa ≈ 0,05.
- **Usos:**
  - depurar el entrenador sin NEAT;
  - elegir γ (1 o 0,99), λ (0,9 o 0,95), N (8 o 16), T (25, 50 o 100) y la tasa de aprendizaje;
  - estimar **cuántas corridas necesita PPO para capturar una ganancia de 0,05 con este ruido**, una "potencia del
    entrenamiento".

Cuesta segundos y responde lo que en NEAT costaría días.

### 5.1 Resultados del paso 1: PPO en el AOS sintético (2026-10-13) [V]

**Código:**

| Archivo | Contenido |
|---|---|
| `src/aos/action_map.hpp` | `actionToP`, `logP0` y fronteras de decisión |
| `src/experiment/aos_toy.hpp` | El sintético: corrida, pool y política oráculo |
| `src/experiment/ppo_lote.hpp` | Lote propio, operaciones y `ConfigConLote` |
| `src/experiment/aos_toy_ppo.cpp` | Modos `calibrar`, `evaluar`, `entrenar` y `verificar` |
| `tests/ppo_tests.cpp` | Tests V0 (362 aserciones) |

Compiló al primer intento. **V2 (contabilidad del runner) aprobado:** 112 corridas iniciadas, 96 terminadas con exactamente
50 decisiones cada una, semillas sin repetir.

**Calibración** (4000 corridas nuevas):

| Política | Score | DE |
|---|---|---|
| Mejor P fija (pesos 0,87, + neurona 0,11, + conexión 0,02) | 0,681 | 0,046 |
| Oráculo | 0,734 | 0,045 |
| Uniforme | 0,256 | — |

- La ganancia del oráculo sobre la mejor P fija es +0,054 (ee 0,0005): DE y ganancia quedan del orden de nuestros números de
  NEAT (§1.3).
- La uniforme es mucho peor, como en bottleneck.
- Sin estado, la mejor P fija ("solo pesos") es óptima.

**Resultados:** cuatro baterías, 290 entrenamientos, 10 semillas de PPO por configuración. Datos en
`~/tesis/runs/ppo_sintetico/` (`resumen.py`, `curvas_sintetico.png`).

Δ = score de PPO − score de P₀ en las mismas 1920–2000 corridas de evaluación, con la política determinista:

| Configuración | Δ a 96 corridas | 192 | 384 | 768 | Semillas que aprenden a 768 |
|---|---|---|---|---|---|
| Inicial (γ = 1, λ = 0,95, 10 épocas) | −0,003 | −0,010 | +0,008 | +0,038 | 10/10 |
| γ = 0,99, λ = 1 | +0,003 | +0,008 | +0,018 | +0,040 | 10/10 |
| … + 20 épocas | +0,007 | +0,018 | +0,033 | +0,046 | 10/10 |
| **… + crítico con α = 10⁻³ (final)** | **+0,009** | **+0,018** | **+0,033** | **+0,046** | **10/10** |
| γ = 0,99, λ = 1, hasta 1536 corridas | | +0,009 | +0,016 | +0,039 | +0,047 a 1536 |
| Control: recompensa barajada (inicial) | −0,006 | −0,024 | −0,070 | −0,081 | 0/10 |
| Control: recompensa barajada (final) | −0,003 | −0,002 | −0,011 | −0,009 | 3/10 |
| Control: óptimo estático (final) | 0,000 | 0,000 | 0,000 | −0,0004 | — |

Variantes que **empeoran**: tasa 10⁻⁴ (−0,033 a 768), 4 épocas (−0,003), N = 8 (−0,077 a 192), σ inicial 0,3 (más lento) y
λ = 0,9. Sin efecto en el score final: w = 0 frente a 0,5, precalentar el crítico, N = 32 y recorte 0,3.

**Lectura:**
1. **La integración con RLtools funciona y PPO aprende** con un ruido como el nuestro: 10/10 semillas, ~85 % de lo alcanzable a
   768 corridas.
2. **192 corridas no alcanzan:** a lo más ~33 % de la ganancia. Con la configuración inicial, PPO incluso empeoraba (un bache
   hasta ~350 corridas). El presupuesto de entrenamiento sube a ≥ 400 corridas (§4.2).
3. **Configuración final:** γ = 0,99, λ = 1, 20 épocas, α 3·10⁻⁴ en el actor y 10⁻³ en el crítico. Todo indica que con pocos
   datos lo crítico es que el crítico aprenda rápido, o que la ventaja no dependa de él (λ = 1).
4. **Sin señal, PPO deriva:** con la recompensa barajada, la configuración inicial empeora hasta −0,08. La final deriva mucho
   menos (−0,01), y cuando el óptimo es estático no se mueve. Aun así, el pre-chequeo (§6) y validar checkpoints contra P₀
   (§4.4) siguen siendo necesarios.

### 5.2 Resultados del paso 2: NEAT como entorno (2026-10-13) [V]

**Código:**

| Archivo | Contenido |
|---|---|
| `src/neat/selector.{hpp,cpp}` | `rankCredit` (el cálculo de `--credito-rango`, ahora compartido) y los campos nuevos del reporte de generación (validaciones, evaluaciones y crédito por rango) |
| `src/neat/population.cpp` | Fracción de la especie mayor en las estadísticas |
| `src/aos/features.{hpp,cpp}` | `FeatureExtractor`: los 14 + 2K rasgos de §2.3 |
| `src/eval/evaluator.{hpp,cpp}` | `evaluatePopulations`: varias poblaciones en una sola región paralela |
| `src/aos/neat_run.{hpp,cpp}` | `NeatRun` (una corrida por ventanas de decisión, con Φ en el stream 3) y `NeatPool` (corridas en paralelo, con la misma interfaz que el pool del sintético) |
| `src/experiment/opciones_neat.hpp` | Las opciones de NEAT con los mismos nombres, valores por defecto y semántica que `neat_control` |
| `src/experiment/aos_neat_check.cpp` | Herramienta de verificación V1 |
| `tests/neat_run_tests.cpp` | 5 tests, 153 aserciones |
| `scripts/verificar_neat_run.sh` | V1 contra los binarios |

En total pasan 87/87 tests.

**Tests unitarios:**
- `NeatRun` coincide bit a bit con el ciclo de `neat_control` escrito a mano, con un cambio de P en una frontera.
- `NeatPool` con 3 corridas en paralelo coincide con las mismas corridas por separado: observaciones, Φ y test final.
- `evaluatePopulations` da el mismo fitness que `evaluatePopulation`.

**V1 contra los binarios** (`~/tesis/runs/verif_neat_run.log`, sin fallas): `aos_neat_check` coincide bit a bit con
`neat_control` en `generations.csv` (salvo los segundos), test final y mejor validación. Casos:
- pendulum con la P de Stanley;
- pendulum con cambio de P (`--p-switch`);
- pendulum con el catálogo de 15 operadores y crédito por rango;
- acrobot-swingup #49, l2f #65 y bottleneck #96, con sus líneas de comando completas;
- el camino paralelo (`NeatPool`, 4 semillas) frente a 4 corridas de `neat_control`.

**Controles:**
- El `neat_control` recompilado coincide con el binario congelado de la sonda (sha ae239e87…): los cambios en `src/neat` no
  alteran nada.
- Otra semilla da otro CSV: la comparación detecta diferencias.

PPO se entrenará sobre exactamente el mismo NEAT con el que después se compara.

### 5.3 Resultados del paso 3: entrenador sobre NEAT (2026-10-13) [V]

**`src/experiment/aos_ppo_train.cpp`:**
- PPO de RLtools sobre `NeatPool`, con la configuración final del paso 1;
- las mismas opciones de NEAT que `neat_control`;
- K de 3 a 10 y 15;
- estandarización de observaciones con las 2 primeras iteraciones;
- checkpoint en tar cuando todas las corridas recién empiezan (`--reanudar`);
- exportación del actor como código, con metadatos para el despliegue;
- validación pareada contra P₀ al final (`--validar`);
- registros `ppo.csv`, `corridas.csv` y `decisiones.csv`.

RLtools va envuelto en el namespace `ppo_rlt`: `libenvs` también lo incluye y algunas de sus funciones no son inline, así
que sin el envoltorio el enlazado choca.

**V4 en pendulum** (`scripts/verificar_ppo_pendulum.sh`; población 30, T = 50, 64 corridas, 8 iteraciones de ~30 s con 8 hilos):
- **Entrena de punta a punta.**
- **Reanudar es exacto:** detener en la iteración 4 y reanudar da lo mismo que correr de corrido, en `ppo.csv` (salvo los
  segundos), `corridas.csv`, `decisiones.csv` y el actor exportado.
- La validación (16 semillas: PPO 0,338 contra P₀ 0,358, Δ −0,021, ee 0,018) solo prueba el circuito. Con 64 corridas no se
  espera aprendizaje (§5.1).

---

## 6. Pre-chequeo: ¿hay algo que aprender? [D]

La sonda del catálogo ya trae dos evidencias (`operadores_candidatos.md` §7):

- **Etapa 4:** Optuna con P₁, P₂ y la generación del cambio, contra el mejor P fijo. Si un P que cambia gana, hay margen para
  adaptar, por igual para AP y PPO.
- **Etapa 5:** registros por generación del éxito de cada operador junto con los rasgos de la población. Con ellos se arman
  "mapas de utilidad": el crédito por rango de cada operador según τ y según el estado (complejidad, estancamiento).
  - Si el mejor operador cambia a lo largo de la corrida, o con el estado, una política condicionada tiene qué explotar.
  - Además, esos registros fijan la lista final de rasgos de §2.3.

Si ninguna muestra margen, PPO convergerá a ≈ P₀: un resultado nulo, reportable, pero que conviene anticipar. En ese caso se
discute el alcance antes de gastar clúster. Por ejemplo:
- otras dimensiones de acción, como la tasa de cruce o la potencia de mutación;
- o un entorno distinto.

---

## 7. Evaluación final (resumen)

- **Configuraciones, por entorno:**
  - estático (P₀);
  - AP ajustado;
  - PPO congelado y determinista.

  Las tres con 11 réplicas (semillas 1–11), el mismo presupuesto y el mismo binario congelado.
- **Métricas y pruebas:** las de la propuesta (§3.4).
- **Complementos de bajo costo:**
  - curvas de desempeño a lo largo de la corrida;
  - trayectorias de P aprendidas (P contra τ) frente a las de AP;
  - **ablación "solo tiempo"** (observación = τ), que separa "calendario aprendido" de "uso del estado"; opcional y solo en
    los entornos baratos.
- **Extensión de transferencia:** entrenar en acrobot-swingup + l2f, con un pool heterogéneo y el mismo T, y probar en
  bottleneck sin reentrenar. Las métricas MGD y MTE de MetaBox sirven para reportarla.

---

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| [!] **Poca señal: PPO no aprende** con las corridas disponibles | **Confirmado en el sintético:** con 192 corridas captura a lo más ~33 % y sin señal deriva (§5.1). Mitigaciones: ≥ 400 corridas, configuración final (γ = 0,99, λ = 1, 20 épocas), validación de checkpoints contra P₀ y el pre-chequeo de §6. Si falla: menos dimensiones (familias de operadores), más corridas en los entornos baratos + transferencia, o entrenar con presupuesto reducido |
| [!] **Costo de bottleneck** | Presupuesto común que lo permita (§4.2). Un agente en bottleneck |
| Varianza entre agentes | 2–3 agentes donde es barato. Reportar todos |
| Sobreajuste a la validación | Conjunto de recompensa propio (stream 3) |
| Complejidad de las plantillas de RLtools | Configuración derivada y sobrecargas, cubiertas por los tests V0–V4 |
| Choque de símbolos (RLtools en varias unidades de compilación) | Entrenador como ejecutable aparte. Despliegue vía `.so` con símbolos ocultos, como `rl_policy_shim` |
| El catálogo aún no está congelado | K como parámetro de compilación. Desarrollo con K = 15; P₀ sale del reajuste |
| Lectura de la propuesta ("la política se ajusta durante la ejecución") | Confirmar (§9.1) |

---

## 9. Decisiones a confirmar [?]

1. **Fuera de línea, por entorno** (recomendado).
   - La propuesta dice que "la política se ajusta durante la ejecución en vez de quedar fijada de antemano" (§2.3). Se lee como
     que **P** se adapta durante la corrida a través de una política condicionada al estado.
   - Aprender en línea dentro de una corrida (~50 decisiones) no da datos para una red. En la literatura, PPO en AOS siempre se
     entrena fuera de línea (Pei).
   - Ajuste fino en línea: extensión, no plan.
2. **T = 50 decisiones** (cada k generaciones; métricas en cada generación) frente a una decisión por generación (k = 1 como
   ablación).
3. **Recompensa:** w = 0,5 (peso del área bajo la curva frente a la mejora final), *shaping* apagado.
4. **Presupuesto de entrenamiento:** ≥ 400 corridas por entorno, idealmente ~800, más 20 de validación (revisado tras el
   paso 1, §4.2). En bottleneck, evaluar la alternativa de transferir desde acrobot-swingup y l2f.
5. **Presupuesto común de NEAT**, considerando el costo de entrenar PPO: con 300 mil, bottleneck se vuelve inviable.
6. **Agentes por entorno:** 1 en bottleneck; 2–3 en acrobot-swingup y l2f si hay tiempo.
7. **Ablaciones opcionales:** decisión por generación, política solo tiempo y transferencia.

---

## 10. Plan de implementación

| Hito | Contenido | Criterio | Esfuerzo |
|---|---|---|---|
| H1 | Extensión de `GenerationReport`, percentiles compartidos, `FeatureExtractor`, `actionToP` | V0. Estático y AP idénticos bit a bit | 1 día. **Hecho (2026-10-13, §5.2)** |
| H2 | `NeatRun` y evaluador de varias corridas | V1 | 1 día. **Hecho (2026-10-13, §5.2)**: también `NeatPool` |
| H3 | `NeatBatch`, configuración derivada, bucle de entrenamiento, registros y *checkpoints* | V2 | 1–2 días. **Hecho (2026-10-13, §5.3)**: V2 con el sintético y V4 en pendulum (reanudar exacto) |
| H4 | Entorno sintético y calibración | V3 | 1 día. **Hecho (2026-10-13, §5.1)** |
| H5 | Exportación, `PpoSelector` y `neat_control --selector ppo` | V4 | 1 día |
| H6 | Piloto en acrobot-swingup | V5 | ½ día + ~3 h de clúster |

**Dependencias:**
- H1–H5 no necesitan el catálogo final. Se desarrollan con K = 15 y la P de Stanley o la ajustada, y no usan clúster.
- H6 y el entrenamiento completo esperan:
  - el catálogo congelado;
  - P₀ (reajuste del estático);
  - el presupuesto común.

---

## Referencias

- Adriaensen, S., Biedenkapp, A., Shala, G., Awad, N., Eimer, T., Lindauer, M., y Hutter, F. (2022). Automated Dynamic
  Algorithm Configuration. *JAIR*, 75. https://arxiv.org/abs/2205.13881
- Andrychowicz, M. et al. (2021). What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study. *ICLR*.
  https://arxiv.org/abs/2006.05990
- Biedenkapp, A., Bozkurt, H. F., Eimer, T., Hutter, F., y Lindauer, M. (2020). Dynamic Algorithm Configuration: Foundation
  of a New Meta-Algorithmic Framework. *ECAI*.
- Eimer, T. et al. (2021). DACBench: A Benchmark Library for Dynamic Algorithm Configuration. *IJCAI*.
- Eschmann, J., Albani, D., y Loianno, G. (2024). RLtools: A Fast, Portable Deep Reinforcement Learning Library for
  Continuous Control. *JMLR*.
- Kohl, N., y Miikkulainen, R. (2012). An integrated neuroevolutionary approach to reactive control and high-level strategy
  (SNAP-NEAT). *IEEE TEC*, 16(4).
- Ma, Z. et al. (2023). MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning.
  *NeurIPS Datasets and Benchmarks*. https://arxiv.org/abs/2310.08252
- Ng, A. Y., Harada, D., y Russell, S. (1999). Policy invariance under reward transformations: Theory and application to
  reward shaping. *ICML*.
- Pardo, F., Tavakoli, A., Levdik, V., y Kormushev, P. (2018). Time Limits in Reinforcement Learning. *ICML*.
  https://arxiv.org/abs/1712.00378
- Pei et al. (2025). Adaptive Operator Selection for Meta-Heuristics: A Survey (texto local en `~/tesis/privado/papers/`).
- Schuchardt, J., Golkov, V., y Cremers, D. (2019). Learning to Evolve. https://arxiv.org/abs/1905.03389
- Schulman, J., Moritz, P., Levine, S., Jordan, M., y Abbeel, P. (2016). High-Dimensional Continuous Control Using
  Generalized Advantage Estimation. *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., y Klimov, O. (2017). Proximal Policy Optimization Algorithms.
  https://arxiv.org/abs/1707.06347
- Shala, G., Biedenkapp, A., Awad, N., Adriaensen, S., Lindauer, M., y Hutter, F. (2020). Learning Step-Size Adaptation in
  CMA-ES. *PPSN XVI*.
- Sharma, M., Komninos, A., López-Ibáñez, M., y Kazakov, D. (2019). Deep reinforcement learning based parameter control in
  differential evolution. *GECCO*.
- Thierens, D. (2005). An adaptive pursuit strategy for allocating operator probabilities. *GECCO*.

---

## 11. Versión 2: estado SNN, acción sobre P, menos decisiones (2026-10-15)

**Motivo:** la versión 1 (plan B, 800 y 1408 corridas) no se distinguió del estático en E1 ([17](experimentos/17_comparacion_e1.md)).
El análisis de las 70.400 decisiones del entrenamiento ([18](experimentos/18_analisis_efectos_acciones.md)) mostró que sí hay algo
que aprender, pero que el planteamiento lo tapaba:
- mezclar entornos con óptimos opuestos;
- intensidades sin efecto;
- decisiones con efecto diminuto;
- evaluación sin poder estadístico.

**Cambios** (decididos el 2026-10-15):

| Elemento | Versión 1 (plan B) | Versión 2 |
|---|---|---|
| Agente | Uno para acrobot-swingup + pendulum-meta (transferencia a bottleneck) | **Uno por entorno**; sin transferencia |
| Estado | Compacto, 9 rasgos | **SNN, 10 rasgos** (`--estado snn`): τ, V*, ΔV*, estancamiento, tamaño de la red y **5 de actividad de la SNN** del campeón (ocultas mudas, ocultas saturadas, entradas mudas, salidas mudas, acción saturada) |
| Acción | P (10) + intensidades (3) | **Solo P** sobre los 10 operadores (`--accion p`): las intensidades no tuvieron efecto |
| Decisiones por corrida | T = 50 (cada ~12 generaciones en acrobot-swingup) | **T = 10** (cada ~58): efecto por decisión ~5× mayor |
| Exploración (σ inicial) | 0,5 (empujón típico ×1,6) | **1,0** (×2,7): efecto ~2× más visible |
| Entrenamiento | Corridas independientes | **Gemelos con la misma semilla solo en acrobot-swingup** (ρ = 0,78); en pendulum-meta no ayudan (ρ = 0,26) |
| Evaluación | 11 semillas | 11 como mínimo, más las que pida el análisis de poder: ~7–20 en acrobot-swingup, ~280 en pendulum-meta |

**Actividad de la SNN** (`eval::ActividadSNN`, `resumirActividad`):
- se mide en el campeón de cada generación durante su validación, que ya se corría: no agrega evaluaciones;
- solo observa: el retorno y la evolución son idénticos con y sin medirla (tests `[snn]`);
- cada rasgo de actividad apunta a palancas del catálogo: neuronas mudas → sesgo y dinámica; entradas mudas → ganancia;
  salidas mudas o acción saturada → excitación hacia la salida.

**Plan con reglas fijadas antes de mirar:**
1. **Regla escrita a mano contra estático**, en semillas nuevas: acrobot-swingup 301–320 y pendulum-meta 301–600.
   - **acrobot-swingup:** P en 2 fases desde los efectos medidos, al inicio más pesos finos y dinámica y menos mutación global,
     poda y borrar conexión (`neat_control --p --p2 --p-switch`, sin código nuevo).
   - **pendulum-meta:** regla por estado (red grande o V* bajo → más sesgo y poda, menos mutación global y conexión).
   - En los dos, contra c0 (uniforme) y contra la P fija derivada de los efectos de toda la corrida.
   - **Regla de decisión:** si la regla gana a c0 (Wilcoxon pareado, p < 0,05), hay espacio confirmado con datos independientes.
     Si además gana a la P fija, el estado aporta más allá de una P mejor.
2. **PPO versión 2 por entorno**, primero acrobot-swingup, ≥ 1500 corridas. Validación pareada contra c0 y contra la regla.
3. **Comparación final** con el número de semillas del análisis de poder.

## 12. Versión 3: PPO con más información (plan, 2026-10-16)

**Diagnóstico** (fichas [20](experimentos/20_ppo_v2.md), [27](experimentos/27_ppo_v2b.md), [28](experimentos/28_fases_pmeta.md)):
- En pendulum-meta hay un objetivo grande y claro: mover P hacia mucho mutar pesos y casi nada de agregar nodo da +0,12 sobre
  c0 (192 semillas nuevas). AP no lo captura (≈ c0 con 300 semillas), porque su crédito inmediato castiga mutar pesos.
- PPO v2 (T = 10, σ 1) y v2b (T = 2, σ 1,5, lr 1e-3) no aprenden esa dirección en 800 corridas cada uno: la política cambia en
  cada actualización (KL 0,03–0,08) pero sin dirección. La señal por corrida es chica frente al ruido del resultado (±0,15).
- PPO residual en acrobot-swingup sí aprende correcciones consistentes: partir cerca de una buena política ayuda.

**Decisión:** se mantiene PPO (algoritmo de la propuesta). Se le da más información para que necesite menos corridas:

| Cambio | Qué aporta | Verificación |
|---|---|---|
| Preentrenar el crítico con las corridas exploratorias existentes (plan B, v2, v2b, sondas; miles de corridas con estado, acción y resultado final) | Ventajas con mucho menos ruido desde la primera iteración | Error del crítico en corridas de validación, antes y después |
| Más corridas por actualización (N = 32–48 en vez de 16) | Gradiente menos ruidoso por actualización | Mismo resultado con N = 16 (bit a bit) cuando no se usa |
| Recompensa con menos ruido (Φ con más episodios o el test final al cierre) | Menos varianza de la recompensa | Correlación de Φ con el test final |
| Residual sobre la mejor regla con estado | Parte desde una política buena; aprende correcciones | Ya verificado (ficha [23](experimentos/23_ppo_residual.md)) |

**Orden:** primero pendulum-meta (barato, objetivo conocido) para medir cuánto acelera cada cambio el aprendizaje de la dirección
del margen; luego acrobot-swingup; bottleneck al final. La evaluación sigue siendo pareada, en semillas nuevas y con reglas
fijadas antes de mirar.
