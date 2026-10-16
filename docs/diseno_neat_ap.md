# Diseño: NEAT estándar y NEAT + Adaptive Pursuit sobre `snn/`

> Documento de diseño, escrito por Claude el 2026-09-26. Decisiones §12 **confirmadas por el usuario** el mismo día;
> estado de la implementación en §14.
> Base: [`propuesta.md`](propuesta.md) (el informe de seminario), [`snn-simulator.md`](snn-simulator.md)
> y verificaciones hechas en mullo el mismo día.
> Alcance: las **dos líneas base** de la tesis, (1) NEAT con probabilidades estáticas y (2) NEAT con AOS
> sin estado (Adaptive Pursuit), dejando lista la interfaz donde después se conecta el agente PPO basado
> en estado.
> Convención: **[P]** propuesta · **[S]** simulador · **[V]** verificado en mullo · **[D]** decisión
> propuesta (a confirmar) · **[!]** riesgo.

---

## 0. Resumen

1. **Las tres configuraciones de la tesis deben diferir solo en quién fija el vector de probabilidades de los operadores.** Propongo un esquema único: en cada hijo se aplica **un** operador de mutación, sorteado con un vector **P** sobre K operadores.
   - **Estático:** P fijo.
   - **Adaptive Pursuit:** P se actualiza con el crédito de cada operador.
   - **PPO:** P es la salida continua del agente (softmax).

   Esto cumple literalmente lo que exige §3.4 ("aislar el mecanismo de selección") y encaja con la restricción de RLtools, cuyo PPO solo tiene acciones continuas (§4).
2. **El simulador aporta el fenotipo** (neurona Izhikevich, red, dinámica y velocidad). Falta todo lo evolutivo: genoma, innovaciones, especiación, reproducción y selector. Hay que escribir NEAT **desde cero, limpio**, usando el NEAT del paper solo como referencia (§6.1).
3. **Hay que corregir 3 cosas del simulador antes de evolucionar nada:**
   - pesos por sinapsis;
   - entrada como corriente constante;
   - estado inicial de la neurona: algunos tipos arrancan en su equilibrio inestable [V] (§3).
4. **El umbral de convergencia de la propuesta está mal definido para recompensas negativas:** "95 % del mejor fitness" empeora el umbral si el fitness es negativo. Hay que normalizar (§9.3).
5. **Evaluación con números aleatorios comunes:** mismas semillas de episodios para todos los genomas de una corrida, y conjunto de test separado para el fitness que se reporta. Así las diferencias padre–hijo que usa Adaptive Pursuit miden genotipo y no suerte (§8.3).

---

## 1. Lo que exige el informe para estas dos configuraciones

| Requisito [P] | Dónde | Implicación de diseño |
|---|---|---|
| Línea base 1: probabilidades de operador **fijas**, sintonizadas en Fase 4 desde los valores de Stanley & Miikkulainen (2002) | §3.4 | Necesito el catálogo **y** sus probabilidades de partida (§5.3) |
| Operadores adicionales: probabilidad del operador estándar "con el que compartan naturaleza" | §3.4 | Cada operador nuevo declara su "naturaleza" (paramétrico o topológico) |
| Línea base 2: AOS sin estado con **Adaptive Pursuit** (Thierens 2005), "representativo de esa familia y del mecanismo empleado por SNAP-NEAT" | §3.4 | Replicar AP fielmente, incluido cómo SNAP-NEAT asigna el crédito (§7) |
| **Mismo catálogo** de operadores en las tres configuraciones | §3.4 | Catálogo definido **antes** de implementar las líneas base (Fase 1.4) |
| AOS sobre operadores de **mutación topológica y paramétrica** | §3.3 | El cruce queda **fuera** del AOS: tasa fija e igual para todos |
| 11 réplicas, **mismas semillas**, **mismo presupuesto de evaluaciones de fitness** | §3.4 | Semillas por réplica compartidas entre configuraciones; contador de evaluaciones como criterio de parada |
| Métricas: fitness final, generaciones a umbral, neuronas y conexiones, varianza | §3.4 | Registro por generación con esas columnas (§9) |
| Tareas de control de complejidad moderada a moderada-alta; las elementales convergen muy rápido | §3.3 | Pilotos para calibrar la dificultad (§10) |
| Entornos: RLtools (decisión 2026-09-26) | §5.2 | Adaptador C++ entorno ↔ SNN (§8) |
| Pesos **por sinapsis**; se evolucionan pesos y topología (decisión 2026-09-26) | — | Modificar `Synapse` (§4) |

## 2. Qué ofrece el simulador para esto [S]

| Pieza | ¿Sirve? | Comentario |
|---|---|---|
| `IzhikevichNeuron` (Euler 2×dt/2, clamps, 25 tipos, conductancias exc/inh) | ✅ | Núcleo numérico validado por tests |
| `Network` (vectores, adyacencia, `step`, recurrencia con retardo de 1 ms, roles in/hidden/out) | ✅ | Estructura del fenotipo. Se **construye una por evaluación** desde el genoma |
| Velocidad | ✅ [V] | ~4,2 ms por episodio Acrobot completo (500 × 50 ms) con 5 neuronas y 1 hilo |
| `RLEncoder::SMALL/LARGE` | ✅ | Los encoders del paper, triviales de reusar |
| `RLDecoder` continuo | ⚠️ | Con 1 neurona, "silencio" = −1 (torque máximo negativo). Mejor un decoder diferencial (§8.2) |
| `Individual`/mutaciones de `Network` (insertNeuron, enable/disable…) | ❌ para NEAT | Orientadas a WANN; los IDs no son estables (swap-and-pop). Las mutaciones van en el **genoma** |
| `Simulator::simulateAll` (OpenMP) | ⚠️ | Paraleliza redes con **una** entrada fija. Para RL se paraleliza **por genoma** (episodios completos por hilo) |
| `Simulator::runEpisode` | ❌ | Roto: el estado no avanza. Escribimos nuestro evaluador |
| Pesos por sinapsis | ❌ | Hay que agregarlos |
| Semillas | ❌ | `random_device`. Hay que hacerlas deterministas |
| NEAT (genoma, innovación, especiación, cruce, reproducción) | ❌ | Todo por escribir |

**Conclusión:** del simulador usamos `snn_core` (neurona y red) y poco más. Lo nuevo es NEAT, los
selectores, el evaluador RL y la infraestructura de experimentos.

## 3. Hallazgo: estado inicial de la neurona [V]

`IzhikevichNeuron` inicia y reinicia en `v = c`, `u = b·c`. Con I = 0, el modelo tiene dos puntos fijos
(solución de `0.04v² + (5−b)v + 140 = 0`): uno **estable** (el reposo) y otro **inestable** (el umbral).
Para b = 0,2 son −70 y −50 mV.

Verificado en mullo (`build/checks/check_rest.cpp`: 500 ms sin entrada, y una perturbación de g = 0,5):

| Tipo | v inicial | Sin entrada | Con g = 0,5 | Lectura |
|---|---|---|---|---|
| RS, FS, TONIC_SPIKING (IZ-A), SFA (IZ-F)… | −65 | 0 spikes | 0 | OK: bajan al reposo (−70) |
| **TONIC_BURSTING (IZ-C)**, CHATTERING | **−50** | 0 (se queda en −50) | **1 spike** | Arranca en el equilibrio **inestable**: disparo "de gatillo fácil" |
| **PHASIC_BURSTING (IZ-D)** | −55 | 0 | **1 spike** | Cerca del inestable (≈ −54,3) |
| RESONATOR, REBOUND_BURST | −65 / −52 | **1 spike espontáneo** | 1 | Artefacto de la inicialización |
| BISTABILITY | −60 | **497 spikes** | 47 | Dispara siempre |
| ACCOMMODATION, INHIBITION_INDUCED_SPIKING/BURSTING | — | **49 / 49 / 231** | — | Disparan sin entrada |

Hay un agravante: el evaluador del paper y `Simulator::runSimulation` **reinician la red en cada paso
del entorno**, así que el artefacto se repite en cada ventana de 50 ms.

Además, algunos de los 19 patrones de Izhikevich (2004) se reproducen en el script original con
ecuaciones o corrientes base modificadas (p. ej. Class 1 excitable, acomodación, inhibition-induced).
El simulador usa la ecuación estándar para todos, así que **el nombre no garantiza el comportamiento**.
Esto hay que verificarlo contra el script original de Izhikevich.

**[D]**
- (a) Iniciar y reiniciar en el **reposo estable** `v* = [−(5−b) − √((5−b)² − 22.4)] / 0.08`, `u* = b·v*`, que existe si b ≤ ≈0,267 (da −70 para b = 0,2 y −87,5 para CLASS1, coherente con lo medido). Si no, usar `v = −65`, `u = b·v`, que es el estado del paper (ANNarchy: −65 y −14).
- (b) Para NEAT, **restringir los tipos permitidos** a los 8 del paper (IZ-A..H) o a un subconjunto que, tras (a), no dispare sin entrada.
- (c) Convertir `check_rest.cpp` en un test.

## 4. Cambios mínimos en `snn/` para esta etapa [D]

1. **Pesos por sinapsis:** `Synapse` recibe `double weight`, y `Network::step` usa `weight × gain` (gain global = 1). El signo lo decide `excitatory` y el peso es la magnitud (ver §6.2).
2. **Entrada como corriente constante:** `Network::setInputOffsets(currents)`, que fija `i_offset` en las neuronas de entrada. Es lo que hacía el paper (`pop[i].I = …`) y evita la conductancia acumulada ≈ 5,5·I de `setInputCurrents`.
3. **Estado inicial** según §3(a).
4. **Determinismo:** sin `random_device`; el ruido de la neurona queda en 0 y, si se usa, con semilla explícita.
5. Menores:
   - quitar la ambigüedad del constructor de `Simulator`;
   - revisar el test FS, que probablemente se arregla con (3);
   - agregar tests para los pesos y para el reinicio.

Todo es local y chico. No toca la estructura del simulador.

## 5. Decisión central: esquema de aplicación de operadores

### 5.1 El problema
NEAT "clásico" aplica los operadores como **Bernoulli independientes** sobre cada hijo: peso con p = 0,8,
nodo con 0,03, conexión con 0,05. Un hijo puede recibir varios operadores o ninguno. Adaptive Pursuit, en
cambio, está definido para **elegir un operador** de K según un vector P, y asigna crédito a ese operador
por el resultado. Con varios operadores sobre el mismo hijo, el crédito no se puede atribuir.

### 5.2 Opciones

| | A: categórico (1 operador por hijo) | B: Bernoulli independientes |
|---|---|---|
| Estático | P fijo sobre K operadores | p_i fijos (NEAT clásico literal) |
| Adaptive Pursuit | Natural (Thierens 2005) | No estándar: requiere adaptar AP y el crédito es ambiguo |
| PPO (futuro) | Salida continua → softmax → P ✅ encaja con RLtools | Salida continua → sigmoides → p_i ✅ |
| Crédito por operador | Limpio | Confundido |
| "Aislar el mecanismo de selección" (§3.4) | ✅ Idéntico en las 3 | ⚠️ AP necesita otro esquema de aplicación |

### 5.3 [D] Opción A
Todo hijo (de cruce o clon) recibe **exactamente un** operador ~ P. Las tres configuraciones comparten
todo lo demás.

- **Costo:** la línea base estática no es literalmente el NEAT de Stanley, porque ningún hijo recibe dos operadores.
- **Mitigación:** en los pilotos de la Fase 4, correr también el **NEAT canónico (opción B)** como referencia de cordura, para mostrar que el estático categórico no es una línea base debilitada. No entra en los contrastes finales.

### 5.4 Catálogo de operadores (Fase 1.4)

**Decisión (2026-09-26): catálogo inicial = NEAT clásico, O1+O2 como un solo operador y O3, O4**, verificado
en Stanley & Miikkulainen (2002):
- §3.1: pesos "perturbados o no" en cada generación; agregar conexión con peso aleatorio entre dos nodos no conectados; agregar nodo dividiendo una conexión (la vieja se deshabilita, la entrante recibe 1 y la saliente el peso viejo).
- §4.1: "80 % de probabilidad de mutar los pesos del genoma, y en ese caso cada peso con 90 % de perturbación uniforme y 10 % de valor nuevo"; agregar nodo 0,03; agregar conexión 0,05 (0,3 en la población grande).

P estático = (0,8 ; 0,05 ; 0,03) / 0,88 = **(0,909 ; 0,057 ; 0,034)**, implementado en `StaticSelector::stanley2002()`.
Separar O1/O2 y agregar O5/O6 queda para después de los pilotos. La tabla siguiente es la propuesta
ampliada original, como referencia:


| # | Operador | Naturaleza | Qué hace | p de Stanley | P estático inicial* |
|---|---|---|---|---|---|
| O1 | Perturbar pesos | Paramétrico fino | Cada conexión habilitada: w += N(0, σ) | 0,8 × 0,9 = 0,72 | 0,75 |
| O2 | Reasignar peso | Paramétrico grueso | Una (o pocas) conexiones reciben un valor nuevo al azar | 0,8 × 0,1 = 0,08 | 0,083 |
| O3 | Agregar conexión | Topológico + | Nueva sinapsis entre un nodo fuente y un nodo destino no conectados | 0,05 | 0,052 |
| O4 | Agregar neurona | Topológico + | Divide una conexión (entrante con peso 1, saliente con el peso original) | 0,03 | 0,031 |
| O5 | Deshabilitar conexión | Topológico − | Simplificación, estilo AGENT (Behjat 2022) | = O3 | 0,052 |
| O6 | Cambiar tipo de neurona | Paramétrico neuronal (SNN) | Oculta o de salida → otro tipo permitido (§3b) | = O4 | 0,031 |

\* Normalización de (0,72, 0,08, 0,05, 0,03, 0,05, 0,03) / 0,96. Las probabilidades de Stanley se
citan de memoria: **verificarlas en la §4 del paper original** (ahí la tasa de agregar conexión es mayor
en poblaciones grandes).

- **O1 y O2 separados:** dan al AOS un dilema real entre exploración gruesa y explotación fina de pesos. Si fueran un solo operador, esa decisión quedaría fija.
- **O5 y O6 son opcionales.** Hacen el catálogo más rico y más "SNN", pero cada operador extra diluye el crédito.
- **Recomendación:** empezar con **O1–O4** (NEAT puro) para validar, y decidir O5 y O6 después de los pilotos. **El catálogo se congela antes de los experimentos finales.**

## 6. NEAT estándar

### 6.1 ¿Portar el NEAT del paper o escribirlo limpio? → **Escribirlo limpio** [D]
El NEAT del paper no sirve como base:
- Embebe Python.
- Tiene bugs de evaluación y de reproducción (ver `comprension.md` §6.4).
- Muta a los **padres in situ** y no muta a los hijos de cruce.
- La mutación de pesos es un paso fijo ±`learningRate` sobre una sola conexión.
- Usa `rand()` sin control de semillas.
- No registra qué operador creó a cada hijo, que es imprescindible para el AOS.

Adaptarlo implicaría reescribir casi todo. Un NEAT canónico compacto (~1 000 líneas) es más seguro, y
el del paper sirve como referencia de parámetros.

### 6.2 Genoma y fenotipo
- **NodeGene:** `id`, `rol` (in / out / hidden), `tipo` (NeuronType o (a,b,c,d)).
- **ConnGene:** `innov`, `in`, `out`, `w` (con signo), `enabled`.
- **[D] Peso con signo:** `excitatory = (w ≥ 0)` y magnitud `|w|` en la sinapsis. O1 puede cruzar el cero y así cambiar la polaridad de forma natural, como en NEAT; no hace falta un operador de polaridad.
- **Fenotipo:** `genoma → Network`, reconstruida en cada evaluación: entradas en orden, salidas, ocultas y sinapsis habilitadas. El costo es despreciable frente a la simulación.
- **Topología [D]:**
  - destinos ∈ ocultas ∪ salidas (nada entra a las entradas);
  - fuentes ∈ entradas ∪ ocultas ∪ salidas;
  - **recurrencia permitida**, porque la SNN la maneja con un retardo de 1 ms;
  - sin autoconexiones, que el simulador rechaza.

### 6.3 Componentes (Stanley & Miikkulainen 2002)
- **Población inicial:** topología mínima (entradas → salidas totalmente conectadas), pesos al azar y tipo de neurona fijo.
- **Innovaciones:** número global. Las mutaciones estructurales idénticas dentro de la misma generación reciben el mismo número.
- **Distancia de compatibilidad:** δ = c1·E/N + c2·D/N + c3·W̄, más **c4·(fracción de nodos homólogos con distinto tipo)** si O6 está activo.
- **Especiación:** representante por especie y umbral δ_t (opcionalmente ajustado para mantener un número objetivo de especies).
- **[!] Fitness negativo:** Pendulum y Acrobot dan retornos negativos, y la cuota de hijos proporcional al fitness compartido requiere valores ≥ 0. Hay que usar f' = f − f_min (o rangos) **antes** de compartir. El paper no lo hacía.
- **Reproducción:**
  - campeón de cada especie con más de 5 miembros copiado sin cambios (élite, sin operador);
  - selección por truncamiento dentro de la especie;
  - cruce en una fracción fija de los hijos (tasa interespecies fija);
  - todo hijo no élite recibe **un** operador ~ P (§5.3).
- **Cruce:** los genes homólogos se heredan al azar; los disjuntos y en exceso, del padre más apto; un gen deshabilitado en algún padre queda deshabilitado con p = 0,75.
- **Estancamiento:** se eliminan las especies sin mejora en 15 generaciones, salvo la del campeón.

### 6.4 Punto de enganche: `OperatorSelector`
```cpp
struct GenerationReport {                 // lo que el selector ve al cerrar la generación t
    int generation; long evaluations_used;
    std::vector<OperatorOutcome> outcomes; // por hijo: operador, fitness del hijo, fitness de referencia del padre
    PopulationStats stats;                 // mejor, media, desv. est., n.º de especies, complejidad…
                                           // (el selector PPO lo usará como ESTADO)
};
class OperatorSelector {
public:
    virtual std::vector<double> probabilities() const = 0;   // P para la siguiente reproducción
    virtual void update(const GenerationReport&) = 0;        // Estático: no hace nada · AP: crédito
                                                             // PPO: estado y recompensa
    virtual ~OperatorSelector() = default;
};
```
El resto de NEAT no sabe qué selector usa. **Esa es la garantía de comparación justa.**

## 7. Adaptive Pursuit (Thierens 2005)

**Actualización**, con K operadores y P_min < 1/K, por ejemplo P_min = 1/(2K):
```
Q_i ← Q_i + α·(R_i − Q_i)                                    (estimación de calidad)
i* = argmax_i Q_i
P_i* ← P_i* + β·(P_max − P_i*),   P_max = 1 − (K−1)·P_min
P_j  ← P_j  + β·(P_min − P_j)      para j ≠ i*
```
- Inicio: P_i = 1/K. Hiperparámetros: α, β y P_min, con **esfuerzo de ajuste comparable** al que recibe el estático en la Fase 4, por justicia.
- Los valores típicos del paper (α, β ≈ 0,8) hay que **verificarlos en Thierens (2005)**.

**Asignación de crédito [D]**, que es donde se juega la calidad de esta línea base:
- **Recompensa por hijo:** `r = max(0, f_hijo − f_ref) / (f_max − f_min + ε)`.
  - `f_ref` = fitness del padre; en hijos de cruce, el mejor de los padres.
  - Se normaliza por el rango de la población, para que no dependa de la escala de cada entorno.
- **Agregado por operador y generación:** R_i = media de los r de los hijos que produjo i. Variante: máximo, el "crédito extremo" de la literatura AOS. Si un operador no se usó, su Q no se actualiza.
- **Frecuencia:** una actualización de AP **por generación** (NEAT es generacional). Con ~150 hijos por generación, cada operador recibe como mínimo ~150·P_min muestras.
- **Crédito retardado:** SNAP-NEAT advierte que las mutaciones estructurales rinden recién después de ajustar sus pesos, y lo mitiga con crédito continuo y un periodo inicial uniforme.
  - **Hay que leer Kohl & Miikkulainen (2012)** y replicar su esquema, porque la propuesta presenta a AP como su representante.
  - Variante implementable: **crédito por linaje**, que acredita al operador que creó al ancestro cuando un descendiente mejora dentro de k generaciones.
- **Ruido del fitness:** usar números aleatorios comunes (§8.3). Si no, `f_hijo − f_ref` mezcla calidad con suerte de episodios, y AP persigue ruido.

**Registro:** P, Q, usos y éxitos por operador en cada generación. Las trayectorias de P son una figura
natural de la tesis y sirven para comparar después con lo que decide PPO.

## 8. Evaluación del fitness (común a todas las configuraciones)

### 8.1 Entorno y codificación
- Adaptador por entorno RLtools: `reset(seed)`, `observe`, `step(action)`, `reward` y `terminated`.
- **Normalización de la observación** a [−1, 1] con cotas conocidas: Pendulum θ̇ ∈ [−8, 8]; Acrobot θ̇₁ ∈ ±4π, θ̇₂ ∈ ±9π.
- **Codificación SMALL:** 2 neuronas por variable, como en el paper, con **corriente constante** `i_offset = |x|·G_in` (§4.2).
  - `G_in` es la ganancia de entrada, un hiperparámetro fijo por entorno calibrado para que las entradas cubran su rango de disparo.
  - [D] No se re-muestrea por evaluación: el paper lo hacía y metía ruido.

### 8.2 Decodificación [D]
- **Decoder diferencial:** 2 neuronas de salida por dimensión de acción, con `a = clamp((n⁺ − n⁻)/n_max, −1, 1)`, donde n = spikes en la ventana. El silencio da a = 0 (torque nulo), no −1.
- Ventana de **50 ms** por paso (como el paper), con la red reiniciada al inicio de cada paso.
  - Alternativa a estudiar: red *con estado* entre pasos (sin reinicio), que da memoria temporal. Cambia la naturaleza del controlador, así que la dejaría como experimento secundario.

### 8.3 Semillas, ruido y presupuesto
- **Semilla maestra por réplica r** (r = 0..10), **la misma para las 3 configuraciones**. Así comparten población inicial y episodios, como pide la propuesta.
- **Flujos de números aleatorios separados:**
  - evolución: un generador por corrida;
  - evaluación: semilla derivada de hash(r, episodio). No depende del hilo, así que el resultado es **idéntico con cualquier número de hilos**.
- **Números aleatorios comunes (CRN):** todos los genomas de una corrida se evalúan en el **mismo conjunto de E estados iniciales de entrenamiento**. El fitness es determinista y las comparaciones padre–hijo son exactas.
- **Test separado:** el campeón de cada generación se evalúa en M episodios **no vistos**. Este es el fitness que se **reporta** y con el que se mide el umbral. Esas evaluaciones de medición no se descuentan del presupuesto (hay que declararlo en la memoria).
- **Presupuesto** = evaluaciones de genoma (población × generaciones), igual para las 3 configuraciones.
- **Paralelismo:** `omp parallel for` sobre genomas. Cada hilo construye su `Network` y corre sus E episodios. En mullo, con hilos acotados por proceso, las 33 corridas por entorno se reparten como procesos.
- **Costo, orden de magnitud:**
  - ~5–15 ms por episodio Acrobot con redes de 10–20 neuronas;
  - E = 10 → ~0,1 s por genoma → 150 genomas ≈ 15 s por generación en serie;
  - 500 generaciones ≈ 2 h por corrida en serie; con 16 hilos, unos minutos;
  - 33 corridas × varios entornos es perfectamente factible.
  - **Medirlo en el piloto.**

## 9. Métricas y registro

### 9.1 Por generación (CSV por corrida)
- **Progreso:** generación, evaluaciones usadas.
- **Fitness de entrenamiento:** mejor, media y desviación estándar.
- **Fitness de test del campeón.**
- **Especies:** número de especies.
- **Complejidad:** neuronas ocultas y conexiones habilitadas del campeón, y la media de la población.
- **Operadores:** P_i y, en AP, Q_i; usos_i y éxitos_i.

### 9.2 Al final
- Campeón serializado (JSON).
- Fitness de test final con muchos episodios.
- Configuración completa y semilla.

Todo en `~/tesis/runs/<experimento>/<config>/<réplica>/`.

### 9.3 [!] Umbral de eficiencia con recompensas negativas
La propuesta fija el umbral en el "95 % del mejor fitness de los preliminares". Con fitness negativo esto
es incorrecto: el 95 % de −150 es −142,5, un umbral **más exigente** que el mejor observado.

**[D]** Definirlo normalizado:
```
umbral = f_ref + 0.95·(f_best − f_ref)
```
donde `f_ref` es el fitness de una política de referencia (aleatoria o acción nula). Si el entorno tiene
criterio de éxito propio (Acrobot: alcanzar la meta), se usa ese. Hay que dejarlo escrito antes de las
réplicas finales, como exige §3.4.

## 10. Entornos RLtools y complejidad incremental

En el commit fijado hay: `pendulum`, `acrobot`, `car` (pista), `l2f` (cuadricóptero), `flag`, `memory`,
`mujoco` y `multi_agent`.

| Nivel | Entorno | Obs → entradas SMALL | Acción | Nota |
|---|---|---|---|---|
| 1 | Pendulum (≈ Pendulum-v1: torque ±2, 200 pasos, sin término anticipado) | 3 → 6 | 1 continua | No tiene criterio de éxito → umbral normalizado (§9.3) |
| 2 | Acrobot (torque continuo; **±5 por defecto**, parámetro ajustable) | 6 → 12 | 1 continua | Con ±5 es probablemente fácil. **Bajar el torque** es una perilla limpia de dificultad |
| 3 | Acrobot swing-up / car / l2f | más | más | Candidatos para "moderada-alta"; l2f probablemente demasiado difícil para NEAT+SNN |

**[D]** Calibrar con pilotos de NEAT estático. Si un entorno converge en muy pocas generaciones, el AOS
no tiene horizonte (§3.3): se sube la dificultad o se descarta.

## 11. Plan de implementación (hitos con criterio de aceptación)

| Hito | Qué | Aceptación |
|---|---|---|
| **M0** | `snn/`: pesos por sinapsis, `setInputOffsets`, estado inicial, determinismo, fixes menores | Tests nuevos verdes; `check_rest` sin disparos espontáneos en los tipos permitidos |
| **M1** | `src/envs` + `src/eval`: adaptadores Pendulum y Acrobot, SMALL y decoder diferencial, bucle de episodio, evaluación paralela con semillas | Red hecha a mano da el mismo fitness con 1 y con N hilos; costo por episodio medido |
| **M2** | `src/neat`: genoma, innovaciones, cruce, compatibilidad, especiación, reproducción; `StaticSelector` | Tests de invariantes (cruce, innovaciones repetidas, conteo de población); NEAT estático **aprende** en Pendulum |
| **M3** | `src/experiment`: ejecutable, configuración, CSV por generación, test del campeón, presupuesto | Corrida completa reproducible bit a bit con la misma semilla |
| **M4** | `AdaptivePursuitSelector` y crédito (§7) | **P se mueve** de forma consistente entre réplicas. Si no se mueve, el catálogo o el crédito no informan: revisar antes de seguir |
| **M5** | Pilotos: calibración de entornos (§10), NEAT canónico de referencia (§5.3), costo real | Entornos elegidos; umbral definido (§9.3) |
| **M6** | `analysis/`: métricas y tests de §3.4 (MWU, Brown-Forsythe, log-rank, Holm, Â12) | Reproduce las tablas y figuras a partir de `runs/` |

**Dependencia externa:** M1 necesita RLtools en `external/rl-tools` (pendiente, lo agrega el usuario).

## 12. Decisiones (confirmadas por el usuario el 2026-09-26)

Confirmadas: (1) esquema categórico; (2) catálogo = NEAT clásico (3 operadores, §5.4); (3) neurona IZ-A
(TONIC_SPIKING); (4) corrección del estado inicial y entrada como corriente constante (hechas en M0).
Pendientes: (5) CRN con test separado se implementa en M1; (6) umbral normalizado, a conversar con el
profesor guía (no se abordó en la reunión 2, §15); (7) papers de Thierens (2005) y Kohl & Miikkulainen (2012) para AP.

Texto original de la consulta:

1. **Esquema categórico, un operador por hijo (§5.3)**, con NEAT canónico solo como referencia en los pilotos.
2. **Catálogo inicial O1–O4**, con O5 (deshabilitar conexión) y O6 (tipo de neurona) a decidir tras los pilotos.
3. **Tipo de neurona** inicial fijo. ¿Cuál: IZ-A (tonic spiking) como base?
4. **Corrección del estado inicial** de la neurona (§3a) y **entrada como corriente constante** (§4.2).
5. **CRN con conjunto de test separado** (§8.3).
6. **Umbral normalizado** (§9.3). Conviene comentarlo con el profesor guía, porque modifica lo escrito en el informe.
7. Leer Kohl & Miikkulainen (2012) y Thierens (2005) para fijar los detalles de AP y del crédito. ¿Tienes los PDF?

## 13. Riesgos de esta etapa

- **La SNN no dispara:** con topología mínima y pesos al azar, las salidas pueden quedar mudas y el fitness plano al inicio. Mitigación: calibrar `G_in` y el rango inicial de pesos para que haya actividad; el decoder diferencial evita el sesgo.
- **AP sin señal:** si el crédito es mayormente ruido o casi todos los hijos empeoran, P no se mueve. Mitigación: CRN, crédito normalizado, variante de crédito extremo o por linaje.
- **Entornos demasiado fáciles o difíciles** (§10).
- **Línea base estática muy fuerte:** si se ajusta por entorno y AP no, AP pierde por falta de ajuste y no por diseño. Hay que dar un esfuerzo de ajuste equivalente.

## 14. Estado de la implementación (2026-09-26)

**M0 — cambios en `snn/` (hecho):**
- Pesos por sinapsis: `Synapse::weight`; `Network::step()` sin peso compartido; `addSynapse(pre, post, exc, w)`.
- Se eliminó todo lo WANN: peso compartido, `Individual`, `findBestWeight` y `evaluateMultipleWeights`.
- `Network::setInputOffsets`: entrada como corriente constante (como ANNarchy). `SimulationConfig::constant_input_current` para el `Simulator`.
- Reposo estable al iniciar y reiniciar la neurona (§3): de los 25 tipos, 23 quedan en silencio sin entrada (antes 18). BISTABILITY y ACCOMMODATION no tienen reposo (b > 0,267) y no deben usarse.
- Determinismo: sin `random_device`; `IzhikevichNeuron::seedNoise`; encoders con semilla fija por defecto.
- Fixes: `runEpisode` (el estado ahora avanza), constructor ambiguo de `Simulator`, decoder duplicado unificado. El test heredado de FS ahora pasa (su fallo era el estado inicial).

**M2 — núcleo NEAT en `src/neat/` (hecho, sin entorno RL todavía):**
- `Genome`: topología mínima, cruce, compatibilidad, los 3 operadores clásicos y `buildNetwork()`, que genera el fenotipo en `snn`.
- `InnovationTracker`: marcas históricas por generación.
- `Population`: especiación, fitness compartido con desplazamiento para retornos negativos, estancamiento por especie y global, élite, truncamiento, cruce e interespecies, **un operador por hijo ~ P** y `Origin` para el crédito.
- `OperatorSelector` y `StaticSelector::stanley2002()`.

**Adaptaciones [SNN] (no están en el paper; a calibrar en los pilotos):**

| Parámetro | Valor | Motivo |
|---|---|---|
| `add_node_in_weight` | 30 | Con peso 1 el nodo nuevo queda mudo y corta la ruta; con ~30 retransmite 1:1 (test en `snn/tests`) |
| `compat_weight_scale` | 10 | Los pesos son conductancias (~10–50), no de orden 1. Sin reescalar, en la generación 4 había 37 especies en 150 genomas |
| `weight_init_range` / `weight_perturb_power` / `weight_max` | 30 / 5 / 60 | Rango útil de conductancias para TONIC_SPIKING |

**Verificación: NEAT + SNN en XOR** (`build/bin/xor_snn`, análogo a §4.2 del paper):
- **20/20 corridas resueltas**, 11,4 generaciones en promedio, 2,25 ocultas y 7,55 conexiones.
- Paper: 100/100, 32 generaciones, **2,35 ocultas y 7,48 conexiones**. La complejidad coincide.
- Las generaciones no son comparables: aquí (0,0)→0 sale gratis, porque sin entrada no hay spikes y no hay neurona de sesgo, y el criterio "dispara o no" es más laxo que el error continuo.
- Las 20 corridas tardan 0,15 s con 16 hilos.

**Tests:** 45/45 (simulador y NEAT), también bajo AddressSanitizer y UBSan. Un bug encontrado así:
especies movidas dentro del bucle de reproducción, que rompía el cruce interespecies.
Ahora hay un test que lo cubre.

**M4 — Adaptive Pursuit en `src/aos/` (hecho):** `AdaptivePursuitSelector` (Thierens 2005) con crédito
por hijo `r = max(0, f_hijo − f_padre) / (f_mejor − f_peor)`, media o máximo por operador y P inicial
= P de Stanley. `aos::makeSelector("static" | "ap")`. Tests: convergencia a P_max/P_min, adaptación
no estacionaria, crédito normalizado e integración con NEAT.

**M1 — entornos y evaluación (hecho):**
- `external/rl-tools` clonado en `b32d998` (falta que el usuario lo registre como submódulo).
- `src/envs`: Pendulum y Acrobot de RLtools tras la interfaz `ControlEnv`; observación normalizada; `torque_scale` escala la acción.
- `src/eval`: codificación SMALL con corriente constante (`input_gain` 40, según la curva f-I medida), decoder diferencial (2 salidas por acción) y 50 ms por paso; semillas deterministas por stream (entrenamiento, test y test final); evaluación paralela **idéntica con 1 o N hilos** (test).
- `neat_control`: ejecutable de experimentos con CSV por generación (fitness, test del campeón, especies, complejidad, P, usos y tasa de mejora por operador) y `summary.txt`.
- Tests: 55/55.

### 14.1 Pilotos (2026-09-26, `~/tesis/runs/piloto/`)

**Costo:** Pendulum, 150 genomas × 5 episodios, **0,14 s por generación** con 16 hilos. Acrobot ±1, ~0,3 s.
Una corrida de 150 generaciones tarda menos de 1 minuto: el cómputo no es un cuello de botella.

**XOR (50 corridas por configuración):**
- Estático: 11,4 generaciones, 2,32 ocultas.
- AP con α = β = 0,8: 23,3 generaciones, 9,12 ocultas.
- AP con α = 0,3, β = 0,1, crédito extremo: 13,8 generaciones, 2,50 ocultas.
- Con actualización **por generación**, β = 0,8 es demasiado codicioso: P cambia de operador dominante de una generación a otra. **β y el tipo de crédito son hiperparámetros críticos de AP.**

**Pendulum, 150 generaciones, 3 semillas:**

| | Entrenamiento | Test final (100 ep.) | Mejor red |
|---|---|---|---|
| Estático | −429 / −256 / −591 | −723 / −369 / −669 | 4–7 ocultas, 18–29 conex. |
| AP (α = 0,3, β = 0,1) | −215 / −747 / −643 | −408 / −840 / −783 | 39–63 ocultas, 104–136 conex. |

1. **Sobreajuste a los E = 5 estados iniciales de entrenamiento** (gran brecha entre entrenamiento y test). Hay que subir E (10–20) o rotar el conjunto de entrenamiento.
2. **AP con crédito "mejora sobre el padre" infla las redes.**
   - "Mutar pesos" es disruptivo (perturba todos los pesos, con 10 % de reasignación por peso) y mejora al padre en 0–5 % de los hijos, aunque es el motor del progreso en el estático (~134 de 150 hijos).
   - "Agregar nodo" es casi neutro (el relevo 1:1) y mejora en ~10 %.
   - AP persigue lo neutro: P llega a (0,02 ; 0,13 ; 0,85), o a 96 % "agregar conexión" y estancamiento.

   Es un caso concreto de la **ceguera de los métodos sin estado** que motiva la tesis, y también un problema de diseño del crédito:
   - penalizar complejidad;
   - dar crédito por posición en la población y no por mejora sobre el padre;
   - no recortar las pérdidas a 0;
   - crédito diferido, como en SNAP-NEAT.
3. **Especiación mal calibrada.** En Pendulum las especies oscilan entre 1 y ~14; en Acrobot quedan **siempre en 1**. Con más de 20 genes, N normaliza y las diferencias estructurales pesan poco. Opciones: bajar δt, cambiar `compat_weight_scale` o usar un **umbral dinámico** con número objetivo de especies (práctica común en implementaciones de NEAT posteriores al paper).

**Acrobot, estático, 40 generaciones:**
- Torque ±5 (RLtools por defecto): la población inicial ya da −31,6. **Trivial, no sirve.**
- Torque ±1 (`--torque-scale 0.2`, como gymnasium): de −394 a −115 en entrenamiento y −152 en test, con aprendizaje gradual. **Candidato para "moderada".**

**Siguiente:**
- Calibración (especiación, E, disrupción del operador de pesos).
- Rediseño o estudio del crédito de AP, con los papers de Thierens 2005 y Kohl & Miikkulainen 2012.
- Scripts de lanzamiento de réplicas y el análisis (M5 y M6).

### 14.2 Calibración de NEAT estático (2026-09-27, `~/tesis/runs/calib_*`, 5 semillas, 150 generaciones)

Herramientas: `scripts/run_grid.py` (variantes × semillas desde `configs/*.json`, reanudable) y
`analysis/summarize.py`. **Los valores del código siguen siendo los de Stanley; los calibrados viven
en los JSON de `configs/`.**

**1. Episodios de entrenamiento E** (Pendulum; test final con 100 episodios):

| E | Test final | Brecha entrenamiento−test | s/corrida (8 hilos) |
|---|---|---|---|
| 5 | −731 ± 257 | 210 | 43 |
| 10 | −641 ± 103 | 117 | 89 |
| 20 | −676 ± 126 | 72 | 164 |

→ **E = 10.**

**2–3. Operador de pesos y especiación** (Pendulum, E = 10):

| Variante | Test final (mediana) | Especies |
|---|---|---|
| δt = 3 fijo, reasignación 0,10 (Stanley) | −641 (−620) | 2,4 |
| 10 especies objetivo, reasignación 0,10 | −717 (−782) | 9,0 |
| δt fijo, reasignación 0,02 | **−368 (−316)** | 2,2 |
| 10 especies objetivo, reasignación 0,02 | −403 (**−264**) | 9,0 |
| 10 especies objetivo, reasignación 0 | −431 (−387) | 8,9 |
| 10 especies objetivo, reasignación 0,05 | −656 (−734) | 9,0 |
| Potencia de perturbación 2,5 / 10 (con 0,10) | −783 / −711 | 9 |

→ **Reasignación 0,02** (en vez del 0,1 de Stanley): con 10 % de reasignación por peso, el operador es demasiado destructivo para estas redes.
→ Especiación: sin diferencia clara con n = 5. Con δt fijo casi no hay especies (~2).
→ **Reproducibilidad verificada:** la misma variante corrida en dos lanzamientos distintos dio resultados idénticos (−403,0 ± 224,0).

**4. Acrobot ±1** (100 generaciones, reasignación 0,02):
- δt fijo (siempre 1 especie): −104 ± 8.
- 10 especies objetivo: −115 ± 11.
- La especiación cuesta algo de velocidad en este entorno.

**Decisión pendiente del usuario:** δt fijo (fiel al paper, pero en la práctica sin especiación) o dinámico con 10 especies objetivo (NEAT con especiación real; recomendado).

## 15. Segunda reunión con el profesor guía (registrada el 2026-10-03)

Se presentaron los resultados de `~/tesis/runs/reunion2_*` (§14) y la página explicativa.

| # | Tema | Resultado |
|---|---|---|
| 1 | ¿Mutación de pesos dentro del catálogo adaptativo? | **[D] Sí.** Los pesos son un operador más del catálogo que decide el AOS. El catálogo actual (pesos, agregar conexión, agregar nodo) se mantiene |
| 2 | Definición del crédito | **[D] Exploración sistemática** de opciones, guiada por la taxonomía de Pei et al. (2025) (§15.1) |
| 3 | Umbral de eficiencia normalizado (§9.3) | No se abordó. **Pendiente** |
| 4 | Elección de entornos | **[D] Tarea prioritaria.** Comparar NEAT + SNN en **todos** los entornos disponibles en RLtools (no solo los del preprint). La elección del entorno de la tesis debe salir de esa comparación de complejidad, y será una sección de los primeros capítulos (§15.2) |
| 5 | Desvíos del NEAT original (peso 30 al nodo nuevo, reasignación 0,02, especiación dinámica) | No se abordó. **Pendiente** |

### 15.1 Taxonomía de AOS (Pei et al., 2025) y dónde están nuestras variantes

Texto extraído en `~/tesis/privado/papers/Adaptive_Operator_Selection_for_Meta-Heuristics_A_Survey.txt`.

**Sin estado** = asignación de crédito (CA) + regla de selección (OSR). Ambos componentes son ortogonales y se combinan libremente.
- **CA:**
  - **basado en evaluación cruda:** el fitness del hijo. Premia aunque no haya mejora;
  - **basado en mejora:** FI, FI reescalado, FI con ventana (media o extremo), FIR (relativo al padre), FRR, éxito, tasa de éxito;
  - **basado en rango:** suma de rangos, AUC por fitness, supervivencia y tasa de supervivencia (fracción de hijos del operador que sobreviven a la selección);
  - **basado en potencial:** diversidad, *compass* (diversidad + fitness), métricas de paisaje de fitness (FLA).
- **OSR:** periódica, ruleta, Probability Matching, **Adaptive Pursuit**, bandidos (DMAB, Sliding MAB, UCB-Tuned, Thompson) y subpoblaciones.

**Con estado** = CA + extracción de características + estrategia de selección (modelo, entrenamiento y decisión), con entrenamiento **en línea** o **fuera de línea**.
- **Características:** de la solución (diversidad, estadísticas de fitness), del problema y del proceso (último operador, último crédito, estancamiento, presupuesto consumido).
- **Modelos:** PPO aparece solo en trabajos fuera de línea ([182], [197], [202]–[204]).
- **Guía §VI-C del survey:**
  - Fuera de línea si hay recursos y problemas de entrenamiento representativos, con FLA + características del proceso como estado y FIR o éxito como crédito.
  - Si no los hay: AP o DMAB con FIR o éxito.

Dónde caen nuestras variantes:

| Variante (§14) | CA según Pei | OSR | Observación |
|---|---|---|---|
| AP extremo | FI reescalado por el rango de la población, **extremo** por generación | AP | Converge a "mutar pesos" |
| AP SNAP | **Evaluación cruda** (fitness normalizado del individuo según su último operador) | AP secuencial | Infla las redes. Coincide con la crítica del survey: premia sin mejora |

Ejes a explorar para el crédito:
- **CA:** FI medio y extremo, FIR, éxito y tasa de éxito, rango (AUC), **tasa de supervivencia** (natural en NEAT: sobrevivir al corte de la especie), potencial (diversidad genómica o compass).
- **OSR:** PM, AP, DMAB y UCB.
- **Lo mismo vale para PPO:** la recompensa del MDP es un CA, de modo que esta exploración también alimenta el diseño de la Fase 1. La decisión entre entrenamiento en línea y fuera de línea sigue pendiente.

### 15.2 Selección de entornos (tarea en curso)

Inventario completo en `docs/entornos_rltools.md`.

Plan:
1. Inventario de todos los entornos de RLtools: dimensiones, tipo de acción, horizonte, recompensa, dependencias.
2. Adaptadores en `src/envs` con tests.
3. NEAT estático con un presupuesto fijo e igual para todos los entornos. Primero un barrido corto para filtrar, luego uno largo con los candidatos.
4. Criterios de complejidad fijados **antes** de correr:
   - brecha respecto de una referencia (aleatoria; PPO u óptimo si hay);
   - generación de estancamiento;
   - varianza entre semillas;
   - tamaño de la red;
   - costo por evaluación.
5. Tabla y figuras para el capítulo de la tesis.

### 15.3 Catálogo para la fase inicial (2026-10-03)

**[D]** Por ahora el catálogo queda con los 3 operadores de mutación de Stanley (pesos, agregar conexión,
agregar nodo). El cruce sigue fijo, con las tasas del paper, y la especiación es un mecanismo de la
población que puede entrar al **estado** del agente, no a la acción. Así lo acota la propuesta: "operadores
de mutación topológica y paramétrica".

El usuario explorará posibles operadores adicionales la semana del 2026-10-05. Candidatos (§5.4):
- separar perturbar pesos de reasignar pesos;
- deshabilitar conexión;
- cambiar tipo de neurona.

El cruce como acción del agente (tasa continua, vía PPO) queda como posible extensión o ablación del paper.
Se plantea al profesor: "¿ampliamos el catálogo de mutaciones o nos quedamos con los 3 de Stanley?".

### 15.4 Fase de comparación de entornos con referencia RL (plan, 2026-10-04)

**Objetivo:** identificar los entornos donde NEAT **aprende pero queda lejos** de un buen desempeño. Ese es el espacio donde el AOS tiene sentido. El análisis de estado (sondas de P fija, no estacionariedad, dependencia de rasgos del proceso) queda para una fase posterior, solo sobre los entornos elegidos.

Punto de partida: los barridos A y B (165 corridas, NEAT estático, 5 semillas × 500 generaciones; resultados en `entornos_rltools.md` §7).

1. **Lista a comparar (~16 entornos).**
   - Candidatos ("NEAT lucha"): pendulum-multitask, pendulum-meta (reactivo), pendulum-position y pendulum-velocity (memoria), memory (memoria), bottleneck, acrobot-swingup, ant, hyperdrone-moving-gate (reactivo), l2f, l2f-tiny.
   - Anclas ("NEAT resuelve"): pendulum, acrobot (±1), car, car-track, reacher.
   - Fuera: los visuales que no superan el retorno trivial (reacher-visual\*, l2f-visual, hyperdrone-target-frame) y flag (sobreajuste total). Se documentan como fuera del alcance de NEAT con píxeles como entradas directas.
2. **Referencia RL por entorno**, en tres niveles:
   - **zoo de RLtools** con entrenadores ajustados: pendulum, pendulum-multitask, pendulum-position, acrobot-swingup, ant, bottleneck, reacher, l2f, flag;
   - **configuración propia** copiando la del zoo más cercana y cambiando solo el entorno: car, car-track, acrobot, pendulum-meta y pendulum-velocity (con memoria → PPO-GRU, como en flag), moving-gate;
   - **óptimo analítico** donde exista (memory).

   Se usan 3–5 semillas, con los hiperparámetros del zoo y sin ajustarlos a favor ni en contra.
3. **Comparación justa:** misma recompensa, largo de episodio, parámetros y **semillas de test**. Se exporta la política RL y se evalúa con **nuestro** evaluador sobre los mismos 100 episodios finales que NEAT. Control negativo: una política aleatoria debe reproducir el retorno trivial del probe.
4. **Score normalizado** `(NEAT − trivial) / (RL − trivial)` con su intervalo entre semillas:
   - ≥ ~0.8 → NEAT resuelve (ancla);
   - ~0.2–0.8 → **objeto de estudio del AOS**;
   - ≤ ~0.2 → fuera de alcance.
5. **Sobreajuste:** NEAT entrena siempre con los mismos 10 episodios. Hay que decidir (usuario y profesor) entre mantenerlo y filtrar por brecha train→test, o rotar las semillas y reejecutar NEAT solo en la lista final.
6. **Selección y documentación:** tabla y figura (trivial, NEAT, RL, score, brecha, costo) para el capítulo y el paper. Se eligen 4–6 entornos: 1–2 anclas y 3–4 donde "NEAT lucha", cubriendo memoria, multitarea y multiagente si es posible.

Recursos: el nodo está libre (2026-10-04), así que se puede usar al 100 %. Mismo protocolo de prudencia: candado, binario congelado y control negativo.

#### 15.4.1 Avance y decisiones (2026-10-04)

**Referencia RL.**
- Tubería, verificaciones y resultados en `entornos_rltools.md` §8.
- **v2** (OpenBLAS, 10 semillas) reemplaza a v1, que no tenía BLAS y quedó interrumpida. El backend genérico es de un hilo y unas 15× más lento en SAC.

**moving-gate (decidido con tope de tiempo): sin referencia RL.** RLtools solo entrena hyperdrone en GPU (`src/rl/environments/l2f_visual/training_cuda_hyperdrone.cu`, tarea target-frame). mullo no tiene GPU, y un PPO con raytracing en CPU por millones de pasos excede el tope. NEAT lo corre igual en el barrido largo.

**Sobreajuste (paso 5): se miden ambos protocolos antes de decidir.**
- Recomendación de Claude, en la que el usuario delegó la opinión: rotar los episodios de entrenamiento en cada generación, comunes a toda la población. Se elimina el sobreajuste a 10 episodios fijos sin perder la comparación justa dentro de la generación.
- Implementado como `neat_control --resample-train`:
  - episodios del stream 1000 + gen;
  - genoma final = campeón de generación con mejor test de validación (stream 1);
  - el test final (stream 2) no interviene en la elección.
- Sin la opción, la salida es idéntica bit a bit a la del binario del barrido A (`scripts/equivalencia_binarios.sh`).
- Efecto en el crédito: con rotación, "hijo mejor que el padre" compara fitness medidos en episodios distintos. Es más ruidoso, y es parte del problema que el AOS debe manejar.

**Barrido largo** (`configs/barrido_largo.json`, `scripts/lote_barrido_largo.sh`):
- Diseño: 23 entornos × {`_fijo`, `_rot`} × 10 semillas × 2000 generaciones = 460 corridas.
- **ant va aparte, con 1000 generaciones** (`configs/barrido_largo_ant.json`, `scripts/lote_barrido_largo_ant.sh`, 10 hilos por corrida):
  - con el nodo lleno medía 150 s por generación (34 s en el barrido A);
  - a 2000 generaciones tardaría 4–5 días;
  - debe estar listo para la reunión del martes 2026-10-06 a las 14:00 (decisión del usuario, 2026-10-04).
- **Tope de 256 hilos desde el 2026-10-04 ~17:30** (entra otro usuario). Prioridad: semillas 1–5 para la reunión (resto 36 × 4 hilos; ant 10 × 10 hilos); las semillas 6–10 se corren después con los mismos lanzadores. Las corridas en curso de las semillas 6–10 y las del resto se detuvieron (≈ 1,5 h de avance; ninguna había terminado) y se relanzaron en este orden.
- Entornos: los que pasaron el filtro A/B, más l2f, l2f-tiny y flag (baratos; la rotación podría rescatar a flag). Los visuales quedan fuera salvo moving-gate.
- Preguntas que responde:
  1. ¿NEAT cierra la brecha con más presupuesto? La curva da los cortes de 500, 1000 y 2000.
  2. ¿Cuánto del mal desempeño es sobreajuste?

## 16. Entornos elegidos y sonda de P (2026-10-07)

**Barrido largo terminado** (2026-10-07 18:15): 440 corridas + ant 20, 10 semillas, sin errores. Referencia RL y
trivial reevaluados en los episodios de las semillas 1–10 (`~/tesis/runs/ref_rl/score_v2_s1-10.json`; cambio < 2 %,
mismo mejor entrenador). Ningún entorno cambia de grupo respecto de 5 semillas. Página de resultados:
`~/tesis/runs/reunion3/generar_pagina.py`.

**[D] Entornos de trabajo (usuario, 2026-10-07): bottleneck, acrobot-swingup y l2f.**
- Sin memoria (`--keep-state` queda fuera: empeora a NEAT en varios péndulos y agrega una variable de diseño).
- Forman una escalera de complejidad de control: 6 obs / 1 agente → 12 obs / 5 agentes → 82 obs / 4 rotores.
- l2f está hoy fuera de alcance (score 0,03 fijo / 0,09 rot): necesita una etapa de viabilidad (sin historial de
  acciones, ganancia de entrada y ventana, hiperparámetros, poda). Criterio: NEAT estático ≥ ~0,2. Respaldo:
  pendulum-meta reactivo.

**Riesgo detectado.** Con la métrica "fracción de hijos que mejoran al padre", agregar nodo gana en todas las fases
en los 3 entornos. La métrica favorece a operadores casi neutros (agregar nodo) frente a uno muy disruptivo (mutar
pesos perturba todos los pesos), así que no prueba que solo importe agregar nodos; pero tampoco muestra cambio de
fase. Se decide medirlo directamente.

**Sonda de P** (`configs/sonda_p.json`, `scripts/lote_sonda_p.sh`, salida `~/tesis/runs/sonda_p`):
- Preguntas: (1) ¿importa P?; (2) ¿un P que cambia a mitad de camino supera a cualquier P fijo? (justifica AOS
  *con estado*); (3) ¿sobra algún operador?
- 10 configuraciones de P = (pesos, conexión, nodo): Stanley (0,8/0,05/0,03), uniforme, pesos (0,98/0,01/0,01),
  estructura (0,5/0,25/0,25), nodos (0,6/0,1/0,3); cambio en la generación 250 estructura→pesos y pesos→estructura;
  ablaciones de Stanley sin conexión, sin nodo y sin pesos.
- 3 entornos × 2 protocolos (fijo, rotación) × 10 semillas × 500 generaciones = 600 corridas. Misma base que el
  barrido largo.
- `neat_control` ganó `--p`, `--p2`, `--p-switch` (solo selector estático). Verificación
  (`~/tesis/runs/sonda_p_verif/verificar.sh`): sin las opciones, idéntico al binario del barrido largo en 10 entornos
  + control negativo; `--p` con el P de Stanley = sin `--p`; el cambio ocurre en la generación indicada; la ablación
  usa el operador 0 veces; errores de uso rechazados. Binario congelado `~/tesis/runs/sonda_p_bin` (sha 245893d0e9eb3b68).
- Control posterior: stanley + fijo debe reproducir las primeras 500 generaciones del barrido largo.
- Recursos: primero 112 × 4 = 448 hilos (nodo casi libre); a las 22:06 se relanzó desde cero con 75 × 4 = 300 (otro usuario necesita 257). El primer lanzamiento no alcanzó a terminar corridas; sus carpetas incompletas se borraron.
- Lectura: P importa y el cambio gana → ideal para AOS con estado; P importa pero gana un P fijo → el AOS con estado
  debe superar a ese P ajustado; P no importa → ampliar el catálogo (pesos fino/grueso, poda, familias de
  Izhikevich) y repetir. En l2f, todos ≈ 0 puede ser efecto piso, no indiferencia a P.

### 16.1 Resultados de la sonda de P (2026-10-09)

600/600 corridas sin errores (terminó 2026-10-09 15:41). Control: stanley + fijo reproduce bit a bit las primeras
500 generaciones del barrido largo en las 10 semillas de los 3 entornos. Tablas completas en
`docs/resultados_sonda_p.md` (copia de `~/tesis/runs/analisis_sonda_p.md`) (`scripts/analisis_sonda_p.py`, con scipy del entorno conda `tesis-analisis`).

| Entorno | ¿Importa P? (Kruskal–Wallis fijo / rot) | Cambio a mitad vs mejor P fijo | Ablaciones vs Stanley |
|---|---|---|---|
| bottleneck | p = 0,065 / 0,022; mucha estructura perjudica (uniforme 0,13 fijo), más pesos ayuda (0,32 rot) | no lo supera (0,32 = 0,32) | sin pesos → ~0 (p Holm 0,015 / 0,009); sin conexión y sin nodo no cambian |
| acrobot-swingup | p = 0,98 / 0,36; todos 0,43–0,53 | no | ninguna cambia; incluso sin pesos (0,43 / 0,53) y sin nodo con 26 conexiones |
| l2f | p = 0,97 / 0,93; todos 0,02–0,11 (efecto piso) | no | ninguna cambia |

Lectura (sin decisión todavía, el usuario decide con todo a la vista):
- Solo en bottleneck la mezcla de operadores mueve el resultado, y el efecto se captura con un P fijo cargado a
  pesos. No hay evidencia de que cambiar de estrategia a mitad de la corrida ayude con estos 3 operadores.
- En acrobot-swingup el techo (~0,45–0,5) no depende de P ni del tamaño de la red (26 a 419 conexiones dan lo
  mismo): el límite parece estar fuera de los operadores (codificación/decodificación de la SNN, perturbación de
  pesos demasiado gruesa, hiperparámetros de NEAT).
- En l2f NEAT no progresa con ningún P (validación plana entre las generaciones 250 y 500).
- Que quitar la mutación de pesos no cambie nada en acrobot-swingup y l2f sugiere que el operador actual
  (perturbar todos los pesos ±5) no está haciendo ajuste fino útil.

## 17. Estudio de hiperparámetros de NEAT con Optuna (2026-10-09)

**[D] Orden (profesor y usuario, 2026-10-09):** primero NEAT estático en su mejor versión, luego Adaptive Pursuit con
el mismo procedimiento; el catálogo ampliado de operadores queda para después. Protocolo: **rotación**. Alcance: A + B.

- **Hiperparámetros (17):** población (50–300; generaciones = 75 000 / población), especies objetivo (3–25), escala
  de compatibilidad (2–30, log), supervivencia (0,1–0,5), perturbación de pesos (0,5–15, log), probabilidad de
  reasignar (0–0,2), peso del nodo nuevo (15–50), rango del peso inicial (10–50), peso máximo (1–3 × rango
  inicial), fracción sin cruce (0,1–0,6), estancamiento por especie (10–40), ganancia de entrada (10–100, log),
  spikes para acción máxima (2–15), ventana por paso (20–100 ms) y P = (pesos 0,05–1; conexión y nodo 0,005–0,5, log).
- **Fijos:** 10 episodios de entrenamiento, 10 de validación, 100 de test final; presupuesto 75 000 evaluaciones.
- `neat_control` ganó `--weight-init-range`, `--weight-max`, `--mutation-only`, `--species-stagnation`; el summary
  registra los hiperparámetros solo si difieren del valor por defecto. Verificación
  (`~/tesis/runs/optuna_verif/verificar.sh`): sin opciones = binario de la sonda en 10 entornos + control negativo;
  valores por defecto explícitos = sin opciones; cada opción nueva cambia el resultado. Binario congelado
  `~/tesis/runs/optuna_neat_bin` (sha 93511927c1b9b345).
- **Optuna** (`scripts/optuna_neat.py`, entorno conda `tdi`, Optuna 4.9): TPE multivariado (20 trials al azar al
  inicio, constant liar), MedianPruner sobre el mejor score de validación en cada 10 % del presupuesto, semillas de
  ajuste 101–103 (nunca 1–10), objetivo = score medio del test final en esas semillas, trial 0 = configuración actual.
  100 trials por entorno (completos + podados); se detiene si acumula 5 fallidos. JournalStorage en
  `~/tesis/runs/optuna_neat/<entorno>/estudio.log`.
- Prueba de humo (`~/tesis/runs/optuna_humo`): encontró un error de índice con presupuestos mínimos (corregido) y
  confirmó la poda (6 de 16 trials podados) y la limpieza de procesos.
- **Lanzado 2026-10-09 16:15–16:17** (`scripts/lote_optuna_neat.sh`): bottleneck 12 trabajadores, acrobot-swingup 8,
  l2f 5 = 300 hebras. `scripts/reasignar_optuna.py` (tmux `optuna_reasignar`) mantiene 25 trabajadores y pasa los
  cupos de los estudios terminados a los que siguen (bottleneck primero).
- Después: evaluar la mejor configuración de cada entorno en las semillas 1–10.
