# Operadores de mutación candidatos para el catálogo del AOS

Inventario de operadores de mutación para NEAT con redes de espigas (Izhikevich), revisado contra la literatura,
las implementaciones de referencia y nuestros propios datos, más una **lista corta** para la sonda del catálogo.
Documento de trabajo (2026-10-13). Decisiones relacionadas: `comprension.md` §9, `diseno_neat_ap.md` §5.4, §15.3 y §16–17.

## 0. Para qué sirve este documento

- El catálogo de operadores es el **espacio de acciones** del AOS (AP y PPO) y lo comparten las tres
  configuraciones: estático, AP y PPO (`propuesta.md` §3.4). Por eso se elige **antes** de ajustar AP o entrenar PPO, y con
  criterios que no favorezcan a ningún método.
- Alcance fijado por la propuesta: **operadores de mutación, paramétricos y topológicos**. El cruce, la selección y
  la especiación quedan fuera de la acción (pueden entrar al estado del MDP).
- Lo que ya sabemos: con los 3 operadores de Stanley **no hay espacio para adaptar** (sonda de P, §2). El catálogo
  ampliado es lo que decide si el AOS tiene algo que hacer.

## 1. Cómo se hizo la búsqueda

| Capa | Qué se revisó | Cómo |
|---|---|---|
| Variantes de NEAT | NEAT (Stanley y Miikkulainen 2002), FS-NEAT, FD-NEAT, búsqueda por fases, simplificación vs complejización, RBF-NEAT, Cascade-NEAT, SNAP-NEAT, MM-NEAT, Modular NEAT, WANN, AGENT, τ-NEAT, NEAT con SNN (Qiu 2019; nuestro grupo, 2026) | Papers y textos completos donde fue posible |
| Implementaciones reales | **SharpNEAT 4**, **neat-python**, **MultiNEAT** | **Código fuente** (clon de los repositorios): qué operadores existen y con qué probabilidades por defecto |
| Neuroevolución de SNN | **EONS** (TENNLab, neuromórfico), **SUNA**, evolución de parámetros de Izhikevich, heterogeneidad de constantes de tiempo, retardos | Papers |
| Operadores de pesos | Gauss vs Cauchy/Lévy (Yao), mutación segura (Lehman 2018), Iso+LineDD (Vassiliades y Mouret 2018), tasas adaptativas (GESMR 2022; tensor 2017), híbridos NEAT + CMA-ES | Papers |
| Selección adaptativa | Survey de Pei et al. (2025), DMAB y OneMax (Da Costa, Fialho et al.), carteras de estrategias de DE | Papers (survey: texto local) |
| Nuestros datos | Sonda de P (600 corridas), Optuna (3 estudios), barrido largo, medición de la zona muerta, código de `snn/` y `src/neat/` | Análisis propio |

## 2. Lo que dicen nuestros datos (requisitos para el catálogo)

1. **P sola no da margen:** ninguna P supera claramente a la de Stanley y ningún cambio a mitad de corrida gana a la
   mejor P fija (sonda de P). Elegir *qué* operador usar no basta: hacen falta operadores que difieran en **intensidad** y en
   **objetivo**.
2. **La intensidad de los pesos importa:** en l2f, la probabilidad de reasignar un peso explica el 34 % de la importancia en Optuna, y
   los mejores casi no reasignan (potencia 0,6–1,3 frente a 5 por defecto). Nuestro operador de pesos toca **todas** las
   conexiones (~1300 en l2f) y reasigna ~10 %: es una macromutación sin contraparte fina.
3. **La estructura útil depende del entorno:** en bottleneck, la estructura de más perjudica (uniforme: 315 conexiones contra 143,
   score 0,13 contra 0,27); en acrobot-swingup, las mejores configuraciones ajustadas usan P de conexión 0,3–0,5 y muchas conexiones.
   → Conviene tener **crecer y podar** en el catálogo.
4. **La interfaz pesa:** la ganancia de entrada explica el 55 % de la importancia en bottleneck, y en acrobot los mejores usan ganancia
   62–84 con pasos de 20–30 ms. Con ganancia 40 y 50 ms, una entrada con |x| ≤ 0,05 no produce espigas (zona muerta medida).
5. **Las redes no tienen parámetros por neurona:** todas las neuronas son TONIC_SPIKING, sin sesgo y con τ fijas. El
   simulador ya soporta por neurona `i_offset`, `tau_exc`, `tau_inh`, el tipo (25 familias) y (a, b, c, d); el genoma no los
   codifica.
6. **NEAT se estanca:** por defecto deja de mejorar antes de 150 mil evaluaciones (barrido largo); falta medir si el ajustado
   también se estanca (corridas de 300 mil en curso).

## 3. Principios de diseño del catálogo (literatura de AOS)

- **Diversidad:** "un AOS eficaz aumenta la eficiencia, especialmente cuando el conjunto de operadores candidatos es muy
  diverso"; se necesitan operadores de paso grande para escapar de óptimos locales y de paso chico para explotar
  (Pei et al., 2025, §I). El catálogo actual es casi homogéneo en intensidad.
- **Intensidades graduadas:** el caso canónico donde el mejor operador **cambia durante la corrida** es una cartera del mismo
  operador con distintas intensidades. En OneMax con voltear 1, 3 o 5 bits, el de 5 bits es el mejor al inicio y el de 1 bit al final
  (Da Costa, Fialho et al., 2008).
- **Rasgos de estado observables:** el survey recomienda como estado el último operador, la última recompensa y el
  presupuesto consumido, más rasgos de la solución (diversidad y estadísticas de fitness). Un operador vale más en el catálogo si
  su utilidad depende de algo que el agente puede observar (estancamiento, diversidad, tamaño de las redes, actividad).
- **Un operador por hijo:** es nuestro esquema categórico, y es también el de **SharpNEAT 4**, donde las probabilidades de mutación
  suman 1 y cada hijo recibe una. Encaja con P, con AP y con la acción de PPO.
- **Tamaño:** ~6–8 operadores en el catálogo final. Con población ~150, cada operador recibe ~20 hijos por generación;
  con 15 o más, el crédito por operador se vuelve demasiado ruidoso.
- **Neutralidad en SNN:** con sinapsis de conductancia, una mutación "neutral" en una red clásica puede no serlo aquí
  (agregar nodo exige un peso de entrada de ~30 para retransmitir). Se distingue entre operadores **neutros**,
  **suaves** y **disruptivos**.

## 4. Inventario completo

Columnas: **Fuente** = dónde existe · **Factible** = costo en nuestro código (bajo / medio / alto) · **Cuándo debería servir**
= fase o estado de la población (insumo para el MDP) · **Nota** = redundancia o riesgo.

### A. Pesos sinápticos (paramétricos)

| ID | Operador | Qué hace | Fuente | Factible | Cuándo debería servir | Nota |
|---|---|---|---|---|---|---|
| A1 | Pesos global (actual) | Perturba **todas** las conexiones U(−5, 5) y reasigna cada una con p = 0,1 | Stanley y Miikkulainen 2002 | ya existe | Inicio: exploración amplia | Macromutación con redes grandes |
| A2 | **Pesos finos dispersos** | Perturba **1–3 conexiones** con σ chico | **SharpNEAT 4**: ~96 % de sus mutaciones de pesos son deltas gaussianos sobre 1, 2 o 3 conexiones | bajo | Fin o meseta: ajuste fino | Respaldado por Optuna en l2f |
| A3 | **Reasignar pesos puntuales** | Da un valor nuevo al azar a 1–3 conexiones | SharpNEAT (~4 %: reinicio de 1–3); NEAT (10 % por peso, mezclado en A1) | bajo | Estancamiento: escapar sin destruir toda la red | Separar de A1 permite controlar su tasa |
| A4 | Sacudida severa | Remuestrea todos los pesos con mayor magnitud | MultiNEAT (`MutateWeightsSevereProb`) | bajo | Estancamiento profundo | Probablemente redundante con A1 |
| A5 | Cola pesada (Cauchy o Lévy) | Perturbación con saltos largos ocasionales | Yao et al. 1999 (FEP/IFEP); Lee y Yao 2004 (Lévy) | bajo | Óptimos locales, multimodalidad | Variante de intensidad de A1/A2 |
| A6 | Enfocado en lo nuevo | Muta solo las conexiones de la estructura más reciente | Cascade-NEAT congela la estructura previa (Kohl y Miikkulainen 2012) | medio (requiere la edad del gen) | Justo después de crecer | Es una variante de A2 |
| A7 | **Dirigido por la población** | Hijo = padre + ruido isotrópico + paso a lo largo de (w_b − w_a), con otro miembro de la especie, sobre **genes coincidentes** (números de innovación) | Iso+LineDD (Vassiliades y Mouret 2018), análogo a DE | medio (necesita a la especie en `reproduce`) | Especie convergida: explota la correlación entre élites | No encontré precedente en NEAT: posible aporte propio |
| A8 | Mutación segura (SM-R) | Escala una perturbación al azar hasta un cambio objetivo en las acciones, medido sobre observaciones guardadas del padre | Lehman et al. 2018 | medio-alto (simular la red sobre un buffer) | Redes grandes o frágiles | Sin gradientes el efecto es débil (SM-G necesita gradientes) |
| A9 | Escalar entradas de una neurona | Multiplica todas las conexiones entrantes de una neurona por s | "response" de neat-python: salida = act(sesgo + response × agregado) | bajo | Neurona saturada o silenciosa | Equivale a una ganancia por neurona; se solapa con B1 |
| A10 | Invertir polaridad | Cambia el signo de una sinapsis (excitatoria ↔ inhibitoria) | Debate sobre el principio de Dale: algunas evoluciones fijan el signo; NEST lo prohíbe | bajo | — | Nuestro peso con signo ya lo permite vía A1/A3 |

### B. Neurona (paramétricos propios de SNN)

| ID | Operador | Qué hace | Fuente | Factible | Cuándo debería servir | Nota |
|---|---|---|---|---|---|---|
| B1 | **Excitabilidad (corriente de reposo)** | Muta el sesgo `i_offset` de una neurona oculta o de salida | Sesgo en neat-python (`bias_mutate_rate` 0,7 por defecto) y MultiNEAT; umbral de neurona en **EONS** (`node_param_rate` 0,27) | bajo (el simulador lo soporta; falta en el genoma) | Neuronas silenciosas o saturadas; ajuste fino del punto de operación | Hoy no hay sesgo en la red |
| B2 | **Dinámica de Izhikevich continua** | Perturba (a, b, c, d) dentro de rangos acotados | Candadai Vasu e Izquierdo 2017 (parámetros intrínsecos evolucionados, con gen excitatoria/inhibitoria); ajuste evolutivo de Izhikevich (Venkadesh et al. 2018) | medio (setter en el simulador y genes de nodo) | Media o fin: especializar neuronas | Versión suave del cambio de familia |
| B3 | **Cambiar familia de neurona** | Cambia el tipo (RS, FS, IB, CH…) de una neurona | Cambio de activación en WANN, neat-python y MultiNEAT ("estructural por su gran impacto"); tipos de SUNA | bajo (`setNeuronType` existe) | Exploración de dinámicas | Al usuario no le gusta como cambio al azar: considerar B2 o C4 |
| B4 | Constantes de tiempo sinápticas | Muta `tau_exc` y `tau_inh` de una neurona | Perez-Nieves et al. 2021 (la heterogeneidad de τ mejora el aprendizaje); "time constants" en MultiNEAT; velocidades de adaptación en SUNA | bajo (setters existen) | Tareas con estructura temporal | Con la red reiniciada cada paso, el efecto se limita a la ventana de 20–100 ms |
| B5 | Despertar neuronas silenciosas | Sube la excitabilidad o los pesos de entrada de las neuronas que no dispararon en la evaluación del padre | No encontré precedente | medio (registrar la actividad) | Muchas neuronas silenciosas | Dirigido por actividad: posible aporte propio |

### C. Estructura: crecer

| ID | Operador | Qué hace | Fuente | Factible | Cuándo debería servir | Nota |
|---|---|---|---|---|---|---|
| C1 | Agregar conexión (actual) | Nueva sinapsis entre nodos no conectados, peso al azar | NEAT; todas las implementaciones | ya existe | Crecer conectividad | |
| C2 | Conexión recurrente o autoconexión | Fuerza una conexión hacia atrás o un lazo | MultiNEAT (`RecurrentProb`, `RecurrentLoopProb`) | bajo | Dinámica dentro de la ventana | C1 ya permite recurrencia, no autoconexiones |
| C3 | Agregar neurona (actual) | Divide una conexión (entrada 30, salida con el peso viejo) | NEAT | ya existe | Crecer capacidad | Casi neutro con el peso de entrada calibrado |
| C4 | **Agregar neurona de una familia** | Como C3, pero la neurona nueva es RS, FS, IB o CH | Idea del usuario (4 familias corticales de Izhikevich 2004); análogo a agregar un nodo de tipo distinto en SNAP-NEAT (RBF, cascada) | bajo | Según la tarea: FS para inhibición rápida, IB/CH para ráfagas | La sonda decide si son 4 operadores o uno con la familia sorteada |
| C5 | **Duplicar neurona** | Copia una oculta con sus entradas y reparte a la mitad sus salidas | Net2WiderNet (Chen et al. 2016); "duplicate" de MM-NEAT | bajo | Agregar capacidad sin romper nada | **Exactamente neutro** en nuestro simulador si el ruido es 0: las conductancias suman de forma lineal (verificar con un test) |
| C6 | Neurona en cascada | Neurona nueva conectada desde todas las entradas y ocultas hacia todas las salidas, congelando lo anterior | Cascade-NEAT (ε = 0,05; Kohl y Miikkulainen 2012) | medio | Problemas "fracturados" | Muy disruptiva con 164 entradas (l2f) |
| C7 | Neurona local tipo RBF | Neurona que responde a una región del espacio de entrada | RBF-NEAT (ε = 0,05) | alto en SNN | Problemas fracturados | Sin equivalente directo en espigas |
| C8 | Módulo nuevo | Agrega un módulo de salida (anterior, al azar o duplicado) con una neurona de preferencia | MM-NEAT (Schrum y Miikkulainen 2014–2016) | alto | Comportamiento multimodal | Fuera de alcance por ahora |
| C9 | **k mutaciones por hijo** | Aplica el operador k veces (por ejemplo, agregar 3 neuronas) | **EONS**: 7 mutaciones por hijo por defecto; **SUNA**: 5 por hijo | bajo | Inicio: construir rápido | Responde a la idea del usuario de "agregar muchos nodos"; es una intensidad, no un operador nuevo |
| C10 | Reactivar conexión | Habilita un gen deshabilitado | neat-python (`enabled_mutate_rate`) | bajo | Recuperar estructura antigua | Casi neutro |
| C11 | Conectar una entrada no usada | Conexión desde una entrada sin conexiones | FS-NEAT (Whiteson et al. 2005) | bajo | Si se parte con conectividad mínima | Nosotros partimos totalmente conectados: no aplica tal cual |

### D. Estructura: podar

| ID | Operador | Qué hace | Fuente | Factible | Cuándo debería servir | Nota |
|---|---|---|---|---|---|---|
| D1 | **Eliminar conexión** | Borra una conexión y las neuronas que quedan huérfanas | **SharpNEAT 4** (0,025 por defecto; 0,4 en la fase de simplificación); neat-python; EONS (0,14); SUNA (0,49) | bajo (el simulador tiene `removeSynapse`) | Redes infladas, mesetas | Idea del usuario |
| D2 | **Eliminar neurona simple** | Borra una neurona con una sola entrada o salida y conecta un puente | MultiNEAT (`MutateRemSimpleNeuronProb`); SharpNEAT (búsqueda por fases) | bajo | Simplificar sin cortar circuitos | Idea del usuario; conserva la función mejor que D3 |
| D3 | Eliminar neurona con sus conexiones | Borra la neurona y todo lo que la toca | neat-python (`node_delete_prob`); EONS (0,07) | bajo | Poda fuerte | Más disruptivo que D2 |
| D4 | Podar por magnitud | Borra la conexión de menor |w| | Poda clásica de redes | bajo | Fin: compactar | Dirigido; redundante con D1 si el AOS ya decide cuándo podar |
| D5 | Deshabilitar conexión | Apaga una conexión sin borrarla (reversible) | O5 de este diseño, estilo AGENT | bajo | Simplificación reversible | Casi igual a D1 |
| D6 | Quitar una entrada irrelevante | Borra las conexiones de una entrada | FD-NEAT (Tan et al. 2009) | bajo | Muchas entradas distractoras (l2f: 128 de 164 son historial) | Caso particular de D1 |

**Antecedente clave:** la búsqueda por fases de SharpNEAT (Green 2004) alterna complejizar y simplificar cambiando P según un
umbral fijo de complejidad. **Es un AOS hecho a mano sobre exactamente este tipo de catálogo**: el agente podría aprender
cuándo podar. James y Tucker (2004) compararon complejizar, simplificar y combinar ambos con NEAT.

### E. Interfaz entorno ↔ SNN (paramétricos)

| ID | Operador | Qué hace | Fuente | Factible | Cuándo debería servir | Nota |
|---|---|---|---|---|---|---|
| E1 | **Sensibilidad de entrada** | Muta la ganancia (u offset) de una neurona de entrada | Campos receptivos evolucionados (Luque et al. 2012); la codificación importa tanto como el modelo de neurona (Loyola-Jara, Fernández-Rodríguez y Baladron 2026) | bajo-medio (gen por entrada) | Inicio: descubrir qué variables mirar y con qué resolución | Optuna: la ganancia global pesó 55 % en bottleneck |
| E2 | Neurona sensorial nueva | Agrega una neurona de entrada con un campo receptivo propio | Codificación poblacional (paper del grupo; Luque 2012) | alto | — | Cambia el número de entradas: fuera de alcance |
| E3 | Escala de salida | Muta las espigas que equivalen a la acción máxima, por salida | Qiu et al. 2019 evolucionan el factor σ de decodificación y lo suman a la distancia de compatibilidad | bajo | Ajuste fino de la acción | Optuna: espigas máx. pesó poco |

### F. Descartados (y por qué)

| Operador | Fuente | Por qué no |
|---|---|---|
| Retardos sinápticos | τ-NEAT (Caamaño et al. 2016); EONS (`edge_param`: retardo) | El simulador no tiene retardos, y con la red reiniciada cada paso aportan poco |
| Plasticidad (STDP, neuromodulación) | SUNA, HyperNEAT adaptativo | Requiere memoria entre pasos; los entornos son reactivos por decisión |
| Búsqueda local, Lamarck, backprop | PropNEAT (2024), NEAT+Q, NEAT + CMA-ES (Hybrid Self-Attention NEAT, 2021) | Gasta evaluaciones fuera del presupuesto común, y la SNN no es diferenciable |
| Cruce y fusión | NEAT, EONS (`merge_rate`) | La propuesta limita la acción a mutaciones; el cruce sigue fijo |
| Inyectar redes al azar | EONS (`random_factor` 0,1) | Mecanismo de población, no operador de mutación |
| Codificación indirecta | HyperNEAT, ES-HyperNEAT | Cambia la representación completa |
| Tasas autoadaptadas | GESMR (Kumar et al. 2022), tasas por locus (Marzullo et al. 2017) | Son mecanismos alternativos de adaptación, no operadores. Útiles como trabajo relacionado |

## 5. Lista corta para la sonda (12 candidatos)

Criterios (fijados antes de mirar resultados de AP o PPO): **(1)** precedente en la literatura o justificación propia de SNN; **(2)** no redundante;
**(3)** factible a costo bajo o medio; **(4)** cubre una dirección que el catálogo actual no tiene (intensidad, poda, neurona,
interfaz o dirección); **(5)** su utilidad debería depender de una señal observable (para que el AOS tenga algo que aprender).

| # | Operador | Familia | Intensidad | Señal de estado esperada | Por qué entra |
|---|---|---|---|---|---|
| 1 | A1 Pesos global (actual) | Pesos | Alta | Inicio, diversidad alta | Referencia de Stanley; exploración amplia |
| 2 | A2 Pesos finos dispersos | Pesos | Baja | Meseta, fin | SharpNEAT + Optuna en l2f; el catálogo no tiene paso chico |
| 3 | A3 Reasignar pesos puntuales | Pesos | Media | Estancamiento | Separar el reinicio de A1 para controlar su tasa |
| 4 | A7 Dirigido por la población | Pesos | Variable | Especie convergida | Usa información de la población; útil distinto según el estado; posible aporte |
| 5 | B1 Excitabilidad (sesgo) | Neurona | Baja | Actividad (neuronas silenciosas o saturadas) | Estándar en neat-python, MultiNEAT y EONS; hoy no existe |
| 6 | B2/B3 Dinámica neuronal (familias) | Neurona | Media | Media o fin | Idea del usuario; la sonda decide entre perturbación continua y cambio de familia |
| 7 | E1 Sensibilidad de entrada | Interfaz | Baja | Inicio | Optuna: la ganancia de entrada pesó mucho en 2 de 3 entornos |
| 8 | C1 Agregar conexión (actual) | Crecer | Media | Redes chicas | Stanley |
| 9 | C3/C4 Agregar neurona (con familia) | Crecer | Media | Redes chicas, meseta | Stanley + idea del usuario; la sonda decide si la familia importa |
| 10 | C5 Duplicar neurona | Crecer | Neutra | Meseta: más capacidad sin perder lo aprendido | Exactamente neutro aquí; complementa a C3 |
| 11 | D1 Eliminar conexión | Podar | Media | Redes infladas | SharpNEAT, EONS, SUNA; idea del usuario |
| 12 | D2 Eliminar neurona simple | Podar | Baja | Redes infladas | MultiNEAT, búsqueda por fases; idea del usuario |

**Variantes de intensidad a tamizar sin agregar operadores:** C9 (k = 1 vs 3 aplicaciones, por ejemplo "agregar 3 neuronas") y
A5 (Gauss vs Cauchy en A2). Si alguna variante resulta distinta, puede entrar como operador propio; si no, se descarta.

**Reserva** (si algún candidato falla en la implementación): B4 (τ sinápticas), B5 (despertar silenciosas), D6 (quitar una entrada).

## 6. Cómo se integran en el código (resumen)

- **Genoma:**
  - `NodeGene` gana `i_offset` (B1), parámetros (a, b, c, d) o familia (B2/B3) y, en las entradas, una ganancia (E1).
  - La distancia de compatibilidad debe sumar un término por diferencias de parámetros de nodo, como hacen Qiu et al. 2019 con σ.
  - El cruce hereda los genes de nodo.
- **Simulador:**
  - `buildNetwork` aplica `setIOffset` y `setNeuronType`.
  - (a, b, c, d) necesitan un setter nuevo en `IzhikevichNeuron` y `Network`.
  - En las neuronas de entrada, `i_offset` se usa para inyectar la observación: la ganancia E1 se aplica en el evaluador.
- **Operadores en `Genome::apply`:** A2, A3, B1, B2/B3, E1, C4, C5, D1 y D2 son locales al genoma. **A7 necesita a la especie**, así que se aplica
  en `Population::reproduce`.
- **Fallos:** si un operador no aplica (por ejemplo, no hay neurona simple que borrar), definir un reintento con otro operador en
  vez de clonar sin cambios. SharpNEAT, cuando no hay conexiones, sortea solo entre operadores no destructivos.
- **Protección:** cada operador queda apagado por defecto (P = 0) y se verifica que, sin él, el binario siga dando resultados idénticos bit a bit;
  cada uno lleva su test (por ejemplo, que C5 sea neutro y que D2 conserve la conectividad).

### 6.1 Implementación (2026-10-13)

Implementados **todos** los candidatos de la lista corta y sus variantes (12 operadores nuevos; 15 con los de Stanley),
en `src/neat/operators.hpp` (catálogo), `src/neat/genome.cpp` (operadores) y `src/neat/population.cpp` (A7).

| Nombre en `--catalog` | ID | Qué hace exactamente | Parámetros (por defecto) |
|---|---|---|---|
| `mutate_weights`, `add_connection`, `add_node` | A1, C1, C3 | Los de Stanley, sin cambios | — |
| `weights_fine` | A2 | 1, 2 o 3 conexiones habilitadas (p = 0,6 / 0,3 / 0,1, como SharpNEAT), w += N(0, σ) | `--fine-sigma` 1,0; `--fine-cauchy` (A5) |
| `weights_reset` | A3 | 1–3 conexiones habilitadas, w ~ U(−rango inicial, rango inicial) | — |
| `weights_directed` | A7 | Iso+LineDD sobre genes coincidentes con otro sobreviviente de la especie (si es el único, cualquier genoma): w += σ_iso·N(0,1) + σ_line·g·(w_compañero − w), un solo g por hijo | `--directed-iso` 0,6; `--directed-line` 0,2 |
| `neuron_bias` | B1 | Corriente de reposo de una oculta o salida: += N(0, σ), acotada | `--bias-sigma` 1,0; `--bias-max` 10 |
| `neuron_dynamics` | B2 | (a, b, c, d) de una oculta o salida: parte de su familia y perturba dentro de rangos que cubren RS, FS, IB, CH, LTS, RZ y TONIC | σ = (0,01; 0,01; 2; 0,5) |
| `neuron_family` | B3 | Otra familia entre RS, FS, IB y CH para una oculta o salida | — |
| `input_gain` | E1 | Ganancia de una entrada × exp(N(0, σ)), en [0,25; 4]; el evaluador multiplica la corriente | `--gain-log-sigma` 0,3 |
| `add_node_family` | C4 | Agregar nodo con la neurona nueva de una familia al azar (RS, FS, IB, CH) | — |
| `add_node_multi` | C9 | Hasta k "agregar nodo" en una mutación | `--multi-k` 3 |
| `duplicate_node` | C5 | Copia una oculta con sus entradas y reparte a la mitad sus salidas; marca histórica propia | — |
| `delete_connection` | D1 | Borra una conexión habilitada y las ocultas que quedan sin genes | — |
| `delete_simple_node` | D2 | Oculta con 1 entrada y 1 salida → puente s → t con el peso de salida (si s → t existe deshabilitada, se reactiva; si está habilitada, se suman); oculta sin entradas o sin salidas → se borra | — |

Decisiones de implementación:
- **Catálogo configurable:** `--catalog` fija los operadores y su orden; P (`--p`, `--p2`), AP y el crédito trabajan sobre las
  posiciones del catálogo. Con un catálogo distinto del de Stanley y selector estático, `--p` es obligatorio; AP parte uniforme.
- **Genes de nodo:** sesgo, ganancia y (a, b, c, d) propios. El cruce hereda todos los parámetros de un nodo común desde
  uno de los padres, con el mismo sorteo que antes usaba solo para el tipo. La distancia δ suma
  `c4 · media de diferencias de nodo` (`--c4` 0,4), y cada término se normaliza a ~[0, 1].
- **Fallas:** si un operador no aplica (por ejemplo, no hay ocultas que duplicar), el hijo queda como copia, igual que antes con
  agregar conexión o nodo. El CSV registra la fracción de fallas por operador (`--credito-rango`); al inicio de una corrida
  `duplicate_node` y `delete_simple_node` fallan casi siempre, porque aún no hay ocultas.
- **Crédito por rango** (`--credito-rango`): por operador, percentil medio de sus hijos en la generación (`rank_`), fracción en
  el 20 % superior (`top_`) y fracción de fallas (`fail_`). Con rotación es más justo que "hijo > padre", porque padre e
  hijo se evalúan en episodios distintos.

Verificación:
- 77/77 tests, 17 nuevos (`tests/operadores_tests.cpp` y uno en `tests/eval_tests.cpp`). Entre ellos:
  - duplicar neurona produce **las mismas espigas de salida** que el original;
  - agregar nodo y luego borrarlo como nodo simple **recupera el fenotipo**;
  - el operador dirigido sin ruido es colineal con (w_compañero − w);
  - ganancia de entrada 2 equivale exactamente a duplicar `--input-gain`.
- **Equivalencia bit a bit** con el binario congelado de Optuna (sha 93511927…) sin las opciones nuevas, en 8 casos: pendulum,
  acrobot-swingup con rotación, bottleneck, l2f con la configuración ajustada #65, memory con memoria, AP, AP SNAP y cambio de P.
  También pasa el control negativo (`~/tesis/runs/operadores_verif/verificar_defecto.sh`). El tiempo de ejecución no cambia.
- Prueba de humo de cada operador y del catálogo completo en los 3 entornos (`humo_operadores.sh`): sin errores.
- Binario congelado para la sonda: `~/tesis/runs/operadores_bin/neat_control` (sha ae239e87152505b2).

## 7. Plan de la sonda del catálogo (con NEAT estático, neutral entre AP y PPO)

1. **Tamizado (barato):** base NEAT ajustada de cada entorno + Stanley + **un** candidato con P moderada (el mismo peso que el
   operador de Stanley más parecido), 5 semillas, 75 mil evaluaciones. Descarta lo que no cambia nada o rompe la red.
2. **Utilidad (ablación):** con los ~8–10 que pasan, catálogo completo frente al catálogo sin cada uno.
3. **P óptima con ceros permitidos:** Optuna sobre P del catálogo (un operador puede quedar en 0); la importancia por operador
   ordena el catálogo.
4. **Espacio para adaptar:** Optuna con P₁, P₂ y la generación del cambio, frente al mejor P fijo. Si un P que cambia gana, hay margen
   para AP y para PPO por igual.
5. **Registros por generación:** éxito de cada operador junto con los rasgos de la población, para elegir el estado del MDP.
6. **Congelar** ~6–8 operadores (reunión con el profesor) y reportar siempre también el catálogo de Stanley como control.

## Referencias

Implementaciones (código revisado el 2026-10-13):
- SharpNEAT 4, `NeatAsexualReproductionSettings` y `WeightMutationSchemeFactory`: https://github.com/colgreen/sharpneat
- neat-python, `genome.py` y `nn/feed_forward.py`; configuración: https://neat-python.readthedocs.io/en/latest/config_file.html
- MultiNEAT, `src/Parameters.h`: https://github.com/peter-ch/MultiNEAT

Papers:
- Stanley y Miikkulainen 2002, Evolving Neural Networks through Augmenting Topologies (texto local en `~/tesis/privado/papers/`).
- Kohl y Miikkulainen 2008, Evolving Neural Networks for Fractured Domains (GECCO): https://gpbib.cs.ucl.ac.uk/gecco2008/docs/p1405.pdf
- Kohl y Miikkulainen 2012, An integrated neuroevolutionary approach to reactive control and high-level strategy (IEEE TEC 16(4)), SNAP-NEAT (texto local).
- Green 2004, Phased Searching with NEAT: https://sharpneat.sourceforge.io/phasedsearch.html
- James y Tucker 2004, A Comparative Analysis of Simplification and Complexification in the Evolution of Neural Network Topologies (GECCO LBP): https://gpbib.cs.ucl.ac.uk/gecco2004/prof171.html
- Whiteson et al. 2005, Automatic Feature Selection in Neuroevolution (FS-NEAT): https://gpbib.cs.ucl.ac.uk/gecco2005/docs/p1225.pdf
- Tan et al. 2009, FD-NEAT (Evolutionary Intelligence): https://researchportal.vub.be/en/publications/automated-feature-selection-in-neuroevolution/
- Schrum y Miikkulainen 2014, módulos en Ms. Pac-Man (MM-NEAT): https://nn.cs.utexas.edu/downloads/papers/schrum.gecco14.pdf
- Reisinger, Stanley y Miikkulainen 2004, Evolving Reusable Neural Modules: https://nn.cs.utexas.edu/downloads/papers/reisinger.gecco04.pdf
- Gaier y Ha 2019, Weight Agnostic Neural Networks: https://arxiv.org/abs/1906.04358
- Behjat et al. 2019, AGENT: https://arxiv.org/abs/1903.07107
- Caamaño et al. 2016, τ-NEAT (Neural Processing Letters): https://link.springer.com/article/10.1007/s11063-015-9426-5
- Qiu et al. 2019, Evolving Spiking Neural Networks for Nonlinear Control Problems: https://arxiv.org/abs/1903.01180
- Loyola-Jara, Fernández-Rodríguez y Baladron 2026, Evolving spiking neural networks: the role of neuron models and encoding schemes (Frontiers in Neuroscience): https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1697163/full
- Schuman et al. 2020, Evolutionary Optimization for Neuromorphic Systems (EONS): https://www.osti.gov/servlets/purl/1649325
- Vargas y Murata 2017, Spectrum-Diverse Neuroevolution with Unified Neural Models (SUNA): https://arxiv.org/abs/1902.06703
- Candadai Vasu e Izquierdo 2017, Evolution and analysis of embodied spiking neural networks… (GECCO): https://arxiv.org/abs/1704.04199
- Perez-Nieves et al. 2021, Neural heterogeneity promotes robust learning (Nature Communications): https://pmc.ncbi.nlm.nih.gov/articles/PMC8490404
- Luque et al. 2012, From sensors to spikes: evolving receptive fields…: https://acn.ugr.es/publication/luque2012/
- Lehman et al. 2018, Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients: https://arxiv.org/abs/1712.06563
- Vassiliades y Mouret 2018, Discovering the Elite Hypervolume by Leveraging Interspecies Correlation: https://arxiv.org/abs/1804.03906
- Yao, Liu y Lin 1999, Evolutionary Programming Made Faster (IEEE TEC 3(2)): https://www.cse.unr.edu/%7Esushil/class/gas/papers/EPMadeFaster.pdf
- Lee y Yao 2004, EP using mutations based on the Lévy probability distribution (IEEE TEC 8(1)).
- Chen, Goodfellow y Shlens 2016, Net2Net: https://arxiv.org/abs/1511.05641
- Kumar, Liu, Miikkulainen y Stone 2022, Effective Mutation Rate Adaptation through Group Elite Selection (GECCO).
- Marzullo et al. 2017, A tensor-based mutation operator for NEAT (IEEE CEC): https://ieeexplore.ieee.org/document/7969376
- Pei et al. 2025, Adaptive Operator Selection for Meta-Heuristics: A Survey (texto local en `~/tesis/privado/papers/`).
- Da Costa, Fialho, Schoenauer y Sebag 2008, Adaptive Operator Selection with Dynamic Multi-Armed Bandits (GECCO): https://www.researchgate.net/publication/220740267_Adaptive_Operator_Selection_with_Dynamic_Multi-Armed_Bandits
- Papavasileiou, Cornelis y Jansen 2021, A Systematic Literature Review of the Successors of NEAT (Evolutionary Computation 29(1)): https://direct.mit.edu/evco/article/29/1/1/97341
