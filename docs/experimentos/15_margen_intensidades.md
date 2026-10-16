# 15 · Prueba de margen de intensidades (plan B)

| | |
|---|---|
| Semana | S04 (lanzada el 2026-10-15 a las 00:58) |
| Estado | terminado (2026-10-15 02:28) |
| Capítulo de la tesis | Metodología: espacio de acción del agente · Resultados: margen para adaptar |

## Pregunta

La prueba de margen de operadores ([13](13_prueba_margen.md)) mostró poco margen al elegir **qué** operador aplicar:
- +0,086 en acrobot-swingup y +0,026 en bottleneck;
- AP ya captura el 60–100 % de ese margen.

Optuna mostró que pesan más **cuánto y cómo** se varía:
- la intensidad de la mutación de pesos y la fracción de hijos sin cruce: 34 % y 26 % de importancia en l2f;
- en acrobot-swingup, la potencia ajustada es 0,69 contra 5 genérica.

Pregunta: ¿cuánto vale ajustar las intensidades de variación, y lo captura un AOS que solo mueve P?

## Diseño

- **Intensidades:** potencia de la mutación de pesos, probabilidad de reasignar pesos y fracción de hijos sin cruce.
- **Valores genéricos:** los de la configuración previa a Optuna (su trial 0): 5, 0,02 y 0,25.
- **Configuraciones:** catálogo de 10 operadores, 5 semillas pareadas (201–205), 75k evaluaciones, rotación:

| Config. | P | Intensidades | Estado |
|---|---|---|---|
| c0 | uniforme | genéricas | nueva (NEAT sin ajuste de nada) |
| c1 | uniforme | ajustadas | ya corrida ([13](13_prueba_margen.md)) |
| c2 | Stanley + P₀ ajustada | ajustadas | ya corrida |
| c3g | AP desde uniforme | genéricas | nueva |

- **Binario:** `fase_ppo_bin` (idéntico bit a bit a `operadores_bin`, [14](14_rendimiento.md)).
- **Corridas:** 20 en bottleneck y acrobot-swingup; 10 en pendulum-meta, después de su Optuna.

**Reglas fijadas antes de mirar:**
- **Margen de intensidades** = c1 − c0. Si es ≥ 0,03 en al menos 2 de 3 entornos, las intensidades importan más allá de P.
- **Captura de AP** = c3g − c0. Si es menor que ½ (c2 − c0), el AOS que solo mueve P no captura el margen: hay espacio para
  un agente que controle las intensidades (plan B).
- Si el margen de intensidades es < 0,03 en todos, el plan B no tiene base en estos datos.

## Resultados

**3 entornos** (`analisis_intensidades.py`, salida en `analisis_intensidades_3entornos.txt`):

| Entorno | c0 genérico | c1 (int. ajustadas) | c2 todo ajustado | c3 AP (int. ajustadas) | c3g AP (int. genéricas) |
|---|---|---|---|---|---|
| bottleneck | 0,332 | 0,332 | 0,358 | 0,359 | 0,296 |
| acrobot-swingup | 0,541 | 0,500 | 0,587 | 0,552 | 0,549 |
| pendulum-meta | 0,354 | 0,324 | 0,430 | 0,361 | 0,276 |

- **Margen de intensidades** c1 − c0: +0,000 (bottleneck), −0,041 (acrobot-swingup, 0/5) y −0,030 (pendulum-meta). Con
  0 de 3, la regla dice que **ajustar las intensidades de forma estática no da margen sobre las genéricas**.
- **Interacción con P:**
  - las intensidades ajustadas para P₀ empeoran con P uniforme en acrobot-swingup;
  - AP con intensidades genéricas cae 0,063 respecto de AP con las ajustadas en bottleneck.

  La intensidad conveniente depende de la mezcla de operadores.
- **Margen total** c2 − c0: +0,026 (4/5), +0,046 (3/5) y +0,077 (3/5). AP con intensidades genéricas no lo captura, e
  incluso empeora en 2 entornos: −0,036, +0,008 y −0,078.

**Decisión (2026-10-15 01:50):** el entrenamiento de PPO desde c0 ([16](16_ppo_planB_e1.md)) sigue con la acción "ambas"
(P + intensidades). No respalda un plan B "estático", que la regla descarta. Mide si un agente con estado que mueve las dos
cosas juntas captura el margen total que AP no captura. Es exploratorio respecto de la regla de esta ficha.

## Conclusión y límites

Pendiente. **Límites conocidos:**
- 5 semillas;
- valores genéricos tomados de la configuración previa a Optuna (no son la peor elección posible);
- la ganancia de entrada (la de mayor importancia en Optuna) queda fuera, por ser parte de la interfaz del problema y no un
  operador de variación.

## Datos y reproducción

- `~/tesis/runs/margen_aos/`: `preparar_intensidades.py` (agrega a `jobs.json`; respaldo en `jobs.json.antes_intensidades`),
  `lanzar.py` (planificador), resultados en `<entorno>/c0_generico` y `<entorno>/c3g_ap_generico`.
