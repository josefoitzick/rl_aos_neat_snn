# 17 · Comparación final en el escenario E1 (sin ajuste previo)

| | |
|---|---|
| Semana | S04 (líneas base lanzadas el 2026-10-15 a las 02:35) |
| Estado | terminado (2026-10-15 08:30): políticas de 800 y de 1408 corridas evaluadas |
| Capítulo de la tesis | Resultados: comparación de métodos |

## Pregunta

Partiendo sin ajuste previo (P uniforme sobre el catálogo de 10 e intensidades genéricas), ¿cuánto del margen hasta NEAT
ajustado con Optuna recupera cada método?
- estático (no adapta nada);
- AP (sin estado, solo mueve P);
- PPO (con estado, mueve P e intensidades).

E1 es el escenario principal según la prueba de margen ([13](13_prueba_margen.md)).

## Diseño

- **Semillas de evaluación** 1–11, nunca usadas al ajustar ni al entrenar. 75k evaluaciones, rotación.
- **Entornos:** bottleneck, acrobot-swingup y pendulum-meta.
- **Métodos:**

  | Método | Configuración |
  |---|---|
  | c0, estático sin ajuste | P uniforme + intensidades genéricas (= acción 0 del agente) |
  | c3g, AP sin ajuste | AP desde P uniforme + intensidades genéricas |
  | PPO (plan B) | Política entrenada en [16](16_ppo_planB_e1.md), evaluada con `aos_ppo_train --evaluar` (mismo binario del entrenamiento). bottleneck es transferencia: el agente no entrenó ahí |
  | c2, referencia ajustada (Optuna) | Para normalizar. En bottleneck y acrobot-swingup, semillas 1–10 de `eval_optuna` (binario idéntico bit a bit) + la 11 nueva |

- **Estadística:** Wilcoxon de rangos con signo pareado por semilla, corrección de Holm por entorno, Â12 pareado
  (`analysis/compare.py --pareado`).
- **Binario:** `fase_ppo_bin2` (sha `52e237c8…` para `neat_control`).

## Resultados

**Política de 800 corridas** (`comparacion_final/analisis_comparacion.py`, salida en `analisis_comparacion_e1.txt`). Score
medio ± DE, 11 semillas pareadas:

| Entorno | c0 estático sin ajuste | AP sin ajuste | **PPO (plan B)** | c2 ref. ajustada |
|---|---|---|---|---|
| bottleneck (transferencia) | 0,289 ± 0,111 | 0,313 ± 0,083 | **0,346 ± 0,070** | 0,327 ± 0,047 |
| acrobot-swingup | 0,539 ± 0,039 | 0,524 ± 0,035 | **0,548 ± 0,047** | 0,565 ± 0,045 |
| pendulum-meta | 0,332 ± 0,153 | 0,344 ± 0,143 | **0,287 ± 0,125** | 0,379 ± 0,156 |

Comparaciones pareadas (Wilcoxon, Holm por entorno):

| Entorno | PPO − c0 | AP − c0 | PPO − AP | Fracción del margen c2 − c0 (AP / PPO) |
|---|---|---|---|---|
| bottleneck | +0,056 (7/11, p 0,17, Holm 0,52) | +0,024 (6/11) | +0,032 (5/11) | 64 % / 151 % |
| acrobot-swingup | +0,010 (8/11, p 0,24) | −0,014 (3/11, p 0,054) | +0,024 (8/11, p 0,10) | −55 % / 38 % |
| pendulum-meta | −0,045 (5/11, p 0,32) | +0,013 (5/11) | −0,057 (5/11) | 27 % / −96 % |

**Lectura:**
- **Indicios a favor de PPO en 2 de 3 entornos:**
  - en bottleneck, entorno en el que no se entrenó (transferencia), supera al estático sin ajuste y a la referencia ajustada;
  - en acrobot-swingup supera al estático y a AP en 8 de 11 semillas.
- **En pendulum-meta empeora**, en línea con la caída del score durante el entrenamiento.
- **Ninguna diferencia es significativa tras Holm.** La variabilidad entre semillas es alta, sobre todo en pendulum-meta
  (DE 0,15).
- Los márgenes c2 − c0 en las semillas de evaluación (+0,026 a +0,047) son menores que en la prueba de margen.

**Política de 1408 corridas** (salida en `analisis_comparacion_e1_1400.txt`):

| Entorno | PPO (800) | PPO (1408) | PPO(1408) − c0 | PPO(1408) − AP |
|---|---|---|---|---|
| bottleneck (transferencia) | 0,346 | 0,303 | +0,013 (6/11, p 0,70) | −0,011 |
| acrobot-swingup | 0,548 | 0,532 | −0,007 (6/11, p 0,58) | +0,007 (7/11) |
| pendulum-meta | 0,287 | 0,369 | +0,037 (7/11, p 0,76) | +0,024 |

**Conclusión:**
- Las diferencias cambian de signo entre la política de 800 y la de 1408 corridas, y ninguna es significativa, ni siquiera sin
  corregir.
- **Con 11 semillas no se distingue un efecto de PPO (plan B) sobre el estático sin ajuste ni sobre AP en E1.** Las diferencias
  observadas están dentro de la variabilidad entre semillas.
- AP tampoco difiere del estático.
- El agente sí aprende una política que depende del estado ([16](16_ppo_planB_e1.md)), pero su efecto sobre el resultado final
  no supera el ruido.

**Límites:**
- 11 semillas (lo que fija la propuesta), con DE entre semillas de 0,04–0,17;
- márgenes chicos (c2 − c0 = 0,026–0,047): ni siquiera el ajuste con Optuna se distingue con claridad del punto de partida en
  estas semillas;
- un solo entrenamiento de PPO (una semilla del agente);
- presupuesto de 75k (en bottleneck, NEAT sigue mejorando hasta 300k, [08](08_neat_300k.md)).

## Datos y reproducción

- `~/tesis/runs/comparacion_final/`: `preparar.py`, `lanzar.py` (planificador, tope compartido de 576 hebras), `jobs.json`,
  resultados en `<entorno>/<método>/seed_XX`.
