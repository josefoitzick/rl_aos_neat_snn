# 05 · Barrido largo y elección de los entornos de trabajo

| | |
|---|---|
| Semana | S02–S03 (2026-10-04 → 07) |
| Estado | terminado |
| Capítulo de la tesis | Metodología: selección de entornos y protocolo de entrenamiento |

## Pregunta

1. ¿NEAT cierra la brecha con la referencia RL si se le da más presupuesto (500, 1000 y 2000 generaciones)?
2. ¿Cuánto del mal desempeño se debe al sobreajuste a 10 episodios fijos? Se compara el protocolo **fijo** con la
   **rotación** de episodios en cada generación.
3. Con eso, ¿qué entornos se eligen para estudiar el AOS?

## Diseño

- **Corridas:** 23 variantes (las que pasaron el filtro de [03](03_barridos_a_b.md), más l2f, l2f-tiny y flag) × {fijo, rot}
  × 10 semillas × 2000 generaciones = 440 corridas. ant aparte, con 1000 generaciones: 20 corridas.
- **Rotación** (`neat_control --resample-train`):
  - los episodios de entrenamiento de la generación g salen del stream 1000 + g y son comunes a toda la población;
  - el genoma final es el campeón de generación con mejor validación (stream 1);
  - el test final (stream 2) no interviene en la elección.

  Sin la opción, la salida es idéntica bit a bit a la del binario del barrido A.
- **Ejecución:** binario congelado `~/tesis/runs/barrido_largo_bin`. Semillas 1–5 primero (para la reunión del 2026-10-06) y
  6–10 después, por disponibilidad de hebras. Del 2026-10-04 15:10 al 10-07 18:15, sin errores.
- **Verificación:** 115 corridas `_fijo` coinciden con el barrido de 500 generaciones en sus primeras 500 (0 diferencias).

## Resultados

Score del test final (100 episodios; referencias de [04](04_referencia_rl.md)), 10 semillas:

| Variante | Fijo, 500 gen. | Fijo, 2000 | Rotación, 2000 | Δ rot − fijo |
|---|---|---|---|---|
| l2f | 0,02 | 0,02 | 0,09 | +0,07 |
| bottleneck | 0,25 | 0,33 | 0,31 | −0,02 |
| pendulum-meta (reactivo) | 0,41 | 0,51 | 0,54 | +0,03 |
| acrobot-swingup | 0,41 | 0,44 | 0,50 | +0,06 |
| pendulum-multitask | 0,48 | 0,82 | 0,85 | +0,03 |
| pendulum-position (memoria) | 0,29 | 0,34 | 0,36 | +0,01 |
| memory (memoria) | 0,50 | 0,59 | 0,56 | −0,04 |
| reacher | 0,73 | 0,69 | 0,92 | +0,24 |
| car-track | 0,93 | 1,00 | 1,17 | +0,17 |
| ant (1000 gen.) | 0,03 | 0,04 | 0,02 | −0,02 |

- **Más presupuesto ayuda poco**, salvo en pendulum-multitask (0,48 → 0,82). Ningún entorno cambia de grupo respecto del
  análisis con 5 semillas.
- **La rotación ayuda donde había sobreajuste:** l2f, acrobot-swingup, reacher y car-track. La validación con rotación es
  optimista (sesgo de hasta 0,22 en score; 0,19 en bottleneck y 0,05 en acrobot-swingup), así que **se reporta siempre el test
  final**.

## Elección de entornos (2026-10-07)

**bottleneck, acrobot-swingup y l2f.**
- Sin memoria: `--keep-state` queda fuera porque empeora a NEAT en varios péndulos y agrega una variable de diseño.
- Forman una escalera de complejidad de control:

  | Entorno | Observaciones | Control |
  |---|---|---|
  | acrobot-swingup | 6 | 1 agente |
  | bottleneck | 12 | 5 agentes |
  | l2f | 82 | 4 rotores |

- l2f está fuera de alcance (score 0,02 fijo / 0,09 rot), así que requiere una etapa de viabilidad. Criterio: NEAT estático
  ≥ ~0,2.
- **Respaldo:** pendulum-meta reactivo. El respaldo se activó el 2026-10-14 ([08](08_neat_300k.md), [13](13_prueba_margen.md)).
- Se descartaron:
  - ant: costo alto y score ~0,03;
  - los visuales: problema de representación.

## Conclusión y límites

- El protocolo de rotación se adoptó para todo lo que sigue (decisión del 2026-10-09).
- **Riesgo detectado al cerrar el análisis:** con la métrica "fracción de hijos que mejoran al padre", agregar nodo gana en
  todas las fases de los 3 entornos. La métrica favorece a los operadores casi neutros, así que no prueba que solo importe
  agregar nodos, pero tampoco muestra cambio de fase. Se decidió medirlo directamente con la sonda de P ([06](06_sonda_p.md)).

## Datos y reproducción

- `~/tesis/runs/barrido_largo`, `barrido_largo_ant` (+ `_bin`, `_logs`); `analisis_barrido_largo.{md,json}` (10 semillas) y
  `analisis_barrido_largo_s1-5.*`.
- `configs/barrido_largo.json`, `barrido_largo_ant.json`; `scripts/lote_barrido_largo.sh`, `lote_barrido_largo_ant.sh`.
- Análisis: `python3 scripts/analisis_barrido_largo.py 1-10`.
- Página de resultados de la reunión del 2026-10-06: `~/tesis/runs/reunion3/generar_pagina.py`.

## Documentos relacionados

- `docs/diseno_neat_ap.md` §15.4.1 (diseño y decisiones) y §16 (entornos elegidos).
