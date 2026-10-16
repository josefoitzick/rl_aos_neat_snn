# 06 · Sonda de P: ¿importa la mezcla de operadores?

| | |
|---|---|
| Semana | S03 (2026-10-07 → 09) |
| Estado | terminado |
| Capítulo de la tesis | Resultados: análisis previo del espacio de adaptación · Discusión |

## Pregunta

Con los 3 operadores de Stanley (mutar pesos, agregar conexión, agregar nodo):
1. ¿Importa P, la probabilidad de cada operador?
2. ¿Un P que cambia a mitad de la corrida supera a cualquier P fijo? Eso justificaría un AOS **con estado**.
3. ¿Sobra algún operador (ablación)?

## Diseño

- **10 configuraciones de P**, como (pesos, conexión, nodo):

  | Configuración | P |
  |---|---|
  | Stanley | 0,8 / 0,05 / 0,03 |
  | uniforme | 1 / 1 / 1 |
  | pesos | 0,98 / 0,01 / 0,01 |
  | estructura | 0,5 / 0,25 / 0,25 |
  | nodos | 0,6 / 0,1 / 0,3 |

  A eso se suman dos cambios en la generación 250 (estructura → pesos y pesos → estructura) y tres ablaciones de Stanley (sin
  conexión, sin nodo, sin pesos).
- **Corridas:** 3 entornos (bottleneck, acrobot-swingup, l2f) × 2 protocolos (fijo, rotación) × 10 semillas × 500 generaciones
  = 600 corridas.
- **Opciones nuevas de `neat_control`:** `--p`, `--p2`, `--p-switch`.
- **Verificación previa:**
  - sin las opciones, idéntico al binario del barrido largo en 10 entornos;
  - `--p` con la P de Stanley da lo mismo que sin `--p`;
  - el cambio ocurre en la generación indicada;
  - la ablación usa el operador 0 veces.
- **Binario congelado** `~/tesis/runs/sonda_p_bin` (sha `245893d0e9eb3b68`).
- **Control posterior:** Stanley con episodios fijos reproduce bit a bit las primeras 500 generaciones del barrido largo (30/30
  semillas).
- **Estadística:** Kruskal–Wallis entre los 7 P sin ablación; Mann–Whitney contra Stanley con corrección de Holm.

## Resultados

Score del test final con el protocolo de rotación (media ± DE, 10 semillas; referencias de `score_v2_s1-10.json`):

| P | bottleneck | acrobot-swingup | l2f |
|---|---|---|---|
| Stanley | 0,27 ± 0,06 | 0,48 ± 0,04 | 0,09 ± 0,03 |
| uniforme | 0,24 ± 0,08 | 0,52 ± 0,03 | 0,11 ± 0,04 |
| pesos | **0,32 ± 0,05** | 0,49 ± 0,04 | 0,10 ± 0,03 |
| estructura | 0,26 ± 0,06 | 0,50 ± 0,04 | 0,09 ± 0,03 |
| nodos | 0,29 ± 0,09 | 0,51 ± 0,04 | 0,10 ± 0,06 |
| pesos → estructura (gen. 250) | 0,32 ± 0,05 | 0,49 ± 0,05 | 0,11 ± 0,04 |
| sin pesos | **0,08 ± 0,10** | 0,53 ± 0,06 | 0,10 ± 0,03 |

| Entorno | ¿Importa P? (Kruskal–Wallis, fijo / rotación) | Cambio a mitad contra mejor P fijo | Ablaciones |
|---|---|---|---|
| bottleneck | p = 0,065 / 0,022. Mucha estructura perjudica (uniforme 0,13 con episodios fijos) y más pesos ayuda | No lo supera (0,32 = 0,32) | Sin pesos → ~0 (p Holm 0,015 / 0,009). Sin conexión y sin nodo no cambian |
| acrobot-swingup | p = 0,98 / 0,36. Todos entre 0,43 y 0,53 | No | Ninguna cambia, ni siquiera sin pesos o sin nodo (26 conexiones) |
| l2f | p = 0,97 / 0,93. Todos entre 0,02 y 0,11 (efecto piso) | No | Ninguna cambia |

## Conclusión y límites

- Solo en bottleneck la mezcla de operadores mueve el resultado, y el efecto se captura con un **P fijo cargado a pesos**.
  **Con estos 3 operadores** no hay evidencia de que cambiar de estrategia a mitad de la corrida ayude.
- En acrobot-swingup, el techo (~0,45–0,5) no depende de P ni del tamaño de la red: con 26 o con 419 conexiones el resultado es
  el mismo. El límite parece estar fuera de los operadores (interfaz de la SNN, perturbación de pesos demasiado gruesa o
  hiperparámetros de NEAT).
- En l2f, NEAT no progresa con ningún P: la validación queda plana entre las generaciones 250 y 500.
- Quitar la mutación de pesos no cambia nada en acrobot-swingup ni en l2f. Eso sugiere que el operador de pesos actual (perturbar
  todos los pesos con potencia 5) no hace ajuste fino útil.
- **Consecuencias:**
  1. Ajustar primero los hiperparámetros de NEAT ([07](07_optuna_neat.md)).
  2. Ampliar el catálogo con operadores de otra naturaleza: pesos finos, poda, neurona e interfaz ([09](09_sonda_operadores_etapa1.md)).
- **Límites:** solo los 3 operadores de Stanley; 500 generaciones; hiperparámetros sin ajustar.

## Datos y reproducción

- `~/tesis/runs/sonda_p` (+ `_bin`, `_logs`, `_verif`); `analisis_sonda_p.{md,json}`.
- `configs/sonda_p.json`; `scripts/lote_sonda_p.sh`; `scripts/analisis_sonda_p.py` (entorno conda `tesis-analisis`).
- Tablas completas por protocolo: `docs/resultados_sonda_p.md`.

## Documentos relacionados

- `docs/diseno_neat_ap.md` §16 (diseño) y §16.1 (resultados).
