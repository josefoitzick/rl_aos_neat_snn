# 11 · NEAT como entorno de RL y entrenador PPO (pasos 2 y 3 del agente)

| | |
|---|---|
| Semana | S04 (2026-10-13 → 14) |
| Estado | terminado |
| Capítulo de la tesis | Metodología: implementación del AOS con estado · Anexo: verificación |

## Pregunta

1. ¿Corre el NEAT que ve el agente exactamente igual que el NEAT de las líneas base (`neat_control`)? Si no, la comparación no
   sería justa.
2. ¿Entrena PPO de punta a punta sobre NEAT real?
3. ¿Es exacto reanudar desde un checkpoint?
4. ¿Se puede evaluar la política congelada con el mismo protocolo que las líneas base?

## Diseño

**Paso 2: NEAT como entorno.**
- `NeatRun`: una corrida de NEAT dividida en T = 50 ventanas de decisión. Φ se mide en el stream 3, con 50 episodios.
- `NeatPool`: muchas corridas en paralelo.
- `FeatureExtractor`: 14 + 2K rasgos de estado (`docs/diseno_ppo.md` §2.3).
- `evaluatePopulations`: varias poblaciones en una sola región paralela.
- `rankCredit` compartido con `neat_control`.
- Las opciones de NEAT tienen los mismos nombres, valores por omisión y semántica que en `neat_control`.

**Paso 3: entrenador** (`aos_ppo_train`):
- PPO de RLtools sobre `NeatPool`, con la configuración final de [10](10_ppo_sintetico.md);
- K de 3 a 10 y 15;
- estandarización de observaciones con las 2 primeras iteraciones;
- checkpoint en tar;
- actor exportado como código;
- validación pareada contra P₀;
- registros `ppo.csv`, `corridas.csv` y `decisiones.csv`.

**Agregado el 2026-10-14:**
- `NeatPool` con una configuración por ranura, para mezclar entornos (transferencia);
- `--mezcla` para entrenar con dos entornos;
- modo `--evaluar DIR --semillas a-b`: corre la política congelada y escribe `generations.csv` y `summary.txt` como
  `neat_control`. Con `--politica-fija P` evalúa una P fija.

**Verificaciones:**
- **V1** (`scripts/verificar_neat_run.sh`): `aos_neat_check` frente a `neat_control`. Casos:
  - pendulum con la P de Stanley y con cambio de P;
  - pendulum con el catálogo de 15 operadores y crédito por rango;
  - acrobot-swingup #49, l2f #65 y bottleneck #96, con sus líneas de comando completas;
  - el camino paralelo (4 semillas).
- **V4** (`scripts/verificar_ppo_pendulum.sh`): pendulum con población 30, 64 corridas y 8 iteraciones. Se compara detener y
  reanudar con correr de corrido.
- **Tests unitarios** (`tests/neat_run_tests.cpp`): `NeatRun` contra el ciclo de `neat_control`; `NeatPool` contra corridas por
  separado; pool mixto (2 ranuras de pendulum + 2 de acrobot) contra 4 corridas por separado.
- **`--evaluar`** contra `neat_control` con P₀ en una semilla de acrobot.

## Resultados

- **V1:** idéntico bit a bit en `generations.csv` (salvo los segundos), test final y mejor validación, en los 7 casos. Controles:
  - el `neat_control` recompilado coincide con el binario congelado de la sonda (ae239e87…), así que los cambios en `src/neat`
    no alteran nada;
  - otra semilla da otro CSV, así que la comparación detecta diferencias.
- **V4:** entrena de punta a punta y **reanudar es exacto**: `ppo.csv`, `corridas.csv`, `decisiones.csv` y el actor
  exportado coinciden. La validación (PPO 0,338 contra P₀ 0,358) solo prueba el circuito: con 64 corridas no se espera
  aprendizaje.
- **Pool mixto:** idéntico bit a bit a las corridas por separado.
- **`--evaluar`:** idéntico bit a bit a `neat_control`.
- Tests: 88/88.

## Conclusión y límites

- PPO se entrena y se evalúa sobre **exactamente el mismo NEAT** con el que se compara.
- La evaluación final usa `aos_ppo_train --evaluar` en lugar de integrar el actor en `neat_control`.
- **Regla:** entrenar y evaluar con el mismo binario congelado. Con binarios distintos, la política dio resultados idénticos en
  solo 10/16 semillas; con el mismo binario, en 16/16. El redondeo de la red puede cambiar trayectorias.
- **Detalle técnico:** RLtools va envuelto en el namespace `ppo_rlt`. `libenvs` también lo incluye, y sin el envoltorio el
  enlazado choca (definición múltiple de `is_nan`).

## Datos y reproducción

- `~/tesis/runs/verif_neat_run.log`, `~/tesis/runs/ppo_pendulum_v4.log` (+ carpetas).
- Código:
  - `src/aos/neat_run.{hpp,cpp}`, `src/aos/features.{hpp,cpp}`;
  - `src/eval/evaluator.{hpp,cpp}`, `src/neat/selector.{hpp,cpp}`;
  - `src/experiment/opciones_neat.hpp`, `aos_neat_check.cpp`, `aos_ppo_train.cpp`.
- Tests: `tests/neat_run_tests.cpp`, `tests/ppo_tests.cpp`.

## Documentos relacionados

- `docs/diseno_ppo.md` §3 (NEAT como entorno de RLtools), §5.2 y §5.3 (resultados).
