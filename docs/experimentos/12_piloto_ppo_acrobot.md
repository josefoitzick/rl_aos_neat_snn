# 12 · Piloto de PPO sobre NEAT real (acrobot-swingup)

| | |
|---|---|
| Semana | S04 (2026-10-14) |
| Estado | terminado (2026-10-15 01:30) |
| Capítulo de la tesis | Metodología: costo y viabilidad del entrenamiento · Anexo |

## Pregunta

1. ¿Corre el circuito completo sobre NEAT real (entrenamiento, checkpoints, exportación y validación) sin errores?
2. ¿Cuánto cuesta una corrida de entrenamiento?
3. ¿Hay algún indicio de aprendizaje temprano?

Por lo visto en [10](10_ppo_sintetico.md), con 96 corridas **no se espera** una mejora: este piloto mide circuito y costo.

## Diseño

- **Entorno y configuración:**
  - acrobot-swingup con la configuración #49 de Optuna: población 130 × 577 generaciones, 75k evaluaciones, rotación;
  - catálogo de Stanley (K = 3), P₀ = (0,520; 0,407; 0,074);
  - observaciones de 14 + 2K = 20 rasgos.
- **PPO:**
  - configuración final de [10](10_ppo_sintetico.md): 16 corridas en paralelo, T = 50;
  - presupuesto de 96 corridas de entrenamiento (12 iteraciones);
  - 150 hebras;
  - validación pareada contra P₀ con `--validar 32`.
- **Binario** congelado en `~/tesis/runs/ppo_piloto_acrobot/bin` (sha `3b54b4ed…`). Se compiló antes de cambiar las
  referencias del entrenador a `score_v2_s1-10.json`, así que usa las de `score_v2.json` (trivial −18,03; RL −6,656). Afecta
  solo la escala del score.
- Lanzado el 2026-10-14 a las 14:35, en las hebras que dejaban libres las otras corridas.

## Resultados

**Entrenamiento** (`salida/ppo.csv`):

| Iteración | Corridas | Score medio de entrenamiento | KL | P media (pesos / conexión / nodo) |
|---|---|---|---|---|
| 2 | 16 | 0,553 | 0,013 | 0,516 / 0,401 / 0,084 |
| 8 | 64 | 0,559 | 0,022 | 0,496 / 0,426 / 0,078 |
| 12 | 96 | 0,549 | 0,038 | 0,511 / 0,410 / 0,079 |

- El score de entrenamiento se mantiene plano en ~0,55 y la P media queda cerca de P₀: no hay señal de aprendizaje, como se
  esperaba con 96 corridas.
- El KL está en 0,01–0,04: las actualizaciones son estables.
- **Costo:** ~3000 s por tanda de 16 corridas (2 iteraciones) con 150 hebras, es decir, unas **7–8 hebra-hora por corrida**
  de entrenamiento a 75k en acrobot-swingup.
- **Validación** (32 semillas pareadas, 2001–2032): PPO 0,5555 contra P₀ 0,5433.
  - Δ = +0,012 (ee 0,009); PPO gana en 19 de 32 semillas.
  - No es significativo, como se esperaba con 96 corridas.
  - Confirma que el entrenamiento no degrada la política: no hay deriva negativa.

## Conclusión y límites

- El circuito corre de punta a punta sobre NEAT real. El costo medido permite planificar los entrenamientos (≥ 400 corridas
  por entorno).
- **Límites:**
  - 96 corridas, muy por debajo del presupuesto necesario;
  - catálogo de Stanley, que deja poco margen ([06](06_sonda_p.md)).

  El entrenamiento real se hará con el catálogo K = 10 y P₀ uniforme (escenario E1, [13](13_prueba_margen.md)).

## Datos y reproducción

- `~/tesis/runs/ppo_piloto_acrobot/`:
  - `lanzar.sh` (línea de comando completa), `salida.log`;
  - `salida/`: `ppo.csv`, `corridas.csv`, `decisiones.csv`, `estado.tar` y `actor_final/checkpoint.h`.
- Evaluación posterior: `aos_ppo_train --evaluar salida --semillas a-b`, con **el mismo binario** ([11](11_neat_como_entorno.md)).

## Documentos relacionados

- `docs/diseno_ppo.md` §4 (entrenamiento) y §6 (pre-chequeo).
