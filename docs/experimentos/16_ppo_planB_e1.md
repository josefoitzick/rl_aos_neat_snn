# 16 · Entrenamiento de PPO del plan B (escenario E1)

| | |
|---|---|
| Semana | S04 (lanzado el 2026-10-15 a las 02:24 con 348 hebras) |
| Estado | entrenamiento terminado (2026-10-15 05:35, 800 corridas, 2,8 h); evaluación en semillas 1–11 en [17](17_comparacion_e1.md) |
| Capítulo de la tesis | Resultados: AOS con estado |

## Pregunta

¿Aprende el agente PPO a controlar P y las intensidades de variación (plan B) desde un NEAT sin ajuste previo, con un
estado compacto?

## Diseño

**Código** (verificado en `~/tesis/runs/optim_verif/verificar_planB.sh`, sin fallas, 91/91 tests):
- `neat::Variacion` y `Population::setVariacion`;
- `NeatRun::reproduce(P, v)`;
- `aos::MapaAccion`: con acción 0 se obtienen exactamente P₀ y las intensidades base;
- estado compacto de 9 rasgos (`FeatureConfig::compacto`);
- `aos_ppo_train --estado compacto --accion ambas`.

El modo de siempre del entrenador es idéntico al de `fase_ppo_bin`. Binarios en `~/tesis/runs/fase_ppo_bin2`.

**Configuración del entrenamiento:**

| Elemento | Valor |
|---|---|
| NEAT | Configuración ajustada de cada entorno, pero con P uniforme sobre el catálogo de 10 e intensidades genéricas (5 / 0,02 / 0,25): el punto de partida c0 de [15](15_margen_intensidades.md) |
| Entornos | acrobot-swingup + pendulum-meta, mitad de las ranuras cada uno, para transferir después a bottleneck |
| Agente | Estado compacto (9 rasgos); acción "ambas" (10 de P + 3 de intensidades) |
| PPO | Configuración final ([10](10_ppo_sintetico.md)), 16 corridas en paralelo, T = 50 |
| Presupuesto | 800 corridas |
| Validación | Pareada contra la acción 0 (estático genérico) en 32 semillas por entorno |
| Hebras | Todas las libres al arrancar (máximo 440); el entrenamiento guarda puntos de control y se reanuda con `--reanudar` |

## Resultados

**Durante el entrenamiento** (2026-10-15 ~03:45, 320 corridas): score de entrenamiento por bloques de 40 corridas:

| Entorno | Bloques de 40 corridas |
|---|---|
| acrobot-swingup | 0,524 · 0,522 · 0,514 · 0,524 (estable) |
| pendulum-meta | 0,367 · 0,360 · 0,310 · 0,282 (en baja) |

En pendulum-meta, el agente sube de a poco la P de agregar nodo (0,104 → 0,128); las intensidades siguen cerca de la base.
Se deja seguir: la validación pareada contra la acción 0, al final, dirá si mejora o empeora cada entorno.

**Política aprendida:** P depende del momento de la corrida. En la primera mitad sube agregar conexión, ganancia de entrada y
borrar conexión (~0,12–0,13); en la segunda, perturbar la dinámica y el sesgo de las neuronas (~0,12–0,13). Las intensidades
quedan cerca de la base: potencia ~5, reasignar ~0,028 y sin cruce ~0,22–0,26.

**Validación pareada** (32 semillas por entorno, 2001–2032, contra la acción 0 = estático genérico):

| Entorno | PPO | Acción 0 | Δ (ee) | PPO gana |
|---|---|---|---|---|
| acrobot-swingup | 0,5210 | 0,5189 | +0,002 (0,007) | 16/32 |
| pendulum-meta | 0,3162 | 0,3241 | −0,008 (0,030) | 16/32 |

**Continuación hasta 1408 corridas** (reanudada a las 06:06 con 540 hebras; respaldo de las 800 en `salida_800`).
Validación pareada (32 semillas):

| Entorno | Δ con 800 corridas | Δ con 1408 corridas |
|---|---|---|
| acrobot-swingup | +0,002 (16/32) | +0,007 (ee 0,010; 17/32) |
| pendulum-meta | −0,008 (16/32) | −0,019 (ee 0,029; 19/32) |

**Resultado nulo:** con 800 corridas, el agente aprende una estrategia que depende del estado, pero no mejora el resultado frente
al punto de partida. La caída del score de entrenamiento en pendulum-meta no se confirma en la validación (Δ −0,008, dentro
del error).

## Datos y reproducción

- `~/tesis/runs/ppo_planB_e1/`:
  - `preparar.py` (genera `lanzar.sh`), `arrancar.sh` (espera hebras y que la prueba de margen no tenga pendientes);
  - `salida/`: `ppo.csv`, `corridas.csv` y `decisiones.csv`, con P, intensidades y observación de cada decisión.
