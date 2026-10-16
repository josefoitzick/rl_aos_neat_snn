# 23 · PPO residual: el agente aprende correcciones sobre una regla con estado

| | |
|---|---|
| Semana | S04 (lanzado el 2026-10-15 a las 17:47) |
| Estado | pausado en la iteración 14 (560 corridas); evaluación pendiente |
| Capítulo de la tesis | Metodología: diseño del agente (aprendizaje residual) · Resultados: AOS con estado |

## Pregunta

PPO desde cero (versión 2, [20](20_ppo_v2.md)) no mostró aprendizaje en sus primeras 432 corridas: P elegida cerca de la
uniforme y score de entrenamiento al nivel de c0. Las reglas escritas a mano con el estado sí muestran efecto (en
acrobot-swingup, +0,012 en 60 semillas, 39/60; [19](19_regla_con_estado.md), [21](21_sonda_final.md)).

¿Un agente que parte de la mejor regla y aprende correcciones sobre ella supera a la regla y al estático?

## Diseño

**Aprendizaje residual** (Silver et al. 2018, *Residual Policy Learning*; Johannink et al. 2019): la acción que se decodifica a
P es la suma de la acción del agente y el empuje de la regla base para el estado actual. Con acción 0 del agente, la decisión
es exactamente la de la regla, así que el agente parte del desempeño de la regla y la exploración ocurre a su alrededor.

- `aos_ppo_train --regla-base <archivo>` (entrenamiento, validación y `--evaluar`). La validación interna compara contra la
  regla base en las mismas semillas.
- **Verificación** (`~/tesis/runs/optim_verif/residual/`):
  - sin `--regla-base`, el entrenamiento es idéntico al de `fase_ppo_bin4` (ppo.csv sin segundos, corridas, decisiones y
    validación);
  - con el actor en 0 exacto (`--ultima-capa 0`), PPO residual da exactamente el score de la regla (Δ = 0 en 16 semillas);
  - control negativo: en esas semillas, la regla difiere de la P uniforme;
  - entrenamiento corto: el empuje se aplica (despertar_salidas 0,048 → 0,175 con salidas mudas).
- **Binarios:** `fase_ppo_bin5` (`aos_ppo_train` dd1cd0d0…; `neat_control` igual a `fase_ppo_bin4`). Tests 96/96.
- **Configuración prevista:** catálogo de 14 (10 + F1–F4), estado SNN, T = 10, σ inicial 0,5 (más chica que la versión 2,
  porque parte de una regla buena), un agente por entorno, ≥ 800 corridas.

**Lanzamiento (2026-10-15 17:47, acrobot-swingup):** la ficha 22 dio nulo con el catálogo de 14, así que se usa el catálogo de
10 (opciones del brazo c0_10 de la ficha 22, iguales a las de la ficha 19). Regla base: la regla por momento de
[19](19_regla_con_estado.md) (×1). σ inicial 0,5; 800 corridas (semillas 1001–1800); 256 hebras; validación interna de 32 corridas
contra la regla base. Salida en `~/tesis/runs/ppo_v3/acrobot-swingup_cat10/`.

**Reglas fijadas antes de mirar:** la política final se evalúa (`--evaluar --regla-base`) en las semillas 401–500 de
[26](26_confirmacion_acrobot.md), pareada con sus brazos c0 y regla (Wilcoxon, Holm entre las dos comparaciones):
1. **PPO residual − c0 > 0, p Holm < 0,05:** el AOS con estado aprendido le gana al estático sin ajuste.
2. **PPO residual − regla > 0, p Holm < 0,05:** el agente mejora la regla escrita a mano.

## Resultados

**Entrenamiento** (acrobot-swingup, 17:47 → 00:18; pausado en el punto de control de la iteración 14, 560 corridas; respaldo en
`pausa_iter14/`, `decisiones.csv` verificado; se reanuda exacto con `lanzar.sh --reanudar`):
- Score de entrenamiento 0,517–0,545 por iteración, en el rango de la regla base.
- **Aprende correcciones consistentes** sobre la regla (desviación media de log P respecto de la regla, al inicio de la corrida;
  la exploración sola promedia ~±0,05):

  | Operador | Iteraciones 1–2 | Iteraciones 8–11 |
  |---|---|---|
  | borrar conexión | +0,04 | +0,14 |
  | ganancia de entrada | +0,05 | +0,16 |
  | mutar pesos | +0,07 | +0,10 |
  | pesos finos / dinámica | −0,05 | −0,10 |

  Suaviza los empujes de la regla (los acerca a la uniforme) y sube la ganancia de entrada, coherente con la dosis
  ([24](24_dosis.md): ×2 empeora).
- **Evaluación pendiente:** la política del punto de control en las semillas 401–500, pareada con c0 y la regla de
  [26](26_confirmacion_acrobot.md) (reglas de arriba).

## Datos y reproducción

- `~/tesis/runs/ppo_v3/preparar.py <entorno> <regla base> <corridas> <hebras> [std0]` (genera `<entorno>/lanzar.sh`).

## Documentos relacionados

- [19](19_regla_con_estado.md) · [20](20_ppo_v2.md) · [21](21_sonda_final.md) · [22](22_operadores_actividad.md).
