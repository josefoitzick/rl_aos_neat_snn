# 04 · Referencia RL y score normalizado

| | |
|---|---|
| Semana | S02 (2026-10-04); reevaluada en S03 (2026-10-07) |
| Estado | terminado |
| Capítulo de la tesis | Metodología: selección de entornos y métrica de desempeño |

## Pregunta

¿Qué tan lejos queda NEAT de un buen controlador en cada entorno? Hace falta una escala común entre entornos para decidir dónde
"NEAT lucha" y, después, para expresar las mejoras de los métodos de AOS.

## Diseño

- **Score** = (J − trivial) / (RL − trivial), donde J es el retorno del test final:
  - 0 = trivial;
  - 1 = referencia RL.
- **Grupos:**
  - score ≥ ~0,8–0,9: ancla;
  - 0,2–0,8: "NEAT lucha", objeto de estudio del AOS;
  - < 0,2: fuera de alcance.
- **Referencia RL**, en tres niveles:
  1. entrenadores del zoo de RLtools, con sus hiperparámetros, sin ajustar a favor ni en contra;
  2. configuraciones propias, copiando la más cercana del zoo y cambiando solo el entorno y el horizonte;
  3. óptimo analítico, solo en memory.
- **v2:** OpenBLAS y 10 semillas por entrenador. Se toma el mejor entrenador de cada entorno.
- **Comparación justa:** la política RL se exporta y se evalúa con **nuestro** evaluador, en los **mismos 100 episodios
  finales** que NEAT.
- **Controles:**
  - el entorno reproduce bit a bit un bucle escrito directamente con RLtools;
  - la biblioteca de la política reproduce los ejemplos del checkpoint;
  - las políticas cero y aleatoria reproducen el trivial;
  - el oráculo de memory da exactamente 0.
- **2026-10-07:** referencias y trivial reevaluados en los episodios de las semillas 1–10 (`score_v2_s1-10.json`). El cambio es
  < 2 % y el mejor entrenador es el mismo. **Estas son las referencias oficiales de la tesis.**

## Resultados

Score de NEAT estático con 500 generaciones (barridos A y B, 5 semillas; tabla completa en `docs/entornos_rltools.md` §8.3):

| Variante | NEAT | RL (entrenador) | Trivial | Score | Grupo |
|---|---|---|---|---|---|
| acrobot ±1 | −100,6 | −73,5 (SAC propio) | −500 | 0,94 | ancla |
| car-track | 599,7 | 641,4 (PPO propio) | 0 | 0,93 | ancla |
| pendulum | −260,3 | −153 (SAC del zoo) | −1252 | 0,90 | ancla |
| reacher | −13,6 | −3,3 (PPO del zoo) | −40,8 | 0,73 | lucha |
| pendulum-multitask | −722 | −156 (SAC-GRU propio) | −1249 | 0,48 | lucha |
| acrobot-swingup | −13,36 | −6,66 (SAC del zoo) | −18,03 | 0,41 | lucha |
| pendulum-meta (reactivo) | −876 | −329 (SAC-GRU del zoo) | −1249 | 0,41 | lucha |
| bottleneck | 15,26 | 61,74 (PPO del zoo) | 0 | 0,25 | lucha |
| ant | 1139 | 6536 (PPO del zoo) | 1000 | 0,03 | fuera |
| l2f | 4,79 | 30,55 (SAC propio) | 4,28 | 0,02 | fuera |

**Referencias oficiales** (`score_v2_s1-10.json`, semillas 1–10) de los entornos de trabajo:

| Entorno | Trivial | RL | Entrenador |
|---|---|---|---|
| bottleneck | 0 | 61,8567 | bottleneck-v0, PPO |
| acrobot-swingup | −18,04437 | −6,8019405 | acrobot-swingup-v0, SAC |
| l2f | 4,17605 | 30,610236 | l2f completo, SAC (configuración propia) |
| pendulum-meta (reactivo) | −1231,444 | −323,56979 | pendulum-multitask-v1, SAC-GRU (la clase del zoo es `PendulumMeta`) |
| pendulum | −1231,444 | −148,79341 | pendulum-v1, SAC |

## Conclusión y límites

- La memoria de la SNN empeora en los péndulos de observación parcial:
  - pendulum-meta: 0,14 con memoria contra 0,41 reactivo;
  - pendulum-velocity: 0,37 contra 0,59.

  RL con GRU sí la aprovecha.
- **Sesgo conocido:** el retorno que reporta el propio zoo no es comparable (aritmética float, penúltimo checkpoint, otro
  horizonte en l2f-tiny). Se usa siempre nuestra evaluación. Si sesga en algo, el RL sale algo peor y el score de NEAT algo
  **optimista**, que es la dirección conservadora.
- **moving-gate queda sin referencia:** RLtools solo entrena hyperdrone en GPU y el nodo de cómputo es solo CPU.
- Algunos análisis intermedios (sonda de operadores, figura de 300k, piloto de PPO) usaron las referencias de `score_v2.json`
  (semillas 1–5). La diferencia es < 2 %. Cada ficha indica qué referencia usa.

## Datos y reproducción

- `~/tesis/runs/ref_rl/`: `score_v2.json`, `score_v2_s1-10.json`, `evaluacion_v2.csv`, binarios y `SHA256SUMS`.
- Tubería en `scripts/ref_rl/`:
  - `compilar_zoo.sh`, `compilar_propias.sh`, `lanzar_zoo.sh`, `lanzar_ref.sh`, `compilar_politica.sh`, `evaluar_zoo.sh`,
    `resumen_ref.py`;
  - `src/ref_rl/` (configuraciones propias y oráculo de memory);
  - `src/experiment/rl_policy_shim.cpp`, `build/bin/rl_ref_eval`.

## Documentos relacionados

- `docs/entornos_rltools.md` §8 (tubería, verificaciones y scores).
- `docs/diseno_neat_ap.md` §15.4 (plan de comparación con referencia RL).
