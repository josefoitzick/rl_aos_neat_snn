# 03 · Barridos A y B: NEAT estático en todos los entornos de RLtools

| | |
|---|---|
| Semana | S02 (2026-10-03 → 04) |
| Estado | terminado |
| Capítulo de la tesis | Metodología: selección de entornos (primeros capítulos) |

## Pregunta

¿En qué entornos de RLtools aprende NEAT + SNN, en cuáles no supera lo trivial y cuáles quedan en una zona intermedia, donde
un AOS podría aportar?

## Diseño

- **33 variantes × 5 semillas × 500 generaciones = 165 corridas.**
  - NEAT estático con la P de Stanley, E = 10, especiación dinámica (10 especies objetivo) y reasignación 0,02.
  - Protocolo de entrenamiento con episodios fijos.
- **Inventario completo:** `docs/entornos_rltools.md` §1–4.
  - Entornos de memoria: con `--keep-state` (memoria de la SNN entre pasos) y, como control, con política reactiva.
  - Entornos visuales: cámara de 16×16. Escenas de entrenamiento: las casas 0 y 1. Escena de test: la casa 2, no vista.
- **Retorno trivial** por entorno: el máximo entre la política aleatoria y la acción cero (sonda de entornos).
- **Binarios congelados:** A `dfd3c0c1`, B `1a5d7d99`. Una prueba de equivalencia los certifica como un solo experimento.
- **Fechas:** A del 2026-10-03 21:14 al 10-04 05:58; B del 10-04 05:58 a 14:06. Sin errores ni corridas omitidas.
- RLtools fijado en la rama `render` @ `6bb2fda`, que incluye reacher y hyperdrone.

## Resultados

Selección de la tabla completa de `docs/entornos_rltools.md` §7. Para cada variante: trivial; test final (media ± DE);
z = (test − trivial)/DE; brecha = (entrenamiento − test)/(entrenamiento − trivial), donde 0 = generaliza y ≥ 1 = pierde todo;
y g90 = generación en que se alcanza el 90 % de la mejora.

| Variante | Trivial | Test ± DE | z | Brecha | g90 |
|---|---|---|---|---|---|
| pendulum | −1301 | −260 ± 61 | 17 | 0,07 | 157 |
| car-track | 0 | 600 ± 52 | 12 | 0,29 | 139 |
| acrobot ±1 | −135,6 | −100,6 ± 3,6 | 9,6 | 0,32 | 71 |
| acrobot-swingup | −17,95 | −13,36 ± 1,2 | 3,8 | 0,32 | 151 |
| bottleneck | 0 | 15,3 ± 9,1 | 1,7 | 0,47 | 329 |
| pendulum-meta (reactivo) | −1301 | −876 ± 130 | 3,4 | 0,40 | 248 |
| pendulum-multitask | −1301 | −722 ± 190 | 3,0 | 0,33 | 348 |
| l2f | 3,457 | 4,79 ± 1,7 | 0,76 | 0,86 | 208 |
| ant | 999,2 | 1139 ± 120 | 1,2 | 0,37 | 328 |
| flag (reactivo) | 230 | 315 ± 120 | 0,73 | 0,92 | 262 |
| reacher-visual | −35 | −35,8 ± 6,3 | −0,13 | 1,0 | 57 |

**Grupos:**
- **NEAT resuelve o se estanca temprano (anclas):** pendulum, acrobot ±1, car, car-track y reacher.
- **NEAT aprende con dificultad (candidatos):**
  - los péndulos de observación parcial o multitarea;
  - memory (con memoria);
  - bottleneck, acrobot-swingup y ant;
  - hyperdrone-moving-gate reactivo, el único visual que aprende;
  - l2f y l2f-tiny, con brecha alta.
- **No supera lo trivial:**
  - los reacher visuales, l2f-visual y hyperdrone-target-frame: el problema parece ser la representación (256 píxeles como
    entradas directas, de 3000 a 12 000 conexiones), no los operadores;
  - flag: sobreajuste total.

**Memoria de la SNN (`--keep-state`):**
- imprescindible en memory;
- ayuda en pendulum-position;
- **empeora** en pendulum-velocity, pendulum-meta, flag, moving-gate y los reacher visuales.

## Conclusión y límites

- Filtro para la etapa siguiente: los visuales (salvo moving-gate reactivo) y flag quedan fuera, con el motivo medido y reportado.
- Con la P de Stanley fija, agregar nodo tiene la mayor tasa de mejora en casi todas las ventanas. Estas corridas **no alcanzan**
  para medir la no estacionariedad de los operadores: eso se mide después, con sondas de P fija ([06](06_sonda_p.md)).
- **Límites:**
  - 5 semillas;
  - 500 generaciones;
  - episodios de entrenamiento fijos (sobreajuste por diseño; ver [05](05_barrido_largo_eleccion_entornos.md)).

## Datos y reproducción

- `~/tesis/runs/barrido_etapa_a`, `barrido_etapa_b` (+ `_bin`, `_logs`), `equivalencia_binarios`, `env_probe_v2*`;
  `analisis_barrido_ab.json`.
- `configs/barrido_etapa_a.json`, `barrido_etapa_b.json`; `scripts/analisis_barrido_ab.py`; `scripts/equivalencia_binarios.sh`.

## Documentos relacionados

- `docs/entornos_rltools.md` §1–7 (inventario, clasificación, implementación, visuales y resultados).
- `docs/diseno_neat_ap.md` §15.2 (plan de selección de entornos).
