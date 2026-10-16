# Documentación del trabajo de tesis

Registro ordenado de todo lo realizado, pensado para la escritura de la tesis:
- **Bitácora semanal** (`bitacora/`): qué se hizo cada semana, qué se decidió y por qué, y qué quedó pendiente.
- **Fichas de experimentos** (`experimentos/`): una por experimento, con pregunta, diseño, resultados, conclusión, datos y
  capítulo de la tesis al que alimenta.
- **Documentos de diseño** (`*.md` de esta carpeta): notas de trabajo extensas. La bitácora y las fichas los citan por sección.

## Bitácora por semana

| Semana | Fechas | Fase | Reunión de avance | Fichas |
|---|---|---|---|---|
| [S01](bitacora/S01_2026-09-21.md) | 09-21 → 09-27 | Base técnica: simulador, NEAT, AP, primeros entornos | — | [01](experimentos/01_nucleo_pilotos_calibracion.md) |
| [S02](bitacora/S02_2026-09-28.md) | 09-28 → 10-04 | Líneas base preliminares → selección de entornos | 09-29 | [02](experimentos/02_estatico_vs_ap_preliminar.md), [03](experimentos/03_barridos_a_b.md), [04](experimentos/04_referencia_rl.md), [05](experimentos/05_barrido_largo_eleccion_entornos.md) |
| [S03](bitacora/S03_2026-10-05.md) | 10-05 → 10-11 | Elección de entornos → sonda de P → NEAT ajustado | 10-06 | [05](experimentos/05_barrido_largo_eleccion_entornos.md), [06](experimentos/06_sonda_p.md), [07](experimentos/07_optuna_neat.md) |
| [S04](bitacora/S04_2026-10-12.md) | 10-12 → 10-18 | Catálogo ampliado + agente PPO → prueba de margen | 10-13 | [08](experimentos/08_neat_300k.md) a [28](experimentos/28_fases_pmeta.md) |

## Fichas de experimentos

| # | Experimento | Semana | Estado |
|---|---|---|---|
| [01](experimentos/01_nucleo_pilotos_calibracion.md) | Núcleo NEAT + SNN: XOR, pilotos y calibración | S01 | terminado (exploratorio) |
| [02](experimentos/02_estatico_vs_ap_preliminar.md) | Estático frente a AP: primeras curvas | S02 | terminado (exploratorio) |
| [03](experimentos/03_barridos_a_b.md) | Barridos A y B: NEAT en todos los entornos de RLtools | S02 | terminado |
| [04](experimentos/04_referencia_rl.md) | Referencia RL y score normalizado | S02–S03 | terminado |
| [05](experimentos/05_barrido_largo_eleccion_entornos.md) | Barrido largo y elección de entornos | S02–S03 | terminado |
| [06](experimentos/06_sonda_p.md) | Sonda de P: ¿importa la mezcla de operadores? | S03 | terminado |
| [07](experimentos/07_optuna_neat.md) | Ajuste de hiperparámetros de NEAT con Optuna | S03 | terminado |
| [08](experimentos/08_neat_300k.md) | NEAT ajustado con 300k evaluaciones | S04 | bottleneck en curso |
| [09](experimentos/09_sonda_operadores_etapa1.md) | Sonda de operadores, etapa 1 | S04 | terminado |
| [10](experimentos/10_ppo_sintetico.md) | PPO en un AOS sintético | S04 | terminado |
| [11](experimentos/11_neat_como_entorno.md) | NEAT como entorno de RL y entrenador PPO | S04 | terminado |
| [12](experimentos/12_piloto_ppo_acrobot.md) | Piloto de PPO sobre NEAT real | S04 | terminado |
| [13](experimentos/13_prueba_margen.md) | Prueba de margen para el AOS | S04 | terminado |
| [14](experimentos/14_rendimiento.md) | Rendimiento: simulador y paralelización | S04 | aplicado y verificado |
| [15](experimentos/15_margen_intensidades.md) | Prueba de margen de intensidades (plan B) | S04 | terminado |
| [16](experimentos/16_ppo_planB_e1.md) | Entrenamiento de PPO del plan B (E1) | S04 | terminado |
| [17](experimentos/17_comparacion_e1.md) | Comparación final en E1 | S04 | terminado |
| [18](experimentos/18_analisis_efectos_acciones.md) | Análisis de los efectos de las acciones del agente | S04 | terminado |
| [19](experimentos/19_regla_con_estado.md) | Prueba de la regla con estado | S04 | terminado |
| [20](experimentos/20_ppo_v2.md) | PPO versión 2: un agente por entorno con estado SNN | S04 | en curso |
| [21](experimentos/21_sonda_final.md) | Sonda aleatorizada sobre el resultado final y confirmación en acrobot-swingup | S04 | en curso |
| [22](experimentos/22_operadores_actividad.md) | Operadores dirigidos por la actividad de la SNN | S04 | terminado |
| [23](experimentos/23_ppo_residual.md) | PPO residual sobre una regla con estado | S04 | pausado (evaluación pendiente) |
| [24](experimentos/24_dosis.md) | Dosis: regla con estado con el doble de empuje | S04 | terminado |
| [25](experimentos/25_margen_pmeta_p.md) | ¿El margen se alcanza moviendo solo P? (pendulum-meta y acrobot-swingup) | S04 | terminado |
| [26](experimentos/26_confirmacion_acrobot.md) | Confirmación con potencia de la regla con estado en acrobot-swingup | S04 | terminado |
| [27](experimentos/27_ppo_v2b.md) | PPO v2b en pendulum-meta (pocas decisiones grandes) contra c0 y AP | S04 | pausado (AP terminado) |
| [28](experimentos/28_fases_pmeta.md) | ¿El momento importa en pendulum-meta? Regla por fases contra la mejor P fija | S04 | terminado |

## Mapa a los capítulos de la tesis

| Capítulo | Contenido | Fuentes |
|---|---|---|
| 1. Introducción | Motivación, problema, objetivos. Evidencia medida de la motivación: AP persigue operadores neutros; una P fija no sirve a todos los entornos | `propuesta.md` §1 y §4 · [01](experimentos/01_nucleo_pilotos_calibracion.md), [06](experimentos/06_sonda_p.md), [09](experimentos/09_sonda_operadores_etapa1.md) |
| 2. Marco teórico y estado del arte | SNN de Izhikevich; NEAT; AOS sin estado y con estado (taxonomía de Pei et al. 2025); PPO; operadores de mutación en la literatura | `snn-simulator.md` · `diseno_neat_ap.md` §15.1 · `diseno_ppo.md` §1.4 · `operadores_candidatos.md` §1–4 · `propuesta.md` §2 |
| 3. Metodología | Implementación de NEAT + SNN y del evaluador (CRN, rotación); selección de entornos y score; líneas base ajustadas; catálogo de operadores; MDP y agente PPO; escenarios E1/E2 y diseño pareado | [01](experimentos/01_nucleo_pilotos_calibracion.md), [03](experimentos/03_barridos_a_b.md), [04](experimentos/04_referencia_rl.md), [05](experimentos/05_barrido_largo_eleccion_entornos.md), [07](experimentos/07_optuna_neat.md), [09](experimentos/09_sonda_operadores_etapa1.md), [10](experimentos/10_ppo_sintetico.md), [11](experimentos/11_neat_como_entorno.md), [13](experimentos/13_prueba_margen.md) · `diseno_neat_ap.md` §14–17 · `diseno_ppo.md` §2–4 |
| 4. Resultados | Espacio para adaptar (sondas, prueba de margen); NEAT estático ajustado y su techo; comparación final estático / AP / PPO (pendiente) | [06](experimentos/06_sonda_p.md), [07](experimentos/07_optuna_neat.md), [08](experimentos/08_neat_300k.md), [09](experimentos/09_sonda_operadores_etapa1.md), [13](experimentos/13_prueba_margen.md) |
| 5. Discusión | Límites de NEAT + SNN (interfaz, techo de acrobot-swingup, piso de l2f); crédito inmediato frente a retardado; amenazas a la validez (referencias, binarios, semillas) | [02](experimentos/02_estatico_vs_ap_preliminar.md), [04](experimentos/04_referencia_rl.md), [06](experimentos/06_sonda_p.md), [08](experimentos/08_neat_300k.md), [09](experimentos/09_sonda_operadores_etapa1.md), [11](experimentos/11_neat_como_entorno.md) |
| Anexos | Pilotos y calibración, verificación bit a bit, tablas completas | [01](experimentos/01_nucleo_pilotos_calibracion.md), [02](experimentos/02_estatico_vs_ap_preliminar.md), [11](experimentos/11_neat_como_entorno.md), [12](experimentos/12_piloto_ppo_acrobot.md) · `resultados_sonda_p.md` · `entornos_rltools.md` §7–8 |

## Convenciones

**Score**
- σ = (J − trivial) / (RL − trivial), donde J es el retorno del test final con 100 episodios:
  - 0 = trivial (lo mejor entre la política aleatoria y la acción cero);
  - 1 = referencia RL evaluada con los mismos episodios.
- Referencias oficiales: `~/tesis/runs/ref_rl/score_v2_s1-10.json`. Cada ficha indica si usó otra.

**Semillas de NEAT**

| Uso | Semillas |
|---|---|
| Ajuste con Optuna | 101–103 |
| Sondas | 201–205 |
| Entrenamiento de PPO | 1001+ |
| Validación de checkpoints | 2001+ |
| Evaluación final | 1–11 (nunca vistas al ajustar ni al entrenar) |

**Streams por corrida**

| Uso | Stream |
|---|---|
| Entrenamiento con rotación | 1000 + generación |
| Validación | 1 |
| Test final | 2 |
| Recompensa del agente | 3 |

**Protocolo**
- Rotación de episodios de entrenamiento (`--resample-train`) desde el 2026-10-09.
- 10 episodios de entrenamiento y 10 de validación.
- Presupuesto común de 75k evaluaciones.

**Prudencia con los resultados**
- Cada experimento corre con un **binario congelado**, identificado por sha.
- Se usa candado y marca por corrida, y los lanzamientos son reanudables.
- Toda opción nueva se verifica: sin la opción, el resultado es idéntico al binario anterior, y hay un control negativo.

**Datos**
- Los resultados viven fuera de git, en `~/tesis/runs/<experimento>/`.
- Configuraciones en `configs/`, lanzadores en `scripts/`, análisis en `analysis/` y en la carpeta de cada experimento.

## Cómo mantener esta documentación

- **Al cerrar cada sesión de trabajo:** actualizar la página de la semana en curso (qué se hizo, decisiones con su motivo,
  pendientes) y la ficha de cada experimento que cambió de estado.
- **Al lanzar un experimento:** crear su ficha desde la plantilla [`experimentos/_plantilla_ficha.md`](experimentos/_plantilla_ficha.md),
  con la pregunta, el diseño y las reglas de decisión **antes** de ver resultados.
- **Semana nueva:** crear su página desde [`bitacora/_plantilla_semana.md`](bitacora/_plantilla_semana.md) y agregarla a la
  tabla de arriba.
- **Redacción neutra:** "se decidió", "reunión de avance"; sin nombres de personas ni detalles personales.
