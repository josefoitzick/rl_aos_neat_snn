# 13 · Prueba de margen para el AOS (plan revisado)

| | |
|---|---|
| Semana | S04 (lanzada el 2026-10-14 a las 21:35) |
| Estado | terminado (2026-10-15 02:28, 80/80 corridas) |
| Capítulo de la tesis | Metodología: escenarios de comparación y diseño estadístico · Resultados: margen para adaptar |

## Pregunta

Antes de gastar ~1 día de clúster en entrenar PPO: **¿hay margen para que un AOS mejore a NEAT?**

Hasta aquí solo se había medido el escenario en que la P estática ya está ajustada con Optuna (~110 trials por entorno). Es el
escenario donde un AOS tiene menos que aportar, y el que la motivación de la tesis cuestiona. La propuesta (§2.3) dice que en
SNN "fijar probabilidades de mutación por prueba y error deja de ser razonable".

## Diseño

**Escenarios de la comparación final** (decididos el 2026-10-14):

| | E1 · Sin ajuste previo | E2 · Con ajuste previo |
|---|---|---|
| P inicial de todos los métodos | Uniforme sobre el catálogo ampliado | P₀ ajustada con Optuna |
| Qué mide | Si el AOS recupera, sin el costo del ajuste, lo que da ajustar, y si el estado ayuda (PPO frente a AP) | Si el AOS suma algo sobre un estático ya ajustado. Se reporta aunque dé nulo |

**Catálogo K = 10:**
- mutate_weights, add_connection, add_node;
- input_gain, weights_fine, neuron_dynamics, neuron_bias, delete_simple_node;
- neuron_family, delete_connection.

Hay que reemplazar **l2f por pendulum-meta reactivo**: con 0,167 a 300k, l2f no alcanza el criterio de 0,2. Para pendulum-meta
se hace primero un Optuna reducido (50 trials × semillas 101–102, mismos 17 hiperparámetros, P con los 3 de Stanley).
- Referencias: trivial −1231,444; RL −323,56979.
- Antecedente: score previo de 0,41 ± 0,14 (barridos A/B).

**Configuraciones:**

| Config. | Qué es | Uso |
|---|---|---|
| c1 | Estático uniforme sobre K = 10 | Línea base de E1 |
| c2 | Estático ajustado (Stanley + P₀). En bottleneck y acrobot-swingup se reutiliza la base de la sonda de operadores, con las mismas semillas y el mismo binario. En pendulum-meta se corre nueva | Referencia de E2 |
| c3 | AP desde uniforme (K = 10) | ¿Captura AP el margen? |
| c4 | P en dos fases: uniforme y, desde la mitad de las generaciones, "afinar y podar" (P2 = 0,1 / 0,05 / 0,05 / 0,1 / 0,3 / 0,1 / 0,1 / 0,15 / 0,025 / 0,025, en el orden del catálogo) | ¿Hay margen para adaptar en el tiempo? |

- **Corridas:** 5 semillas pareadas (201–205), 75k evaluaciones, rotación. Son 3 configuraciones × 5 semillas × 2 entornos
  + 4 × 5 en pendulum-meta = **50 corridas nuevas**.
- **Binario:**
  - bottleneck y acrobot-swingup: `operadores_bin` (sha `ae239e87…`), congelado;
  - pendulum-meta (Optuna y sus 4 configuraciones): `fase_ppo_bin` (sha `52e237c8…`). Es idéntico bit a bit, verificado en la
    ficha [14](14_rendimiento.md), y ~2,5× más rápido. Se cambió antes de que arrancara la cadena (`encadenar_pendulum3.sh`).
- **Ejecución:** tope de 576 hebras; bottleneck y acrobot-swingup con 8 hebras por corrida, pendulum-meta con 4.
- **Cadena de pendulum-meta:** fin del piloto de PPO → Optuna → `pendulum-meta_config.json` → corridas → `CERRADO` →
  `FIN_margen`.

**Reglas fijadas antes de mirar** (todas pareadas por semilla):
- **Margen E1** = c2 − c1. Si es ≥ 0,03 en al menos 2 entornos, E1 es el escenario principal.
- **Captura de AP** = c3 − c1. Si AP captura menos de la mitad del margen, hay espacio para que PPO lo supere.
- **Margen E2** = c4 − c2. Si es ≤ 0, se espera un nulo en E2 y se reporta como tal.
- **Si no hay margen en E1** (< 0,03 en los tres entornos): **plan B**. Se amplía la acción con palancas de variación de mayor
  peso: intensidad de la mutación de pesos y fracción de hijos sin cruce (34 % y 26 % de importancia en el Optuna de l2f). Esto
  se conversa antes en la reunión de avance.

**Estadística de la comparación final** (a proponer en la reunión de avance): Wilcoxon de rangos con signo pareado por
semilla (números aleatorios comunes), con Holm, Â12 pareado y log-rank para la eficiencia, en lugar de Mann-Whitney
([09](09_sonda_operadores_etapa1.md)). Se fija antes de correr las réplicas finales.

## Resultados

**3 entornos** (`margen_aos/analisis_margen.py`, salida en `analisis_margen_3entornos.txt`). Score medio de 5 semillas pareadas:

| Entorno | c1 uniforme | c2 ajustado | c3 AP | c4 dos fases |
|---|---|---|---|---|
| bottleneck | 0,332 | 0,358 | 0,359 | 0,355 |
| acrobot-swingup | 0,500 | 0,587 | 0,552 | 0,537 |
| pendulum-meta | 0,324 | 0,430 | 0,361 | 0,319 |

| Regla | bottleneck | acrobot-swingup | pendulum-meta |
|---|---|---|---|
| Margen E1 = c2 − c1 (≥ 0,03) | +0,026: no | +0,086 (5/5): sí | +0,106 (3/5): sí |
| Captura de AP = c3 − c1 | +0,027 | +0,051 (60 %): AP captura la mayor parte | +0,037 (35 %): espacio para PPO |
| Margen E2 = c4 − c2 | −0,003: nulo | −0,050: nulo | −0,111: nulo |

**Se cumple la regla: E1 (sin ajuste previo) es el escenario principal**, con margen en 2 de 3 entornos. En E2 se espera un nulo
en los tres.

**Observaciones:**
- c2 usa el catálogo de Stanley con P ajustada y redes más grandes. Parte del margen puede deberse al catálogo o al tamaño de red,
  no solo a ajustar P.
- Con 5 semillas, el p mínimo de Wilcoxon es 0,0625: son indicios, no pruebas.
- En pendulum-meta se interrumpió un trial rezagado de Optuna (el 16, de la fase aleatoria) para no detener la cadena. La mejor
  configuración salió de los trials completos: el 38, con score 0,625 en las semillas de ajuste.

## Conclusión y límites

Pendiente. **Límites conocidos:**
- 5 semillas: la prueba de margen decide el escenario, no reemplaza a la comparación final con 11;
- c4 es una sola P en dos fases elegida a mano, no la mejor posible.

## Datos y reproducción

- `~/tesis/runs/margen_aos/`:
  - `preparar.py` (genera `jobs.json`), `lanzar.py` (planificador con tope en `tope.txt`, reanudable);
  - `encadenar_pendulum.sh`, `planificador.log`, `encadenar.log`;
  - resultados en `<entorno>/<config>/seed_<s>/`.
- Optuna de pendulum-meta: `~/tesis/runs/optuna_neat/pendulum-meta/`, lanzado con
  `scripts/lote_optuna_neat.sh pendulum-meta 8 50 "--seeds 101,102 --ref pendulum-meta_reactivo"`.
- Análisis: `analysis/compare.py --pareado`.

## Documentos relacionados

- `docs/comprension.md` §9 (2026-10-14, plan revisado).
- [09](09_sonda_operadores_etapa1.md) (origen del catálogo K = 10 y del diseño pareado) · [08](08_neat_300k.md) (cambio de l2f).
