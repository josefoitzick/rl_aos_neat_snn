# 09 · Sonda de operadores, etapa 1: tamizado del catálogo ampliado

| | |
|---|---|
| Semana | S04 (2026-10-13 → 14) |
| Estado | terminado (210/210). Las etapas 2–6 previstas se reemplazaron por el plan revisado ([13](13_prueba_margen.md)) |
| Capítulo de la tesis | Metodología: diseño del catálogo de operadores · Resultados: análisis previo · Discusión |

## Pregunta

¿Qué operadores nuevos cambian el resultado de NEAT ajustado al agregarse al catálogo de Stanley? La sonda de P mostró que
los 3 operadores clásicos dejan poco margen ([06](06_sonda_p.md)). Un AOS necesita operadores cuyo efecto **dependa del
estado o del entorno**.

## Diseño

- **Catálogo candidato:** búsqueda en la literatura de NEAT, SNN y AOS más operadores propios de SNN
  (`docs/operadores_candidatos.md`). Criterios fijados antes de mirar resultados de AP o PPO:
  1. precedente o justificación propia de SNN;
  2. no redundante;
  3. costo bajo o medio;
  4. cubre una dirección que el catálogo no tiene;
  5. su utilidad debería depender de una señal observable.
- **13 candidatos:**

  | Familia | Operadores |
  |---|---|
  | Pesos | `weights_fine` (pesos finos dispersos); `weights_reset` (reasignar pesos puntuales); `weights_directed` (dirigido por la población) |
  | Neurona | `neuron_bias` (excitabilidad); `neuron_dynamics` (perturbar parámetros de Izhikevich); `neuron_family` (cambiar de familia) |
  | Interfaz | `input_gain` (sensibilidad de entrada) |
  | Crecer | `add_node_family` (nodo con familia); `duplicate_node` (duplicar neurona) |
  | Podar | `delete_connection`; `delete_simple_node` |
  | Variantes de intensidad | `add_node_multi` (3 nodos de una vez); `weights_fine_cauchy` (Cauchy en vez de Gauss) |

- **Implementación** (2026-10-13), verificada bit a bit contra el binario de Optuna cuando no se usan los operadores nuevos.
  Binario congelado `~/tesis/runs/operadores_bin` (sha `ae239e87152505b2`).
- **Tamizado:**
  - por entorno, la base NEAT ajustada ([07](07_optuna_neat.md)) + catálogo de Stanley + **un** candidato, con P moderada
    (el mismo peso que el operador de Stanley más parecido);
  - 5 semillas (201–205, pareadas con la base), 75k evaluaciones, rotación;
  - 14 variantes × 5 semillas × 3 entornos (bottleneck, acrobot-swingup, l2f) = 210 corridas;
  - se registra el crédito por rango: percentil de cada hijo en su generación.
- **Clasificación fijada antes de mirar:**
  - AYUDA si Δ > 0 y gana en ≥ 4/5 semillas;
  - DAÑA si Δ < 0 y pierde en ≥ 4/5;
  - si no, NEUTRO.
- **Ejecución:** del 2026-10-13 20:41 al 10-14 21:16, con un tope de 520 hebras.

## Resultados

Δ del test final contra la base, pareado por semilla, en score (referencias de `score_v2.json`; con las de s1-10 los Δ cambian
< 2 %):

| Candidato | bottleneck | acrobot-swingup | l2f |
|---|---|---|---|
| *base (score)* | *0,359* | *0,579* | *0,098* |
| input_gain | +0,028 AYUDA | −0,008 | +0,031 AYUDA |
| neuron_dynamics | +0,040 | −0,003 | +0,006 |
| neuron_bias | +0,032 | −0,003 | +0,014 |
| delete_simple_node | +0,024 | +0,017 | +0,027 AYUDA |
| weights_fine | +0,022 | −0,029 | +0,026 |
| weights_directed | +0,021 | −0,016 | +0,002 |
| weights_fine_cauchy | +0,014 | −0,024 DAÑA | +0,021 |
| duplicate_node | +0,011 | −0,006 | +0,021 AYUDA |
| add_node_family | +0,006 | −0,047 DAÑA | +0,015 |
| weights_reset | +0,004 | −0,007 | +0,026 AYUDA |
| add_node_multi | −0,012 | −0,025 | +0,015 AYUDA |
| delete_connection | −0,021 DAÑA | −0,028 DAÑA | +0,024 AYUDA |
| neuron_family | **−0,063 DAÑA** | −0,020 | +0,015 AYUDA |

Detalle por entorno (score, semillas que ganan, percentil de los hijos, fallas, conexiones): copia parchada de
`analisis_etapa1.py` (ver "Datos"). Reporte estadístico por entorno en `<entorno>/analysis/report.md`.

**Análisis posterior** (2026-10-14, sobre la sonda de P y esta sonda):
- **La semilla explica la mayor parte de la varianza.** La correlación intra-semilla entre configuraciones es 0,72 en
  acrobot-swingup, 0,79 en l2f y 0,18–0,22 en bottleneck. La semilla fija también los episodios de entrenamiento, validación
  y test, así que un diseño **pareado** (mismas semillas para todos los métodos) elimina ese ruido.

  Mejora mínima detectable con 11 réplicas (α = 0,05, poder 0,8):

  | Entorno | Pareada | Sin parear |
  |---|---|---|
  | acrobot-swingup | ≈ 0,025 | 0,05–0,07 |
  | l2f | ≈ 0,024 | 0,05–0,07 |
  | bottleneck | ≈ 0,05 | 0,05–0,07 |

- **El percentil de los hijos distingue operadores en acrobot-swingup:** mutar pesos baja de 0,41 a 0,33 a lo largo de la
  corrida, mientras que los demás quedan en 0,60–0,70. Aun así, la selección filtra a los malos hijos y el resultado final
  casi no cambia.
- **En bottleneck y l2f el crédito inmediato es ruido:** todos los operadores quedan en 0,50–0,55. Un AOS basado en crédito
  por hijo (AP) no puede distinguirlos ahí.

## Conclusión y límites

- Los efectos son de ±0,02–0,04 y, con 5 semillas, casi todos quedan dentro del ruido. Con la comparación prevista (11 réplicas,
  Mann-Whitney), ningún AOS podría mostrar una mejora de ese tamaño.
- Hay operadores con **efecto dependiente del entorno**: neuron_family (−0,063 en bottleneck, +0,015 en l2f) y delete_connection
  (−0,021 / −0,028 / +0,024). Una P fija genérica no puede servir a los tres entornos: ese es el caso de uso de un AOS.
- **Decisión** (2026-10-14): plan revisado ([13](13_prueba_margen.md)):
  - catálogo K = 10:
    - los 3 de Stanley;
    - input_gain, weights_fine, neuron_dynamics, neuron_bias y delete_simple_node;
    - neuron_family y delete_connection, por su efecto dependiente del entorno;
  - diseño pareado;
  - escenario sin ajuste previo (P uniforme).
- **Límites:**
  - 5 semillas;
  - un candidato a la vez, sobre una base ajustada para el catálogo de Stanley;
  - la clasificación AYUDA/DAÑA no es una prueba de significancia.

## Datos y reproducción

- `~/tesis/runs/sonda_operadores/` (`preparar.py`, `lanzar.py`, `jobs.json`, `planificador.log`, `FIN_sonda`); análisis
  `analisis_etapa1.py`.

  **Nota:** el script falla al encontrar la carpeta `analysis/` dentro de cada entorno. Para correrlo, se omite esa carpeta del
  listado de variantes.
- `compare.py --pareado` (Wilcoxon pareado + Â12 pareado) en `analysis/compare.py` del repositorio.
- Operadores: `src/neat/` (catálogo y operadores nuevos); verificación en `~/tesis/runs/operadores_verif`.

## Documentos relacionados

- `docs/operadores_candidatos.md` (búsqueda, inventario, criterios, implementación §6.1 y plan de la sonda §7).
