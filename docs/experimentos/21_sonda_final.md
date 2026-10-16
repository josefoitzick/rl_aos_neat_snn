# 21 · Sonda aleatorizada sobre el resultado final y confirmación en acrobot-swingup

| | |
|---|---|
| Semana | S04 (lanzada el 2026-10-15 a las 14:10) |
| Estado | en curso |
| Capítulo de la tesis | Resultados: qué operadores sirven al resultado final y cuándo · Discusión: efecto local frente a efecto final |

## Pregunta

La regla con estado ([19](19_regla_con_estado.md)) mostró que los efectos de los operadores medidos sobre el progreso a corto
plazo ([18](18_analisis_efectos_acciones.md)) no predicen el resultado final: en pendulum-meta, empujar hacia los operadores
"buenos" empeoró. En acrobot-swingup, la regla le ganó a la P fija y a c0, al borde de lo significativo.

- **A.** ¿Se confirma en semillas nuevas que la regla con estado le gana a c0 en acrobot-swingup?
- **B.** ¿Qué operadores mejoran o empeoran el **resultado final**, y depende del momento de la corrida (inicio o resto)?

## Diseño

Escenario E1 (c0: P uniforme sobre el catálogo de 10, intensidades genéricas), 75k evaluaciones, rotación. Binarios
`fase_ppo_bin3` (`neat_control` 42256036…, `aos_ppo_train` 13c25b49…).

- **A. Confirmación:** acrobot-swingup, semillas **321–360** (40, nuevas), brazos c0 y regla, con los mismos comandos de
  [19](19_regla_con_estado.md) salvo las semillas.
- **B. Sonda aleatorizada:** cada corrida usa una P al azar en el primer tercio de las generaciones y otra en el resto
  (`neat_control --p --p2 --p-switch G/3`): P = uniforme·exp(z), z ~ N(0, 1) independiente por operador y por fase
  (empuje típico ×2,7). Semilla del sorteo 20261015; los z de cada corrida quedan en `diseno.json`.
  - acrobot-swingup: 320 corridas (semillas 5001–5320);
  - pendulum-meta: 1920 corridas (semillas 5001–6920).
  - Modelo: score final ~ z₁ + z₂ centrados por corrida (contraste de suma cero), MCO con errores estándar por bootstrap.
  - Precisión esperada por coeficiente: ~±2,6 × 10⁻³ en acrobot-swingup y ~±4 × 10⁻³ en pendulum-meta.
- Corre junto al entrenamiento de PPO v2 en acrobot-swingup ([20](20_ppo_v2.md)), con tope global de 540 hebras.

**Reglas fijadas antes de mirar:**
1. **A:** regla − c0 en las semillas 321–360 con Wilcoxon pareado p < 0,05 (dos colas) → el efecto de la regla con estado en
   acrobot-swingup queda confirmado con datos independientes.
2. **B, global:** permutación del R² con p < 0,05 → la P (qué operadores se usan) afecta el resultado final en ese entorno.
3. **B, por operador:** coeficiente con |t| > 3 → el operador es útil (positivo) o dañino (negativo) para el resultado final,
   en esa fase o en toda la corrida.
4. **B, momento:** F de β₁ = β₂ con p < 0,05 → el mejor uso de los operadores cambia durante la corrida (base de un AOS con
   estado).
5. Si la regla global da p ≥ 0,05 en ambos entornos, la elección de operadores de este catálogo no mueve el resultado final a
   75k, y el catálogo se revisa (operadores dirigidos por la actividad de la SNN).

## Resultados

**A. Confirmación en acrobot-swingup** (terminada 2026-10-15 15:43; `analisis_confirmacion.txt`):

| Semillas | c0 | regla | regla − c0 | Gana | Wilcoxon p |
|---|---|---|---|---|---|
| 321–360 (40, nuevas) | 0,525 | 0,534 | +0,009 (ee 0,006) | 25/40 | 0,11 |

- **Regla 1 no se cumple:** la confirmación no alcanza p < 0,05. La dirección se repite (+0,009 aquí, +0,019 en [19](19_regla_con_estado.md)),
  con un efecto más chico que en la muestra de descubrimiento.
- Exploratorio (no confirmatorio, porque incluye la muestra con que se escribió la regla): las 60 semillas juntas dan +0,012
  (ee 0,005), 39/60, p 0,015. El efecto de la regla por momento en acrobot-swingup sería de ~+0,01: real pero chico.

**B. Sonda aleatorizada:** pendiente.

## Datos y reproducción

- `~/tesis/runs/sonda_final/`: `preparar.py` (genera `jobs.json` y `diseno.json`), `lanzar.py` (planificador; tope en
  `tope.txt`), `analisis.py`; confirmación en `confirmacion/<brazo>/<semillas>/seed_XXX`, sonda en `sonda/<entorno>/seed_XXXX`.

## Documentos relacionados

- [18](18_analisis_efectos_acciones.md) · [19](19_regla_con_estado.md) · [20](20_ppo_v2.md).
