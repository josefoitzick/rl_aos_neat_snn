# 24 · Dosis: ¿crece el efecto de la regla con estado si empuja más fuerte?

| | |
|---|---|
| Semana | S04 (2026-10-15, 17:07 → 18:30) |
| Estado | terminado |
| Capítulo de la tesis | Resultados: tamaño del efecto del AOS con estado · Discusión: hasta dónde llega la palanca P |

## Pregunta

La regla con estado de acrobot-swingup ([19](19_regla_con_estado.md)) le gana a c0 por ~+0,012 (60 semillas, 39/60;
[21](21_sonda_final.md)). Empuja ±1 en el logit (×2,7 / ×0,37) al inicio de la corrida. ¿El efecto crece si el empuje se duplica?

Motivación: el análisis del ruido de exploración de PPO v2 (`~/tesis/runs/efecto_final/`) indica que, en acrobot-swingup, cada
operador por separado mueve poco el resultado final (≤ ~0,01 por empuje ×2,7 sostenido en una fase). Si el efecto de la regla
es aproximadamente lineal en el empuje, ×2 debería acercarse a +0,02–0,025. Si se satura o se invierte, la palanca P con este
catálogo está cerca de su techo en acrobot-swingup.

## Diseño

- **Brazo nuevo:** regla ×2 (los mismos empujes de la regla con el doble de magnitud: ±2 antes del mapa, ×7,4 / ×0,14).
- **Pareja:** los brazos c0 y regla ×1 ya corridos en las semillas 301–360 ([19](19_regla_con_estado.md) y
  [21](21_sonda_final.md)), con el mismo binario (`fase_ppo_bin3`) y los mismos comandos, salvo el archivo de regla.
- 60 corridas nuevas (~300 hebra-hora).

**Reglas fijadas antes de mirar** (Wilcoxon pareado, Holm entre las dos comparaciones):
1. **×2 − c0 > 0 con p Holm < 0,05:** la regla con doble dosis le gana al estático.
2. **×2 − ×1 > 0 con p Holm < 0,05:** el efecto crece con la dosis; se prueba ×3 y se usa la mejor dosis como base del PPO
   residual ([23](23_ppo_residual.md)).
3. **×2 − ×1 ≤ 0:** el efecto está saturado; la base del PPO residual queda en ×1.

## Resultados

60 semillas pareadas (301–360; `analisis.py`, salida en `analisis_resultado.txt`):

| c0 | regla ×1 | regla ×2 | ×2 − c0 | ×2 − ×1 |
|---|---|---|---|---|
| 0,528 | **0,540** | 0,528 | +0,001 (32/60, Holm 0,74) | −0,012 (26/60, p 0,13, Holm 0,27) |

- **Regla 3:** el efecto no crece con la dosis; con el doble de empuje desaparece (×2 ≈ c0) y queda por debajo de ×1. El efecto
  de la regla en acrobot-swingup está saturado (o se invierte) por encima de ×1: la base del PPO residual y la dosis de la
  confirmación ([26](26_confirmacion_acrobot.md)) quedan en ×1.
- Lectura: en acrobot-swingup, la palanca P con este catálogo tiene un óptimo cercano a empujes moderados; empujar más concentra
  la variación en pocos operadores y pierde lo que aportan los demás.

## Datos y reproducción

- `~/tesis/runs/dosis/`: `reglas/acrobot_regla_x2.txt`, `preparar.py` (genera `jobs.json` y los agrega a la cola de
  `sonda_final`), `analisis.py`.
- Análisis del ruido de exploración: `~/tesis/runs/efecto_final/analisis.py` y `resultado.txt`.

## Documentos relacionados

- [19](19_regla_con_estado.md) · [21](21_sonda_final.md) · [22](22_operadores_actividad.md) · [23](23_ppo_residual.md).
