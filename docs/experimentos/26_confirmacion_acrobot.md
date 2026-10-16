# 26 · Confirmación con potencia de la regla con estado en acrobot-swingup

| | |
|---|---|
| Semana | S04 (preparada el 2026-10-15 17:50) |
| Estado | terminado (2026-10-15 23:30) |
| Capítulo de la tesis | Resultados: AOS con estado frente al estático (evidencia confirmatoria) |

## Pregunta

La regla con estado de acrobot-swingup le ganó a c0 en la muestra de descubrimiento (+0,019, 20 semillas;
[19](19_regla_con_estado.md)) y en la confirmación (+0,009, 40 semillas, p 0,11; [21](21_sonda_final.md)), con la misma
dirección. Con un efecto de ~+0,01 y una desviación de la diferencia pareada de ~0,039, 40 semillas no alcanzan (poder ~0,4).

¿Se confirma el efecto con una muestra nueva del tamaño necesario?

## Diseño

- **Semillas nuevas 401–500 (100)**, ninguna usada para escribir o ajustar la regla. Poder ~0,85 para +0,012 (α 0,05, dos colas).
- **Brazos:**
  - c0: en 401–440 ya existe (c0_10 de [22](22_operadores_actividad.md)); en 441–500 se corre con el mismo comando y binario
    (`fase_ppo_bin4`);
  - regla: la regla por momento de [19](19_regla_con_estado.md) con la dosis que elija [24](24_dosis.md), según esta regla
    fijada antes de ver la dosis: **×2 si ×2 − ×1 > 0 con p Holm < 0,05 en la ficha 24; si no, ×1**.
- No se mira ningún resultado parcial; el análisis corre una vez, al terminar los 100 pares.

**Regla de decisión fijada antes de mirar:** regla − c0 > 0 con Wilcoxon pareado p < 0,05 (dos colas) en las 100 semillas →
**el AOS con estado (escrito a mano) le gana al estático sin ajuste en acrobot-swingup**, confirmado con datos independientes.

## Resultados

100 semillas pareadas 401–500 (`analisis.py`, salida en `analisis_resultado.txt`):

| c0 | regla (×1) | regla − c0 | Gana | Wilcoxon p |
|---|---|---|---|---|
| 0,515 | 0,519 | +0,004 (ee 0,004) | 61/100 | 0,11 |

- **La regla no se cumple:** con la muestra del tamaño planificado, el efecto de la regla con estado en acrobot-swingup no es
  significativo. La dirección se mantiene (gana en 61 de 100), pero el tamaño es ~+0,004, menor que en el descubrimiento
  (+0,019, 20 semillas) y en la primera confirmación (+0,009, 40 semillas): el efecto inicial estaba inflado (selección de la
  regla con la misma muestra y regresión a la media).
- **Lectura:** en acrobot-swingup, la regla por momento con este catálogo tiene, a lo más, un efecto muy chico sobre el
  resultado final. No sirve como evidencia de que el AOS con estado le gane al estático en este entorno.

## Datos y reproducción

- `~/tesis/runs/confirmacion_acrobot/`: `preparar_c0.py` (brazo c0 en 441–500) y `preparar_regla.py` (brazo regla ×1 en 401–500,
  elegido por la regla 3 de [24](24_dosis.md)), ambos en la cola de `sonda_final`; `analisis.py` corre al terminar
  (`esperar_y_analizar.sh`).

## Documentos relacionados

- [19](19_regla_con_estado.md) · [21](21_sonda_final.md) · [22](22_operadores_actividad.md) · [24](24_dosis.md).
