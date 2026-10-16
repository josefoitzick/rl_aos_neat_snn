# 22 · Operadores dirigidos por la actividad de la SNN

| | |
|---|---|
| Semana | S04 (2026-10-15, 14:50 → 17:27) |
| Estado | terminado |
| Capítulo de la tesis | Metodología: operadores para SNN y estado informado · Resultados: AOS con estado frente al estático |

## Pregunta

El estado SNN del agente ([20](20_ppo_v2.md)) informa problemas de la red: salidas mudas (mediana 29 % en pendulum-meta),
acciones saturadas (45 %), ocultas y entradas mudas. Pero ningún operador del catálogo de 10 corrige esos problemas
directamente: sesgo y dinámica actúan sobre una neurona al azar. Por eso el estado no podía traducirse en una mejor decisión.

¿Operadores que corrigen la neurona señalada por su actividad, usados cuando el estado lo indica, le ganan al estático?

## Diseño

**Operadores nuevos** (`src/neat/genome.cpp`, F1–F4). Usan la tasa de disparo de cada neurona en la evaluación del padre,
que el evaluador mide con `NeatConfig::medir_actividad` (solo observa; el fitness es idéntico, test `[actividad]`):

| Operador | Neurona elegida | Cambio |
|---|---|---|
| despertar_salidas (F1) | salida con tasa ≤ 0,005 espigas por paso | sesgo + \|N(0, 2)\|; entradas excitatorias × exp(\|N(0, 0,5)\|) |
| despertar_ocultas (F2) | oculta con tasa ≤ 0,005 | ídem |
| calmar_saturadas (F3) | oculta o salida con tasa ≥ 0,08 | sesgo − \|N(0, 2)\|; entradas excitatorias × exp(−\|N(0, 0,5)\|) |
| ganancia_entradas_mudas (F4) | entrada con tasa ≤ 0,005 | ganancia × exp(\|N(0, 0,6)\|) |

- Sin neurona que cumpla, el operador no se aplica.
- Umbrales calibrados con los cuantiles de la tasa por rol en una corrida corta de cada entorno (máximo observado ~0,16);
  con los umbrales iniciales (0 y 0,2) F1 y F3 no se aplicaban nunca.
- **Verificación:** 96/96 tests; sin F1–F4 en el catálogo, `neat_control` y `aos_ppo_train --regla` dan resultados idénticos a
  `fase_ppo_bin3`; con F1–F4 cambian y se aplican (F1 y F4 ~80 %, F3 ~23 %). Binarios `fase_ppo_bin4` (`aos_ppo_train`
  1f894698…).

**Brazos** (E1, estado SNN, 10 decisiones por corrida, 75k evaluaciones; mismas opciones de NEAT que [19](19_regla_con_estado.md)):

| Brazo | Catálogo | Política |
|---|---|---|
| c0_10 | 10 | P uniforme (el estático sin ajuste) |
| c0_14 | 10 + F1–F4 | P uniforme |
| F | 10 + F1–F4 | regla por actividad: con salidas mudas sobre la mediana, despertar_salidas +1,5; con acción saturada sobre la mediana, calmar_saturadas +1,5 (ambas: +1,5 y +1) |
| tF (solo acrobot-swingup) | 10 + F1–F4 | regla por momento de [19](19_regla_con_estado.md) al inicio (τ ≤ 0,3) y después la regla por actividad, con borrar conexión −1 |

Umbrales de la regla: medianas del estado en el brazo c0 de [19](19_regla_con_estado.md). Archivos en
`~/tesis/runs/actividad_aos/reglas/`.

**Semillas nuevas:** pendulum-meta 701–892 (192), acrobot-swingup 401–440 (40).

**Reglas fijadas antes de mirar** (Wilcoxon pareado; Holm entre las comparaciones de cada entorno):
1. **F − c0_10 > 0, p Holm < 0,05:** el AOS con estado y operadores dirigidos le gana al estático sin ajuste.
2. **F − c0_14 > 0, p Holm < 0,05:** usar los operadores según el estado aporta más que solo tenerlos en el catálogo.
3. **c0_14 − c0_10:** efecto de los operadores nuevos sin estado (informativo).
4. acrobot-swingup: **tF − c0_10** y **tF − F** (combinar momento y actividad).
5. Si F gana a c0_10, sigue el entrenamiento de PPO con el catálogo de 14 y estado SNN en ese entorno.

## Resultados

Score del test final, pareado por semilla (`analisis.py`, salida en `analisis_resultado.txt`):

| Entorno (semillas) | c0_10 | c0_14 | F | tF | F − c0_10 | F − c0_14 | c0_14 − c0_10 |
|---|---|---|---|---|---|---|---|
| acrobot-swingup (40) | 0,514 | 0,510 | 0,514 | 0,516 | −0,001 (Holm 1) | +0,003 (Holm 1) | −0,004 (Holm 1) |
| pendulum-meta (192) | **0,363** | 0,332 | 0,344 | — | −0,019 (Holm 0,52) | +0,012 (Holm 0,52) | −0,031 (p 0,068, Holm 0,20) |

acrobot-swingup: tF − c0_10 +0,002 y tF − F +0,003 (Holm 1).

**Lectura con las reglas fijadas:**
- **Reglas 1 y 2 no se cumplen** en ningún entorno: los operadores dirigidos por actividad, con esta calibración, no le ganan al
  estático. Por la regla 5, no se entrena PPO con el catálogo de 14.
- Agregar F1–F4 al catálogo uniforme **empeora** pendulum-meta (−0,031): diluye la probabilidad de los operadores útiles y/o
  los nuevos dañan. Usarlos según el estado recupera parte (F − c0_14 +0,012 en pendulum-meta, +0,003 en acrobot-swingup),
  sin significancia.
- **Pista para el margen** (análisis posterior, no preregistrado): el estático ajustado con Optuna usa en pendulum-meta casi
  solo mutar pesos (P 0,78 / 0,21 / 0,01, potencia 3,4) y casi nunca agregar nodo. Las reglas de las fichas 19 y 22 bajaban
  mutar pesos (por los efectos de corto plazo de la ficha 18), justo al revés. Se prueba en [25](25_margen_pmeta_p.md).

## Datos y reproducción

- `~/tesis/runs/actividad_aos/`: `reglas/`, `preparar.py` (genera `jobs.json` y los pone al frente de la cola de
  `sonda_final`), `analisis.py`; resultados en `<entorno>/<brazo>/<semillas>/seed_XXX`.
- Verificación: `~/tesis/runs/optim_verif/actividad/` (`verificar.sh`, `r_nuevo`/`r_cong`/`r_F`); congelado con
  `optim_verif/congelar4.sh`.

## Documentos relacionados

- [18](18_analisis_efectos_acciones.md) · [19](19_regla_con_estado.md) · [20](20_ppo_v2.md) · [21](21_sonda_final.md).
