# 25 · ¿El margen de pendulum-meta se alcanza moviendo solo P?

| | |
|---|---|
| Semana | S04 (2026-10-15 → 16; pendulum-meta 18:25, acrobot-swingup 00:40) |
| Estado | terminado |
| Capítulo de la tesis | Resultados: dónde está el margen para un AOS · Discusión: efecto de corto plazo frente a efecto final |

## Pregunta

En E1, el estático ajustado con Optuna le saca +0,077 a c0 en pendulum-meta ([15](15_margen_intensidades.md)). Ese ajuste
usa casi solo mutar pesos (P de Stanley 0,78 / 0,21 / 0,01; potencia 3,4) y casi nunca agregar nodo. Las reglas con estado
probadas ([19](19_regla_con_estado.md), [22](22_operadores_actividad.md)) bajaban mutar pesos y empeoraron.

¿Una P fija con mutar pesos alto y agregar nodo casi nulo, con intensidades genéricas, recupera ese margen? Si sí, la palanca P
es fuerte en pendulum-meta y el AOS tiene un objetivo grande y claro. Si no, el margen está en las intensidades.

Es un diagnóstico (usa información del ajuste), no un método a comparar.

## Diseño

- **Brazos** (P fija; catálogo de 10; intensidades genéricas; mismo comando que c0_10 de [22](22_operadores_actividad.md) salvo la
  regla):

  | Brazo | Empuje (antes del mapa c·tanh) | P resultante aprox. |
  |---|---|---|
  | mw_alto | mutar pesos +3, agregar nodo −3 | mutar pesos 0,55; nodo 0,006; los otros 8 ~0,056 |
  | como_ajustado | mutar pesos +3, conexión +1,5, los otros 8 −3 | mutar pesos 0,67; conexión 0,27; los otros 8 ~0,007 |

- **Semillas** 701–892 (192), pareadas con c0_10 de [22](22_operadores_actividad.md); binario `fase_ppo_bin4`.

**Reglas fijadas antes de mirar** (Wilcoxon pareado, Holm entre los dos brazos):
1. **Brazo − c0_10 ≥ +0,03 con p Holm < 0,05:** la palanca P alcanza una parte grande del margen en pendulum-meta. El AOS debe
   apuntar ahí: se usa como referencia para la recompensa y la base del PPO residual, y la sonda ([21](21_sonda_final.md))
   dice en qué fase conviene.
2. **Si ninguno llega:** el margen de pendulum-meta está en las intensidades (potencia, fracción sin cruce), no en P.

**25b · acrobot-swingup** (agregado el 2026-10-15 17:48, antes de cualquier resultado): el estático ajustado (#49) usa P de Stanley
0,52 / 0,41 / 0,07 con potencia **0,69** (fina, contra 5 genérica). Brazo como_ajustado: esa P dentro del catálogo de 10
(mutar pesos +3, conexión +2,49, nodo +0,33, los otros 7 −3; tras el mapa ~0,50 / 0,39 / 0,07 / 0,005), con intensidades
genéricas, en las semillas 401–440, pareado con c0_10 de [22](22_operadores_actividad.md). Misma lectura: ≥ +0,03 con p < 0,05
→ el margen de acrobot-swingup está en P; si no, en la potencia de la mutación de pesos.

## Resultados

**pendulum-meta** (192 semillas pareadas 701–892; `analisis_resultado_pmeta.txt`):

| c0_10 | mw_alto | como_ajustado | mw_alto − c0_10 | como_ajustado − c0_10 |
|---|---|---|---|---|
| 0,363 | 0,469 | **0,483** | **+0,106** (ee 0,016; 138/192; Holm < 0,0001) | **+0,120** (ee 0,016; 138/192; Holm < 0,0001) |

- **Regla 1 se cumple con holgura:** moviendo solo P (más mutar pesos, casi nada de agregar nodo), con intensidades genéricas, se
  gana +0,11 a +0,12 sobre c0, más que el margen estimado con 5 semillas en [15](15_margen_intensidades.md) (+0,077). En
  pendulum-meta la palanca P es grande: el AOS tiene un objetivo de ~+0,1 y una dirección clara.
- Explica los fracasos anteriores: las reglas de [19](19_regla_con_estado.md) y [22](22_operadores_actividad.md) bajaban mutar
  pesos, guiadas por su efecto de corto plazo ([18](18_analisis_efectos_acciones.md)). El efecto de corto plazo de mutar pesos
  (empeora al hijo) es opuesto a su efecto sobre el resultado final.

**acrobot-swingup (25b)** (40 semillas pareadas 401–440): c0_10 0,514, como_ajustado **0,549**; Δ **+0,035** (ee 0,008), 34/40,
Wilcoxon p 0,0001. **El margen de acrobot-swingup también está en P:** la P de Stanley ajustada (mutar pesos 0,50, conexión
0,39, nodo 0,07) dentro del catálogo de 10, con intensidades genéricas, recupera +0,035 de los +0,046 que da el ajuste completo
([15](15_margen_intensidades.md)). Como en pendulum-meta, va en la dirección contraria a la regla por momento
([19](19_regla_con_estado.md)), que bajaba mutar pesos guiada por efectos de corto plazo; eso explica por qué la regla rendía tan
poco ([26](26_confirmacion_acrobot.md)).

**Lectura conjunta:** en los dos entornos baratos hay una palanca P grande y clara (+0,12 y +0,035), que AP no captura y que los
efectos de corto plazo esconden. Es el objetivo del AOS aprendido (`docs/diseno_ppo.md` §12).

## Datos y reproducción

- `~/tesis/runs/margen_pmeta_p/`: `reglas/`, `preparar.py` (agrega los trabajos a la cola de `sonda_final`), `analisis.py`.

## Documentos relacionados

- [15](15_margen_intensidades.md) · [19](19_regla_con_estado.md) · [22](22_operadores_actividad.md) · [21](21_sonda_final.md).
