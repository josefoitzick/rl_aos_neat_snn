# 07 · Ajuste de hiperparámetros de NEAT con Optuna

| | |
|---|---|
| Semana | S03 (2026-10-09 → 11) |
| Estado | terminado |
| Capítulo de la tesis | Metodología: líneas base ajustadas · Resultados: NEAT estático |

## Pregunta

¿Cuánto mejora NEAT estático si se ajustan sus hiperparámetros, incluido P, en cada entorno? Comparar los métodos de AOS
contra un NEAT con probabilidades arbitrarias no sería justo: la línea base debe ser **la mejor versión** del estático.

## Diseño

- **Orden acordado:** primero NEAT estático, luego Adaptive Pursuit con el mismo procedimiento. El catálogo ampliado queda para
  después. Protocolo de rotación.
- **17 hiperparámetros:**

  | Grupo | Hiperparámetros (rango) |
  |---|---|
  | Población | tamaño (50–300; generaciones = 75 000 / población); especies objetivo (3–25); escala de compatibilidad (2–30, log); supervivencia (0,1–0,5); estancamiento por especie (10–40) |
  | Pesos | perturbación (0,5–15, log); probabilidad de reasignar (0–0,2); peso del nodo nuevo (15–50); rango inicial (10–50); peso máximo (1–3 × rango inicial) |
  | Cruce | fracción de hijos sin cruce (0,1–0,6) |
  | Interfaz SNN | ganancia de entrada (10–100, log); espigas para la acción máxima (2–15); ventana por paso (20–100 ms) |
  | P | pesos (0,05–1); conexión y nodo (0,005–0,5, log) |

- **Fijos:** 10 episodios de entrenamiento, 10 de validación, 100 de test final; presupuesto de **75 000 evaluaciones**.
- **Optuna** (4.9):
  - TPE multivariado: 20 trials al azar al inicio, *constant liar*;
  - MedianPruner sobre la mejor validación en cada 10 % del presupuesto;
  - semillas de ajuste 101–103 (nunca 1–10);
  - objetivo: score medio del test final;
  - trial 0 = configuración calibrada previa;
  - ~100 trials por entorno.
- **Verificación previa:**
  - sin opciones, idéntico al binario de la sonda de P en 10 entornos + control negativo;
  - valores por omisión explícitos = sin opciones;
  - cada opción nueva cambia el resultado.

  La prueba de humo encontró un error de índice con presupuestos mínimos, ya corregido.
- **Binario congelado** `~/tesis/runs/optuna_neat_bin` (sha `93511927c1b9b345`).
- **Evaluación:** la mejor configuración de cada entorno se corre en las semillas 1–10 y se compara de forma pareada con Stanley
  bajo rotación (sonda de P, mismo presupuesto).

## Resultados

**Estudios:**

| Entorno | Trials (completos / podados / fallidos) | Mejor trial | Objetivo (semillas 101–103) | Trial 0 |
|---|---|---|---|---|
| bottleneck | 116 (43 / 70 / 3) | #96 | 0,365 | 0,313 |
| acrobot-swingup | 120 (79 / 41 / 0) | #49 | 0,624 | 0,514 |
| l2f | 104 (52 / 52 / 0) | #65 | 0,120 | 0,060 |

**Evaluación en las semillas 1–10** (score del test final con las referencias de `score_v2_s1-10.json`; Wilcoxon pareado):

| Entorno | Stanley, rotación | Ajustado | Δ | Semillas que mejoran | p |
|---|---|---|---|---|---|
| bottleneck (#96) | 0,267 ± 0,058 | **0,332 ± 0,045** | +0,065 | 9/10 | 0,049 |
| bottleneck (#40, segundo mejor) | 0,267 | 0,305 ± 0,061 | +0,038 | 8/10 | 0,064 |
| acrobot-swingup (#49) | 0,484 ± 0,042 | **0,573 ± 0,038** | +0,089 | 10/10 | 0,002 |
| l2f (#65) | 0,091 ± 0,031 | **0,147 ± 0,043** | +0,056 | 10/10 | 0,002 |

En retorno del entorno: bottleneck 16,5 → 20,6; acrobot-swingup −12,61 → −11,61; l2f 6,59 → 8,07.

**Configuraciones ganadoras** (P normalizada como pesos / conexión / nodo):

| | bottleneck #96 | acrobot-swingup #49 | l2f #65 |
|---|---|---|---|
| Población × generaciones | 70 × 1071 | 130 × 577 | 210 × 357 |
| P₀ | 0,879 / 0,040 / 0,080 | 0,520 / 0,407 / 0,074 | 0,983 / 0,008 / 0,009 |
| Perturbación de pesos · reasignación | 3,26 · 0,0065 | 0,69 · 0,048 | 0,62 · 0,0024 |
| Ganancia de entrada · ventana | 86,6 · 50 ms | 62,0 · 30 ms | 37,8 · 80 ms |
| Fracción sin cruce | 0,23 | 0,42 | 0,32 |

**Importancia de los hiperparámetros** (fANOVA, semilla 0):
- bottleneck: ganancia de entrada 55 %, P de conexión 13 %, P de pesos 8 %;
- acrobot-swingup: ganancia de entrada 91 %;
- l2f: probabilidad de reasignar pesos 34 %, fracción sin cruce 26 %, supervivencia 8 %.

Figuras: `~/tesis/runs/figuras_optuna/optuna_neat_3_entornos.png` (historia del estudio) y `curvas_neat_3_entornos.png`
(curvas del estándar frente al ajustado).

## Conclusión y límites

- El ajuste mejora a NEAT en los 3 entornos, con significancia en acrobot-swingup y l2f y al límite en bottleneck. Esta es la
  línea base "estático ajustado" (P₀) de la tesis.
- La interfaz SNN–entorno (ganancia de entrada) es el parámetro más importante en 2 de 3 entornos. Los operadores (P) pesan
  poco, en línea con la sonda de P ([06](06_sonda_p.md)).
- l2f sigue por debajo del criterio de viabilidad (0,147 < 0,2).
- **Abierto:**
  - si los mejores trials necesitaban más presupuesto (las curvas seguían subiendo) → [08](08_neat_300k.md);
  - AP con el mismo procedimiento;
  - asimetría de pesos iniciales;
  - P según el tamaño de la red;
  - diagnóstico de la interfaz (zona muerta medida: |x| ≤ 0,05 → 0 espigas con ganancia 40 y 50 ms).
- **Límites:**
  - 3 semillas de ajuste por trial;
  - la importancia por fANOVA depende de la muestra de trials (TPE concentra la búsqueda).

## Datos y reproducción

- Estudios: `~/tesis/runs/optuna_neat/<entorno>/estudio.log` (JournalStorage), carpetas `trial_NNNN` y `logs/`.
- Evaluación: `~/tesis/runs/eval_optuna/` (`preparar.py`, `lanzar*.sh`, `cmds_*.txt` con la línea de comando completa de cada
  configuración).
- `scripts/optuna_neat.py` (entorno conda `tdi`), `scripts/lote_optuna_neat.sh` y `scripts/reasignar_optuna.py`.
- Figuras: `~/tesis/runs/figuras_optuna/figura_optuna.py` y `figura_curvas.py`.

## Documentos relacionados

- `docs/diseno_neat_ap.md` §17.
