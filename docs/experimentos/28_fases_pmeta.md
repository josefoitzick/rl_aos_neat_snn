# 28 · ¿El momento importa en pendulum-meta? Regla por fases contra la mejor P fija

| | |
|---|---|
| Semana | S04 (2026-10-15, 22:35 → 23:50) |
| Estado | terminado |
| Capítulo de la tesis | Resultados: el estado aporta más allá de la mejor P fija · Discusión: crédito inmediato frente a resultado final |

## Antecedentes (análisis de datos existentes, 2026-10-15)

**1. El crédito inmediato engaña** (`~/tesis/runs/efecto_final/credito_inmediato_pmeta.txt`; c0 de [19](19_regla_con_estado.md),
300 corridas): la fracción de hijos que superan a su padre, que es lo que premia AP, es mínima para mutar pesos (0,22) y
borrar conexión (0,23), y máxima para los cambios suaves (pesos finos, dinámica, sesgo, poda: 0,37–0,38). Pero usar mucho mutar
pesos es lo que más mejora el resultado final (+0,11; [25](25_margen_pmeta_p.md)). AP lo baja a 4–7 % y no captura nada de
ese margen: con 300 semillas queda igual que c0 (−0,003, p 0,70; [27](27_ppo_v2b.md)).

**2. El mejor operador cambia durante la corrida** (`~/tesis/runs/efecto_final/resultado_pmeta_v2.txt`): con el ruido de
exploración de PPO v2 (800 corridas, política casi uniforme, σ 1; [20](20_ppo_v2.md)), efecto sobre el score final por +1 en
log P sostenido en la fase (×10⁻³):

| | inicio (t 0–2) | medio (t 3–5) | final (t 6–9) |
|---|---|---|---|
| mutar pesos +1 y agregar nodo −1 | +25 ± 13 | +25 ± 14 | −20 ± 16 |
| sesgo neuronal | +7 ± 9 | **+30 ± 9** | **+42 ± 10** |
| ganancia de entrada | −14 ± 9 | −20 ± 8 | −26 ± 11 |
| borrar conexión | −23 ± 9 | −17 ± 8 | +5 ± 10 |

La dirección de la P fija ganadora ayuda en la primera mitad; el sesgo, en la segunda. Una P fija no puede hacer las dos cosas.

## Pregunta

¿Una regla que cambia los operadores según el momento de la corrida le gana a la mejor P fija conocida? Si sí, el estado (el
momento) aporta más allá de cualquier P fija: el argumento central de un AOS con estado.

## Diseño

- **Brazos** (catálogo de 10, intensidades genéricas, E1; mismo comando y binario que c0_10 de [22](22_operadores_actividad.md),
  `fase_ppo_bin4`, salvo la regla):
  - c0: P uniforme;
  - mw_alto: la mejor P fija conocida ([25](25_margen_pmeta_p.md)): mutar pesos +3, agregar nodo −3;
  - fases: primera mitad (τ ≤ 0,5) mutar pesos +2, agregar nodo −2, borrar conexión −1; segunda mitad sesgo +2, ganancia −1,
    mutar pesos +1.
- **Semillas nuevas** 901–1092 (192). ~175 hebra-hora.
- La regla se escribió con datos de exploración de E1 (sin información del ajuste); mw_alto usa información del ajuste y sirve
  como referencia exigente.

**Reglas fijadas antes de mirar** (Wilcoxon pareado, Holm entre las tres comparaciones):
1. **fases − mw_alto > 0, p Holm < 0,05:** el momento aporta más allá de la mejor P fija.
2. **fases − c0 > 0, p Holm < 0,05:** la regla por fases le gana al estático sin ajuste.
3. mw_alto − c0 (réplica de [25](25_margen_pmeta_p.md) en semillas nuevas).

## Resultados

192 semillas nuevas pareadas (901–1092; `analisis.py`, salida en `analisis_resultado.txt`):

| c0 | mw_alto (mejor P fija) | fases (regla por fases) |
|---|---|---|
| 0,342 | 0,465 | **0,474** |

| Comparación | Δ | Gana | p (Holm) |
|---|---|---|---|
| fases − mw_alto | +0,010 (ee 0,016) | 102/192 | 0,49 |
| fases − c0 | **+0,132** (ee 0,016) | 140/192 | 7·10⁻¹³ |
| mw_alto − c0 | **+0,123** (ee 0,016) | 135/192 | 2·10⁻¹¹ |

- **Regla 2 se cumple con holgura:** la regla por fases le gana al estático sin ajuste por +0,13 (140/192).
- **Regla 1 no se cumple:** la regla por fases no se distingue de la mejor P fija (+0,010, p 0,49). Con este tamaño de muestra
  (EE 0,016) no se detectan diferencias menores a ~0,045: el aporte del momento más allá de la mejor P fija, si existe, es chico.
- **Regla 3:** la P fija de [25](25_margen_pmeta_p.md) se replica en semillas nuevas (+0,123).
- **Transparencia sobre el origen de la regla:** la dirección "más mutar pesos y menos agregar nodo" se descubrió con la P
  ajustada por Optuna ([25](25_margen_pmeta_p.md)); los datos de exploración de E1 la respaldan para la primera mitad, y aportan
  la parte del sesgo en la segunda mitad. No es una regla obtenida solo con información de E1.

**Lectura:** en pendulum-meta, lo que da el margen (+0,12 a +0,13) es elegir bien la mezcla de operadores; una política que
cambia por fases rinde lo mismo que la mejor P fija. AP no captura nada de ese margen ([27](27_ppo_v2b.md)). El objetivo para el
AOS aprendido (PPO) queda claro: descubrir solo, desde la P uniforme, esa mezcla.

## Datos y reproducción

- `~/tesis/runs/fases_pmeta/`: `reglas/`, `preparar.py` (genera `jobs.json`; `--encolar` los agrega a la cola de `sonda_final`),
  `analisis.py`.

## Documentos relacionados

- [20](20_ppo_v2.md) · [22](22_operadores_actividad.md) · [25](25_margen_pmeta_p.md) · [27](27_ppo_v2b.md).
