# 20 · PPO versión 2: un agente por entorno con estado SNN

| | |
|---|---|
| Semana | S04 (lanzado el 2026-10-15 a las 12:08) |
| Estado | pausado (acrobot-swingup, iteración 12); pendulum-meta no se lanzó |
| Capítulo de la tesis | Resultados: AOS con estado (PPO) frente al estático y a la regla · Metodología: diseño del agente |

## Pregunta

La versión 1 del agente (plan B, [16](16_ppo_planB_e1.md) y [17](17_comparacion_e1.md)) no se distinguió del estático. El
análisis de sus decisiones ([18](18_analisis_efectos_acciones.md)) indica que el planteamiento tapaba los efectos. ¿Un agente
por entorno, con estado SNN, acción solo sobre P y pocas decisiones grandes, aprende una política que le gane al estático sin
ajuste (c0)? ¿Y a la regla escrita a mano ([19](19_regla_con_estado.md))?

## Diseño

- **Agente** (`docs/diseno_ppo.md` §11): `--estado snn` (10 rasgos), `--accion p` (K = 10), `--T 10`, `--std0 1.0`; el resto
  como en la versión 1 (γ 0,99, λ 1, 20 épocas, c = 3, w = 0,5, escala 10). Semilla de PPO 1.
- **NEAT:** configuración c0 de cada entorno (P uniforme sobre el catálogo de 10, intensidades genéricas), idéntica al brazo
  c0 de [19](19_regla_con_estado.md) (lo comprueba `preparar.py`). 75k evaluaciones por corrida, rotación.
- **Entrenamiento:** 1600 corridas por entorno (semillas 1001–2600), 16 en paralelo; validación interna de 32 corridas
  (semillas 2001–2032, contra la acción 0).
  - acrobot-swingup: lanzado el 2026-10-15 12:08 con 256 hebras.
    **Pausado el 2026-10-15 a las 17:07** en la iteración 12 (480 corridas, sin aprendizaje visible: P elegida cerca de la
    uniforme, score de entrenamiento al nivel de c0). El ruido de exploración indica efectos de cada operador sobre el resultado
    final ≤ ~0,01 en acrobot-swingup (`~/tesis/runs/efecto_final/`), difíciles de aprender desde cero; las hebras pasan a la
    ficha 22, la dosis ([24](24_dosis.md)) y el PPO residual ([23](23_ppo_residual.md)). Punto de control respaldado en
    `pausa_iter12/` y `decisiones.csv` recortado a las 4800 decisiones del punto de control (verificado: semillas 1001–1480
    completas); se reanuda exacto con `bash lanzar.sh --reanudar`.
  - pendulum-meta: se lanza solo al terminar [19](19_regla_con_estado.md) y pendulum-meta a 300k ([08](08_neat_300k.md)),
    con las hebras libres (máximo 280).
    **Detenido antes de arrancar (2026-10-15 14:05):** la regla 3 de [19](19_regla_con_estado.md) pide revisar el planteamiento
    antes de entrenar, y en pendulum-meta la regla y la P fija empeoraron frente a c0.
    **Relanzado el 2026-10-15 a las 17:46** con 2400 corridas y 192 hebras (nodo completo hasta la 01:00). Motivo: el estático
    ajustado en pendulum-meta usa casi solo mutar pesos y casi nunca agregar nodo ([22](22_operadores_actividad.md),
    [25](25_margen_pmeta_p.md)), la dirección contraria a las reglas probadas; un agente que explora en todas direcciones puede
    encontrar ese margen. Evaluación prevista en las semillas 301–600, pareada con el brazo c0 de [19](19_regla_con_estado.md).
    **Pausado el 2026-10-15 a las 19:19** en la iteración 20 (800 corridas): sin aprendizaje visible (P de mutar pesos y de
    agregar nodo en ~10 %). Lo reemplaza la variante v2b con 2 decisiones por corrida ([27](27_ppo_v2b.md)). Punto de control en
    `pendulum-meta/pausa_iter20/` y `decisiones.csv` recortado a las 8000 decisiones del punto de control (verificado).
- **Sin gemelos:** el diseño §11 los proponía en acrobot-swingup, pero el entrenador no los implementa; se entrena sin ellos
  (criterio de mínimo necesario).
- **Binario:** `fase_ppo_bin3` (`aos_ppo_train` 13c25b49…), el mismo de la regla.

**Evaluación y reglas fijadas antes de mirar:**
- La política final se evalúa con `--evaluar` en las mismas semillas que la regla: acrobot-swingup 301–320 y pendulum-meta
  301–600, pareada contra los brazos c0, fija y regla de [19](19_regla_con_estado.md) (Wilcoxon pareado, Holm entre
  PPO − c0, PPO − fija y PPO − regla de cada entorno).
1. **PPO − c0 > 0 con p Holm < 0,05:** el agente con estado captura margen sobre el estático sin ajuste.
2. **PPO − regla ≥ 0 sin diferencia significativa, o > 0:** el agente iguala o supera lo que se escribe a mano con el
   análisis de la ficha 18.
3. **Si PPO − c0 no es significativo pero la regla sí le gana a c0:** hay espacio que el agente no aprendió (presupuesto de
   entrenamiento o exploración), no ausencia de efecto.

## Resultados

Pendiente.

## Datos y reproducción

- `~/tesis/runs/ppo_v2/`: `preparar.py <entorno> <corridas> <hebras>` (genera `<entorno>/lanzar.sh`), `arrancar_pmeta.sh`
  (cadena de pendulum-meta); salida en `<entorno>/salida/` (`ppo.csv`, `corridas.csv`, `decisiones.csv`, `validacion.txt`).

## Documentos relacionados

- [16](16_ppo_planB_e1.md) · [17](17_comparacion_e1.md) · [18](18_analisis_efectos_acciones.md) · [19](19_regla_con_estado.md).
