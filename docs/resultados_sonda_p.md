# Resultados de la sonda de P (2026-10-09)

Copia de respaldo dentro del repo de `~/tesis/runs/analisis_sonda_p.md`, generado por `scripts/analisis_sonda_p.py`
(con el python del entorno conda `tesis-analisis`). Diseño, verificación y lectura en `docs/diseno_neat_ap.md` §16–16.1.

- 3 entornos (bottleneck, acrobot-swingup, l2f) × 2 protocolos (fijo, rotación) × 10 configuraciones de P × 10 semillas
  × 500 generaciones = 600 corridas, sin errores. Binario `~/tesis/runs/sonda_p_bin/neat_control` (sha 245893d0e9eb3b68).
- Score = (test final − trivial) / (RL − trivial), con la referencia de `~/tesis/runs/ref_rl/score_v2_s1-10.json`.
- P = (mutar pesos, agregar conexión, agregar nodo), normalizado. Los cambios ocurren en la generación 250.
- "val. gen 250 → 500": score de validación (10 episodios) del campeón en esas generaciones.
- p (Holm): Mann–Whitney contra Stanley, corregido por Holm dentro de cada entorno y protocolo.

## Resumen

| Entorno | ¿Importa P? (Kruskal–Wallis fijo / rot) | Cambio a mitad vs mejor P fijo | Ablaciones vs Stanley |
|---|---|---|---|
| bottleneck | p = 0,065 / 0,022: mucha estructura perjudica, más pesos ayuda | no lo supera | sin pesos → ~0 (p 0,015 / 0,009); sin conexión y sin nodo no cambian |
| acrobot-swingup | p = 0,98 / 0,36: todos 0,43–0,53 | no | ninguna cambia (sin pesos ni sin nodo con 26 conexiones) |
| l2f | p = 0,97 / 0,93: todos 0,02–0,11 (efecto piso) | no | ninguna cambia |

## bottleneck

Referencia: trivial 0, RL 61.86 (bottleneck-v0_ppo). Control stanley+fijo = barrido largo (500 gen.): 10/10 semillas idénticas.

### bottleneck · episodios fijos

| P | P (pesos/conex./nodo) | n | score ± sd | mediana | Δ vs Stanley | p (Holm) | val. gen 250 → 500 | conexiones | horas |
|---|---|---|---|---|---|---|---|---|---|
| stanley | 0.8,0.05,0.03 | 10 | 0,27 ± 0,12 | 0,32 |  |  | 0,18 → 0,26 | 143 | 5.4 |
| uniforme | 1,1,1 | 10 | 0,13 ± 0,07 | 0,13 | −0,14 | 0.058 | 0,05 → 0,11 | 315 | 12.0 |
| pesos | 0.98,0.01,0.01 | 10 | 0,24 ± 0,12 | 0,28 | −0,03 | 0.689 | 0,18 → 0,25 | 108 | 4.5 |
| estructura | 0.5,0.25,0.25 | 10 | 0,16 ± 0,13 | 0,14 | −0,12 | 0.625 | 0,10 → 0,14 | 322 | 11.4 |
| nodos | 0.6,0.1,0.3 | 10 | 0,23 ± 0,08 | 0,25 | −0,04 | 0.625 | 0,22 → 0,25 | 256 | 12.1 |
| estr_a_pesos | 0.5,0.25,0.25 → 0.98,0.01,0.01 (gen. 250) | 10 | 0,22 ± 0,14 | 0,30 | −0,06 | 0.625 | 0,10 → 0,19 | 231 | 9.6 |
| pesos_a_estr | 0.98,0.01,0.01 → 0.5,0.25,0.25 (gen. 250) | 10 | 0,24 ± 0,15 | 0,29 | −0,03 | 0.689 | 0,18 → 0,23 | 146 | 5.8 |
| sin_conexion | 0.8,0,0.03 | 10 | 0,24 ± 0,08 | 0,27 | −0,03 | 0.625 | 0,22 → 0,28 | 122 | 5.7 |
| sin_nodo | 0.8,0.05,0 | 10 | 0,22 ± 0,08 | 0,23 | −0,05 | 0.316 | 0,16 → 0,24 | 107 | 4.1 |
| sin_pesos | 0,0.05,0.03 | 10 | 0,02 ± 0,07 | −0,00 | −0,25 | 0.015 | 0,02 → 0,02 | 389 | 12.6 |

- **¿Importa P?** Kruskal–Wallis entre 7 P sin ablación: p = 0.065.
- **¿Cambiar a mitad de camino supera al mejor P fijo?** mejor fijo = stanley (0,27), mejor cambio = pesos_a_estr (0,24); Mann–Whitney p = 0.68.
- **Ablaciones vs Stanley:** sin_conexion Δ −0,03 (p Holm 0.625); sin_nodo Δ −0,05 (p Holm 0.316); sin_pesos Δ −0,25 (p Holm 0.015)

### bottleneck · rotación

| P | P (pesos/conex./nodo) | n | score ± sd | mediana | Δ vs Stanley | p (Holm) | val. gen 250 → 500 | conexiones | horas |
|---|---|---|---|---|---|---|---|---|---|
| stanley | 0.8,0.05,0.03 | 10 | 0,27 ± 0,06 | 0,27 |  |  | 0,21 → 0,25 | 153 | 5.8 |
| uniforme | 1,1,1 | 10 | 0,24 ± 0,08 | 0,24 | −0,03 | 1.000 | 0,16 → 0,20 | 436 | 17.1 |
| pesos | 0.98,0.01,0.01 | 10 | 0,32 ± 0,05 | 0,32 | 0,05 | 0.454 | 0,23 → 0,26 | 111 | 4.6 |
| estructura | 0.5,0.25,0.25 | 10 | 0,26 ± 0,06 | 0,27 | −0,01 | 1.000 | 0,13 → 0,26 | 448 | 13.9 |
| nodos | 0.6,0.1,0.3 | 10 | 0,29 ± 0,09 | 0,29 | 0,02 | 1.000 | 0,22 → 0,23 | 368 | 15.8 |
| estr_a_pesos | 0.5,0.25,0.25 → 0.98,0.01,0.01 (gen. 250) | 10 | 0,26 ± 0,03 | 0,28 | −0,00 | 1.000 | 0,13 → 0,27 | 279 | 11.1 |
| pesos_a_estr | 0.98,0.01,0.01 → 0.5,0.25,0.25 (gen. 250) | 10 | 0,32 ± 0,05 | 0,31 | 0,06 | 0.448 | 0,23 → 0,29 | 237 | 7.1 |
| sin_conexion | 0.8,0,0.03 | 10 | 0,32 ± 0,03 | 0,32 | 0,05 | 0.206 | 0,29 → 0,25 | 130 | 5.9 |
| sin_nodo | 0.8,0.05,0 | 10 | 0,30 ± 0,06 | 0,29 | 0,03 | 1.000 | 0,21 → 0,25 | 108 | 4.0 |
| sin_pesos | 0,0.05,0.03 | 10 | 0,08 ± 0,10 | 0,07 | −0,19 | 0.009 | 0,06 → 0,07 | 391 | 19.4 |

- **¿Importa P?** Kruskal–Wallis entre 7 P sin ablación: p = 0.022.
- **¿Cambiar a mitad de camino supera al mejor P fijo?** mejor fijo = pesos (0,32), mejor cambio = pesos_a_estr (0,32); Mann–Whitney p = 0.85.
- **Ablaciones vs Stanley:** sin_conexion Δ 0,05 (p Holm 0.206); sin_nodo Δ 0,03 (p Holm 1.000); sin_pesos Δ −0,19 (p Holm 0.009)

## acrobot-swingup

Referencia: trivial -18.04, RL -6.802 (acrobot-swingup-v0_sac). Control stanley+fijo = barrido largo (500 gen.): 10/10 semillas idénticas.

### acrobot-swingup · episodios fijos

| P | P (pesos/conex./nodo) | n | score ± sd | mediana | Δ vs Stanley | p (Holm) | val. gen 250 → 500 | conexiones | horas |
|---|---|---|---|---|---|---|---|---|---|
| stanley | 0.8,0.05,0.03 | 10 | 0,43 ± 0,09 | 0,45 |  |  | 0,40 → 0,36 | 63 | 2.1 |
| uniforme | 1,1,1 | 10 | 0,45 ± 0,08 | 0,46 | 0,01 | 1.000 | 0,36 → 0,37 | 194 | 5.1 |
| pesos | 0.98,0.01,0.01 | 10 | 0,44 ± 0,06 | 0,45 | 0,01 | 1.000 | 0,34 → 0,37 | 35 | 1.5 |
| estructura | 0.5,0.25,0.25 | 10 | 0,45 ± 0,06 | 0,45 | 0,02 | 1.000 | 0,43 → 0,42 | 200 | 4.9 |
| nodos | 0.6,0.1,0.3 | 10 | 0,45 ± 0,08 | 0,48 | 0,02 | 1.000 | 0,38 → 0,36 | 188 | 5.5 |
| estr_a_pesos | 0.5,0.25,0.25 → 0.98,0.01,0.01 (gen. 250) | 10 | 0,46 ± 0,05 | 0,46 | 0,03 | 1.000 | 0,43 → 0,44 | 144 | 4.1 |
| pesos_a_estr | 0.98,0.01,0.01 → 0.5,0.25,0.25 (gen. 250) | 10 | 0,44 ± 0,06 | 0,44 | 0,01 | 1.000 | 0,34 → 0,39 | 92 | 2.3 |
| sin_conexion | 0.8,0,0.03 | 10 | 0,44 ± 0,06 | 0,44 | 0,00 | 1.000 | 0,43 → 0,39 | 58 | 2.3 |
| sin_nodo | 0.8,0.05,0 | 10 | 0,44 ± 0,07 | 0,45 | 0,01 | 1.000 | 0,34 → 0,41 | 26 | 1.2 |
| sin_pesos | 0,0.05,0.03 | 10 | 0,43 ± 0,13 | 0,47 | 0,00 | 1.000 | 0,37 → 0,37 | 263 | 5.4 |

- **¿Importa P?** Kruskal–Wallis entre 7 P sin ablación: p = 0.98.
- **¿Cambiar a mitad de camino supera al mejor P fijo?** mejor fijo = nodos (0,45), mejor cambio = estr_a_pesos (0,46); Mann–Whitney p = 0.91.
- **Ablaciones vs Stanley:** sin_conexion Δ 0,00 (p Holm 1.000); sin_nodo Δ 0,01 (p Holm 1.000); sin_pesos Δ 0,00 (p Holm 1.000)

### acrobot-swingup · rotación

| P | P (pesos/conex./nodo) | n | score ± sd | mediana | Δ vs Stanley | p (Holm) | val. gen 250 → 500 | conexiones | horas |
|---|---|---|---|---|---|---|---|---|---|
| stanley | 0.8,0.05,0.03 | 10 | 0,48 ± 0,04 | 0,50 |  |  | 0,44 → 0,44 | 83 | 2.4 |
| uniforme | 1,1,1 | 10 | 0,52 ± 0,03 | 0,52 | 0,03 | 0.431 | 0,45 → 0,48 | 362 | 8.5 |
| pesos | 0.98,0.01,0.01 | 10 | 0,49 ± 0,04 | 0,50 | 0,01 | 1.000 | 0,44 → 0,43 | 34 | 1.8 |
| estructura | 0.5,0.25,0.25 | 10 | 0,50 ± 0,04 | 0,51 | 0,02 | 1.000 | 0,47 → 0,42 | 245 | 7.2 |
| nodos | 0.6,0.1,0.3 | 10 | 0,51 ± 0,04 | 0,52 | 0,03 | 1.000 | 0,44 → 0,46 | 250 | 8.1 |
| estr_a_pesos | 0.5,0.25,0.25 → 0.98,0.01,0.01 (gen. 250) | 10 | 0,48 ± 0,04 | 0,51 | −0,00 | 1.000 | 0,47 → 0,43 | 217 | 5.8 |
| pesos_a_estr | 0.98,0.01,0.01 → 0.5,0.25,0.25 (gen. 250) | 10 | 0,49 ± 0,05 | 0,51 | 0,00 | 1.000 | 0,44 → 0,45 | 102 | 3.2 |
| sin_conexion | 0.8,0,0.03 | 10 | 0,49 ± 0,05 | 0,50 | 0,00 | 1.000 | 0,44 → 0,45 | 53 | 2.5 |
| sin_nodo | 0.8,0.05,0 | 10 | 0,48 ± 0,04 | 0,48 | −0,00 | 1.000 | 0,42 → 0,44 | 26 | 1.2 |
| sin_pesos | 0,0.05,0.03 | 10 | 0,53 ± 0,06 | 0,54 | 0,04 | 0.339 | 0,46 → 0,46 | 419 | 8.4 |

- **¿Importa P?** Kruskal–Wallis entre 7 P sin ablación: p = 0.36.
- **¿Cambiar a mitad de camino supera al mejor P fijo?** mejor fijo = uniforme (0,52), mejor cambio = pesos_a_estr (0,49); Mann–Whitney p = 0.24.
- **Ablaciones vs Stanley:** sin_conexion Δ 0,00 (p Holm 1.000); sin_nodo Δ −0,00 (p Holm 1.000); sin_pesos Δ 0,04 (p Holm 0.339)

## l2f

Referencia: trivial 4.176, RL 30.61 (tesis-l2f-full_sac). Control stanley+fijo = barrido largo (500 gen.): 10/10 semillas idénticas.

### l2f · episodios fijos

| P | P (pesos/conex./nodo) | n | score ± sd | mediana | Δ vs Stanley | p (Holm) | val. gen 250 → 500 | conexiones | horas |
|---|---|---|---|---|---|---|---|---|---|
| stanley | 0.8,0.05,0.03 | 10 | 0,02 ± 0,05 | 0,03 |  |  | 0,08 → 0,07 | 1345 | 1.2 |
| uniforme | 1,1,1 | 10 | 0,04 ± 0,05 | 0,04 | 0,02 | 1.000 | 0,07 → 0,08 | 1388 | 1.4 |
| pesos | 0.98,0.01,0.01 | 10 | 0,04 ± 0,05 | 0,05 | 0,02 | 1.000 | 0,09 → 0,06 | 1324 | 1.2 |
| estructura | 0.5,0.25,0.25 | 10 | 0,04 ± 0,05 | 0,05 | 0,02 | 1.000 | 0,07 → 0,08 | 1448 | 1.4 |
| nodos | 0.6,0.1,0.3 | 10 | 0,03 ± 0,06 | 0,03 | 0,01 | 1.000 | 0,08 → 0,08 | 1438 | 1.5 |
| estr_a_pesos | 0.5,0.25,0.25 → 0.98,0.01,0.01 (gen. 250) | 10 | 0,04 ± 0,04 | 0,06 | 0,02 | 1.000 | 0,07 → 0,09 | 1436 | 1.4 |
| pesos_a_estr | 0.98,0.01,0.01 → 0.5,0.25,0.25 (gen. 250) | 10 | 0,04 ± 0,04 | 0,05 | 0,02 | 1.000 | 0,09 → 0,08 | 1361 | 1.3 |
| sin_conexion | 0.8,0,0.03 | 10 | 0,03 ± 0,06 | 0,03 | 0,01 | 1.000 | 0,10 → 0,05 | 1327 | 1.2 |
| sin_nodo | 0.8,0.05,0 | 10 | 0,04 ± 0,05 | 0,04 | 0,02 | 1.000 | 0,07 → 0,07 | 1328 | 1.2 |
| sin_pesos | 0,0.05,0.03 | 10 | 0,05 ± 0,03 | 0,05 | 0,02 | 1.000 | 0,08 → 0,07 | 1408 | 1.2 |

- **¿Importa P?** Kruskal–Wallis entre 7 P sin ablación: p = 0.97.
- **¿Cambiar a mitad de camino supera al mejor P fijo?** mejor fijo = uniforme (0,04), mejor cambio = pesos_a_estr (0,04); Mann–Whitney p = 0.85.
- **Ablaciones vs Stanley:** sin_conexion Δ 0,01 (p Holm 1.000); sin_nodo Δ 0,02 (p Holm 1.000); sin_pesos Δ 0,02 (p Holm 1.000)

### l2f · rotación

| P | P (pesos/conex./nodo) | n | score ± sd | mediana | Δ vs Stanley | p (Holm) | val. gen 250 → 500 | conexiones | horas |
|---|---|---|---|---|---|---|---|---|---|
| stanley | 0.8,0.05,0.03 | 10 | 0,09 ± 0,03 | 0,09 |  |  | 0,10 → 0,08 | 1345 | 1.0 |
| uniforme | 1,1,1 | 10 | 0,11 ± 0,04 | 0,11 | 0,01 | 1.000 | 0,10 → 0,08 | 1559 | 1.5 |
| pesos | 0.98,0.01,0.01 | 10 | 0,10 ± 0,03 | 0,10 | 0,01 | 1.000 | 0,10 → 0,12 | 1320 | 1.0 |
| estructura | 0.5,0.25,0.25 | 10 | 0,09 ± 0,03 | 0,09 | 0,00 | 1.000 | 0,11 → 0,10 | 1472 | 1.3 |
| nodos | 0.6,0.1,0.3 | 10 | 0,10 ± 0,06 | 0,09 | 0,01 | 1.000 | 0,10 → 0,12 | 1428 | 1.4 |
| estr_a_pesos | 0.5,0.25,0.25 → 0.98,0.01,0.01 (gen. 250) | 10 | 0,10 ± 0,04 | 0,09 | 0,00 | 1.000 | 0,11 → 0,11 | 1435 | 1.2 |
| pesos_a_estr | 0.98,0.01,0.01 → 0.5,0.25,0.25 (gen. 250) | 10 | 0,11 ± 0,04 | 0,10 | 0,01 | 1.000 | 0,10 → 0,13 | 1400 | 1.1 |
| sin_conexion | 0.8,0,0.03 | 10 | 0,09 ± 0,04 | 0,08 | 0,00 | 1.000 | 0,10 → 0,10 | 1330 | 1.0 |
| sin_nodo | 0.8,0.05,0 | 10 | 0,09 ± 0,04 | 0,10 | −0,00 | 1.000 | 0,11 → 0,08 | 1325 | 0.9 |
| sin_pesos | 0,0.05,0.03 | 10 | 0,10 ± 0,03 | 0,10 | 0,01 | 1.000 | 0,14 → 0,11 | 1298 | 1.5 |

- **¿Importa P?** Kruskal–Wallis entre 7 P sin ablación: p = 0.93.
- **¿Cambiar a mitad de camino supera al mejor P fijo?** mejor fijo = uniforme (0,11), mejor cambio = pesos_a_estr (0,11); Mann–Whitney p = 0.73.
- **Ablaciones vs Stanley:** sin_conexion Δ 0,00 (p Holm 1.000); sin_nodo Δ −0,00 (p Holm 1.000); sin_pesos Δ 0,01 (p Holm 1.000)

