# 19 · Prueba de la regla con estado (¿hay espacio para un AOS con estado?)

| | |
|---|---|
| Semana | S04 (2026-10-15, 12:08 → 14:04) |
| Estado | terminado |
| Capítulo de la tesis | Resultados: evidencia independiente de que el estado importa · Metodología: AOS con estado para SNN |

## Pregunta

El análisis de las decisiones de PPO ([18](18_analisis_efectos_acciones.md)) sugiere que el mejor operador depende del estado:
- en acrobot-swingup, del momento de la corrida;
- en pendulum-meta, del tamaño de la red y del nivel alcanzado.

Antes de entrenar la versión 2 del agente se pregunta, **con semillas nuevas**:
- ¿Una regla escrita a mano que use ese estado le gana al estático sin ajuste?
- ¿Y a una P fija derivada de los mismos efectos?

## Diseño

- **Configuración:** c0 de cada entorno (P uniforme sobre el catálogo de 10 e intensidades genéricas), estado SNN, 10 decisiones
  por corrida, 75k evaluaciones. Binario `fase_ppo_bin3` (`aos_ppo_train --regla`), verificado contra `neat_control`.
- **Brazos:** la acción pasa por el mismo mapa que la del agente; el empuje ±1 ≈ ×2,7 / ×0,37 sobre la probabilidad.

  | Brazo | acrobot-swingup | pendulum-meta |
  |---|---|---|
  | c0 | sin empujones | sin empujones |
  | fija | pesos −, conexión −, nodo +, pesos finos +, dinámica +, podar + | pesos −, conexión −, nodo −, dinámica +, sesgo +, podar + |
  | regla | si τ ≤ 0,3: pesos finos +, dinámica +, pesos −, podar −, borrar conexión −; después: borrar conexión − | si la red es grande (log(conex/iniciales) > −0,061) o V* ≤ 0,351: pesos −, conexión −, dinámica +, sesgo +, podar +; si no, nada |

  Umbrales: medianas del entrenamiento de PPO. Archivos en `~/tesis/runs/regla_estado/reglas/`.
- **Semillas** (análisis de poder de [18](18_analisis_efectos_acciones.md)):
  - acrobot-swingup: 301–320 (20);
  - pendulum-meta: 301–600 (300).

  Ninguna se usó antes.
- **Costo estimado:** ~450 hebra-hora: 1,5 h con 300 hebras o ~3,5 h con 128.

**Reglas fijadas antes de mirar** (Wilcoxon pareado, Holm entre las 3 comparaciones de cada entorno):
1. **Regla − c0 > 0 con p Holm < 0,05:** hay espacio para un AOS con estado, confirmado con datos independientes.
2. **Regla − fija > 0 con p Holm < 0,05:** el estado aporta más allá de una P fija mejor.
3. **Si la regla no gana a c0 en ningún entorno:** los efectos de la ficha 18 no se traducen en el resultado final, y se revisa
   el planteamiento antes de entrenar.

La evaluación también registra el estado de cada decisión (`decisiones.csv`), incluida la actividad de la SNN, para diseñar
reglas basadas en actividad.

## Resultados

Score del test final (100 episodios; referencias de `score_v2_s1-10.json`), pareado por semilla, Wilcoxon con Holm entre las 3
comparaciones de cada entorno (`analisis.py`, salida en `analisis_resultado.txt`):

| Entorno (semillas) | c0 | fija | regla | regla − c0 | fija − c0 | regla − fija |
|---|---|---|---|---|---|---|
| acrobot-swingup (20) | 0,532 | 0,526 | **0,551** | +0,019 (14/20), p 0,053, Holm 0,106 | −0,006 (7/20), Holm 0,20 | +0,025 (15/20), p 0,017, Holm 0,052 |
| pendulum-meta (300) | **0,358** | 0,337 | 0,332 | −0,026 (137/300), p 0,031, Holm 0,092 | −0,021 (140/300), Holm 0,16 | −0,005, Holm 0,65 |

**Lectura con las reglas fijadas:**
- Ninguna comparación llega a p Holm < 0,05: **se aplica la regla 3** (los efectos de la ficha 18 no se traducen en una mejora
  confirmada del resultado final) y se revisa el planteamiento antes de entrenar. El entrenamiento de PPO v2 en pendulum-meta,
  que estaba encadenado, se detuvo antes de arrancar ([20](20_ppo_v2.md)).
- **acrobot-swingup:** la dirección es la esperada y está en el límite: la regla con estado le gana a la P fija derivada de los
  mismos efectos (+0,025, 15/20, Holm 0,052) y a c0 (+0,019, Holm 0,11), mientras la P fija no le gana a c0. Sugiere que en
  acrobot-swingup lo que ayuda es **cuándo** se usa cada operador (al inicio), no una P mejor.
- **pendulum-meta:** con 300 semillas, empujar hacia los operadores "que ayudan" según la ficha 18 **empeora** el resultado
  final (regla −0,026 y fija −0,021 frente a c0, p sin corregir 0,03 y 0,08). Los efectos medidos sobre el progreso a corto plazo
  (ΔΦ a 5 ventanas) no predicen el efecto sobre el resultado final, o lo invierten.

## Datos y reproducción

- `~/tesis/runs/regla_estado/`:
  - `preparar.py` (genera `jobs.json`), `lanzar.py` (planificador; tope global en `tope.txt`: 540 hebras, compartidas con [20](20_ppo_v2.md));
  - `analisis.py`;
  - resultados en `<entorno>/<brazo>/<semillas>/seed_XXX`.
- Para lanzar: `tmux new -d -s regla_estado 'python3 ~/tesis/runs/regla_estado/lanzar.py'`.
