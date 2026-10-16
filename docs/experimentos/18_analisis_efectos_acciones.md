# 18 · Análisis de los efectos de las acciones del agente (diagnóstico del AOS con estado)

| | |
|---|---|
| Semana | S04 (2026-10-15) |
| Estado | terminado |
| Capítulo de la tesis | Resultados: ¿hay algo que aprender? · Discusión: por qué el AOS con estado no se distinguió · Metodología: poder estadístico |

## Pregunta

PPO no se distinguió del estático en la comparación final ([17](17_comparacion_e1.md)). ¿Por qué?
- ¿Las acciones no mueven NEAT?
- ¿Lo mueven, pero de una forma que una P fija ya captura?
- ¿O el efecto existe, depende del estado y el ruido lo tapa?

## Diseño

**Datos:** las 70.400 decisiones del entrenamiento del plan B ([16](16_ppo_planB_e1.md), `decisiones.csv`). En cada una hay
estado, P e intensidades decididas, y Φ antes y después.

**Por qué el análisis es causal:** PPO explora con ruido gaussiano independiente del estado, así que cada decisión es un pequeño
experimento aleatorizado.

**Modelo:** regresión del progreso ΔΦ en las 5 ventanas siguientes sobre log P (contraste de suma cero: efecto de cada operador
respecto del promedio) y las intensidades, controlando el estado (9 rasgos). Se hace por entorno, por tramo de la corrida y por
estado de la red. Se marcan los efectos con |t| > 3.

**Ruido de la evaluación:** desviación entre semillas y correlación entre configuraciones con la misma semilla, en la
comparación final (semillas 1–11).

Script: `~/tesis/runs/ppo_planB_e1/analisis/` (copias de `efectos.py`, `interaccion2.py` e `interaccion3.py`).

## Resultados

**1. Las acciones sí mueven NEAT, con efectos consistentes y detectables** (todo el tramo t 0–44; ×10⁻³ de ΔΦ por +1 en log P):

| Operador | acrobot-swingup | pendulum-meta |
|---|---|---|
| mutar pesos (global, potencia 5) | **−1,95** | **−1,92** |
| agregar conexión | **−1,42** | **−1,53** |
| agregar nodo | +0,81 | **−1,45** |
| dinámica neuronal | **+1,83** | **+1,52** |
| sesgo neuronal | +0,28 | **+3,04** |
| podar nodo | +1,10 | **+1,86** |

Las intensidades no tienen efecto detectable (|t| < 1,5), salvo reasignar en pendulum-meta (t 2,4).

**2. El mejor operador depende del entorno y del estado** (es la base de un AOS con estado):
- **Del entorno:** agregar nodo ayuda en acrobot-swingup y daña en pendulum-meta. El agente entrenado con los dos entornos
  mezclados subió agregar nodo en pendulum-meta (0,104 → 0,128), justo donde daña, y su score de entrenamiento allí cayó.
  Respalda un agente por entorno.
- **Del momento de la corrida:**
  - en acrobot-swingup, casi todo el efecto está al inicio (t 0–14: pesos finos +3,2, dinámica +3,1, podar −2,1); a la mitad y
    al final casi nada importa;
  - en pendulum-meta, el sesgo y los pesos finos cambian de signo entre el inicio y el final.
- **Del estado de la red:**
  - en acrobot-swingup, los efectos aparecen con redes chicas y con V* bajo (dinámica +2,1 a +2,5; mutar pesos −2,0) y
    desaparecen con redes grandes o con V* alto;
  - en pendulum-meta, aparecen con redes grandes (sesgo +3,9; mutar pesos −3,1) y con V* bajo (sesgo +3,9; conexión −2,6).

**3. El tamaño del efecto es chico frente al ruido de la evaluación:**

| Entorno | DE entre semillas | ρ (misma semilla) | DE de una diferencia pareada | Semillas para detectar 0,03 (poder 0,8) |
|---|---|---|---|---|
| acrobot-swingup | 0,042 | 0,78 | 0,028 | ~7 |
| bottleneck | 0,082 | 0,07 | 0,112 | ~110 |
| pendulum-meta | 0,147 | 0,26 | 0,179 | ~280 |

Un efecto acumulado de ~0,02–0,04 en el score final es coherente con los márgenes medidos ([13](13_prueba_margen.md)). Con 11
semillas, **solo acrobot-swingup puede detectarlo**; en pendulum-meta y bottleneck la comparación final no tenía poder.

## Conclusión y límites

- **Hay algo que aprender, y depende del estado:** del entorno, del momento de la corrida, del tamaño de la red y del nivel
  alcanzado. Es la dependencia que plantea SNAP-NEAT.
- **El planteamiento anterior lo tapaba:**
  - mezclar entornos con óptimos opuestos;
  - acciones de intensidad sin efecto, que agregan ruido;
  - evaluar con 11 semillas donde se necesitan 100–300.
- **El pareo por semilla (gemelos) solo ayuda en acrobot-swingup** (ρ 0,78). En los otros dos entornos, la variabilidad está
  dentro de la corrida, no en la semilla.
- **Límites:**
  - regresión lineal en log P;
  - ΔΦ a 5 ventanas como aproximación del efecto sobre el score final;
  - los efectos por subgrupo tienen errores estándar de 0,3–1,0 × 10⁻³ y deben confirmarse con una prueba independiente
    (semillas nuevas).

## Datos y reproducción

- Datos: `~/tesis/runs/ppo_planB_e1/salida/decisiones.csv` y `corridas.csv`; comparación final en `~/tesis/runs/comparacion_final/`.
