# 01 · Núcleo NEAT + SNN: XOR, pilotos y calibración

| | |
|---|---|
| Semana | S01 (2026-09-26 y 27) |
| Estado | terminado (exploratorio) |
| Capítulo de la tesis | Metodología: implementación y calibración de NEAT sobre SNN · Anexo: pilotos |

## Pregunta

1. ¿Funciona el NEAT propio sobre SNN de Izhikevich? Se prueba en XOR, como el paper de referencia.
2. ¿Con qué parámetros base conviene correr NEAT en tareas de control?
3. ¿Qué problemas aparecen al usar Adaptive Pursuit (AP) sin ajuste?

## Diseño

- **XOR:** 20 corridas (`build/bin/xor_snn`).
- **Pilotos:** pendulum y acrobot, NEAT estático y AP, 150 generaciones, 3 semillas, 5 episodios de entrenamiento.
- **Calibración:** pendulum, 5 semillas, 150 generaciones, 8 hilos por corrida. Se variaron los episodios de entrenamiento E,
  la probabilidad de reasignar pesos, la potencia de perturbación y la especiación (umbral fijo o dinámica con 10 especies
  objetivo). Acrobot ±1 con 100 generaciones.
- **Lote preliminar de líneas base:** estático frente a AP (α = 0,3, β = 0,1, crédito extremo).
  - 11 semillas en pendulum y acrobot;
  - variantes del crédito de AP con 5 semillas en pendulum;
  - horizonte largo: acrobot con 300 generaciones y pendulum con 400.
- Población de 150. Test final con 100 episodios nuevos.
- RLtools en `b32d998` (rama main). Después se fijó la rama `render`, cuyo generador aleatorio es distinto, así que **estas
  corridas son exploratorias** y no se mezclan con las posteriores.

## Resultados

**XOR:**

| | Resueltas | Generaciones | Neuronas ocultas | Conexiones |
|---|---|---|---|---|
| Este trabajo | 20/20 | 11,4 | 2,25 | 7,55 |
| Paper de referencia | 100/100 | 32 | 2,35 | 7,48 |

La complejidad coincide. Las generaciones no son comparables: aquí (0,0) → 0 sale gratis, porque sin entrada no hay espigas, y
el criterio "dispara o no" es más laxo que el error continuo.

**Pilotos** (pendulum, 150 generaciones, 3 semillas):

| | Entrenamiento | Test final | Mejor red |
|---|---|---|---|
| Estático | −429 / −256 / −591 | −723 / −369 / −669 | 4–7 ocultas, 18–29 conexiones |
| AP | −215 / −747 / −643 | −408 / −840 / −783 | 39–63 ocultas, 104–136 conexiones |

1. **Sobreajuste** a los 5 estados iniciales de entrenamiento: hay una gran brecha entre entrenamiento y test.
2. **AP con crédito "mejora sobre el padre" infla las redes.**
   - Mutar pesos es disruptivo: mejora al padre en 0–5 % de los hijos, aunque es el motor del progreso en el estático.
   - Agregar nodo es casi neutro y mejora en ~10 %.
   - AP persigue lo neutro: P llega a (0,02; 0,13; 0,85).

   Es un ejemplo concreto de la ceguera de los métodos sin estado que motiva la tesis.
3. **Especiación mal calibrada:** en acrobot queda siempre 1 especie.
4. **Acrobot ±5 es trivial:** la población inicial ya da −31,6. Con ±1 el aprendizaje es gradual.

**Calibración** (pendulum, test final):

| Episodios de entrenamiento E | Test final | Brecha entrenamiento − test |
|---|---|---|
| 5 | −731 ± 257 | 210 |
| 10 | −641 ± 103 | 117 |
| 20 | −676 ± 126 | 72 |

| Variante (E = 10) | Test final (mediana) | Especies |
|---|---|---|
| Umbral fijo, reasignación 0,10 (Stanley) | −641 (−620) | 2,4 |
| Umbral fijo, reasignación 0,02 | **−368 (−316)** | 2,2 |
| 10 especies objetivo, reasignación 0,02 | −403 (**−264**) | 9,0 |
| 10 especies objetivo, reasignación 0,10 | −717 (−782) | 9,0 |

La misma variante corrida en dos lanzamientos distintos dio resultados idénticos: **la reproducibilidad está verificada**.

**Líneas base preliminares** (11 semillas, test final, media ± DE):

| Entorno | Estático | AP | Ocultas (estático / AP) |
|---|---|---|---|
| pendulum | −381 ± 209 | −607 ± 242 | 6,3 / 31,3 |
| acrobot ±1 | −117 ± 11 | −115 ± 8 | 4,7 / 25,1 |

Ninguna variante del crédito de AP (5 semillas) superó al estático en pendulum (−403 ± 224). La mejor fue el crédito extremo,
con −563 ± 86.

## Conclusión y límites

- **Decisiones:**
  - E = 10;
  - reasignación de pesos 0,02;
  - acrobot con torque ±1;
  - especiación dinámica con 10 especies objetivo (decidida el 2026-10-03).
- Los valores del código siguen siendo los de Stanley; los calibrados viven en `configs/*.json`.
- **Para la tesis:** sin estado, AP con crédito inmediato persigue a los operadores neutros. Es una motivación medida para el
  AOS con estado y para estudiar el crédito.
- **Límites:** pocas semillas y solo dos entornos simples. La versión de RLtools es distinta de la de las corridas posteriores.

## Datos y reproducción

- `~/tesis/runs/piloto`, `calib_episodes`, `calib_speciation_weights`, `calib_replace`, `calib_acrobot`, `lb_preliminar_*`,
  `ap_credito_pendulum` y `horizonte_static`. Resumen en `~/tesis/runs/lote_2026-09-27_resumen.txt`.
- Configuraciones en `configs/calib_*.json`, `lb_preliminar_*.json`, `ap_credito_pendulum.json` y `horizonte_static.json`.
- Lanzador `scripts/run_grid.py` (variantes × semillas, reanudable). Resumen con `analysis/summarize.py`.

## Documentos relacionados

- `docs/diseno_neat_ap.md` §14 (implementación), §14.1 (pilotos) y §14.2 (calibración).
- `docs/snn-simulator.md` (cambios al simulador).
