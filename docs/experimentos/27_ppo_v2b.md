# 27 · PPO v2b en pendulum-meta: pocas decisiones grandes, contra c0 y AP

| | |
|---|---|
| Semana | S04 (lanzado el 2026-10-15 a las 20:31) |
| Estado | pausado en la iteración 4 (800 corridas; sin aprendizaje visible); AP terminado |
| Capítulo de la tesis | Resultados: AOS con estado (PPO) frente al estático y a AP · Discusión: crédito inmediato frente a resultado final |

## Pregunta

En pendulum-meta hay un margen grande que se alcanza moviendo solo P: más mutar pesos y casi nada de agregar nodo da +0,11 a
+0,12 sobre c0 ([25](25_margen_pmeta_p.md)). AP (sin estado, crédito inmediato) va en la dirección contraria: baja mutar pesos
a 4–7 % porque ese operador empeora al hijo a corto plazo (en [15](15_margen_intensidades.md), −0,078 bajo c0 con 5 semillas).

PPO v2 (10 decisiones por corrida, σ 1,0; [20](20_ppo_v2.md)) no mostró aprendizaje en 800 corridas. Un cálculo de
señal/ruido indica que, para una dirección casi fija, pocas decisiones grandes informan ~5 veces más por corrida.

¿Un agente con estado que aprende del resultado de la corrida encuentra esa dirección y le gana a c0 y a AP?

## Diseño

- **Agente:** como la versión 2 (estado SNN, acción P sobre el catálogo de 10, E1 con c0 de [19](19_regla_con_estado.md)), con
  **T = 2** (una decisión por mitad de corrida), **σ inicial 1,5** y **tasa de aprendizaje del actor 1e-3**. 2000 corridas
  (semillas 1001–3000), 192 hebras, binario `fase_ppo_bin5` (sin `--regla-base`, idéntico a `fase_ppo_bin3`). Ninguna
  información del ajuste con Optuna entra al agente.
- **Prueba de humo** (`~/tesis/runs/optim_verif/v2b/`): con T = 2 el entrenador completa iteraciones (decisiones t = 0 y 1),
  σ inicial correcto y actualización de la política.
- **AP** (sin estado) en las mismas semillas 301–600, con el comando de c3g de [17](17_comparacion_e1.md) (`fase_ppo_bin2`).
- PPO v2 (T = 10) quedó pausado en su punto de control (iteración 20, 800 corridas) para liberar sus hebras.

**Reglas fijadas antes de mirar:**
- **Primaria** (semillas 301–600, pareada con c0 de [19](19_regla_con_estado.md) y AP; Wilcoxon, Holm entre las tres):
  1. **PPO − c0 > 0, p Holm < 0,05:** el AOS con estado aprendido le gana al estático sin ajuste en pendulum-meta.
  2. **PPO − AP > 0, p Holm < 0,05:** el AOS con estado que aprende del resultado le gana al AOS sin estado de crédito inmediato.
  3. AP − c0 (se espera < 0, como en [15](15_margen_intensidades.md)).
- **Secundaria** (semillas 701–892): PPO − c0_10 y fracción del margen de la P fija de [25](25_margen_pmeta_p.md)
  ((PPO − c0_10) / (como_ajustado − c0_10)), descriptiva.

## Resultados

**AP − c0** (terminado el 2026-10-15 21:30; 300 semillas 301–600): AP 0,354, c0 0,358; Δ −0,003 (ee 0,011), 149/300, p 0,70.
Con 300 semillas, AP no empeora a c0 (el −0,078 de [15](15_margen_intensidades.md) venía de 5 semillas), pero tampoco captura
nada del margen de +0,11 de [25](25_margen_pmeta_p.md). Las comparaciones con PPO quedan pendientes de su evaluación.

**Entrenamiento de PPO v2b** (pausado el 2026-10-15 a las 22:33, en el punto de control de la iteración 4; 800 corridas, 3
actualizaciones): la política cambia en cada actualización (KL 0,03–0,08) pero sin dirección consistente; P media de mutar
pesos 10,5 → 9,0 %, agregar nodo 9,9 → 9,3 %, sesgo ~9 %; σ sin bajar (log σ ≈ 0,40); score de entrenamiento 0,335–0,354. No
aprende la dirección del margen con este presupuesto. Respaldo en `pausa_iter4/`; `decisiones.csv` recortado al punto de control
(verificado); se reanuda exacto con `lanzar.sh --reanudar` (y `evaluar.sh`, que se detuvo). Sus hebras pasan a la ficha 28.

## Datos y reproducción

- `~/tesis/runs/ppo_v2b/`: `preparar.py <entorno> <corridas> <hebras> <T> <std0> <lr>`, `lanzar_cuando_haya_espacio.sh` (lanza
  cuando el total baja a ≤ 328 y devuelve el tope a 520), `evaluar.sh` (evaluación encadenada), `analisis.py`.
- `~/tesis/runs/ap_pmeta/` (AP en 301–600, en la cola de `sonda_final`).

## Documentos relacionados

- [15](15_margen_intensidades.md) · [19](19_regla_con_estado.md) · [20](20_ppo_v2.md) · [25](25_margen_pmeta_p.md).
