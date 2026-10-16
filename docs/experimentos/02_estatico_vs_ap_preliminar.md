# 02 · Estático frente a AP: primeras curvas de aprendizaje

| | |
|---|---|
| Semana | S02 (2026-09-28; presentado el 2026-09-29) |
| Estado | terminado (exploratorio) |
| Capítulo de la tesis | Resultados preliminares · Metodología: motivación del estudio del crédito |

## Pregunta

¿Mejora Adaptive Pursuit (AP) a NEAT estático en pendulum y acrobot? ¿Qué pasa con el crédito de SNAP-NEAT?

## Diseño

- **pendulum:** 400 generaciones, 60 000 evaluaciones.
- **acrobot ±1:** 200 generaciones.
- **Parámetros comunes:** población 150, E = 10, test de validación con 10 episodios, test final con 100, reasignación de pesos
  0,02 y 10 especies objetivo.
- **11 semillas** por variante.
- **Variantes:**
  - **estático:** P de Stanley;
  - **AP:** α = 0,3, β = 0,1, crédito extremo;
  - **AP-SNAP:** crédito de SNAP-NEAT (Kohl y Miikkulainen 2012), es decir, el fitness normalizado del último operador, con
    actualización tras cada evaluación, α = β = 0,05, P_min = 0,10 y un período inicial uniforme de 1/5 de las generaciones.

## Resultados

Test final con 100 episodios (media ± DE de 11 semillas):

| Entorno | Variante | Test final | Ocultas | Conexiones | s/corrida |
|---|---|---|---|---|---|
| pendulum | estático | **−221,6 ± 36,9** | 15,2 | 57,3 | 647 |
| pendulum | AP | −360,2 ± 172,8 | 9,3 | 36,7 | 545 |
| pendulum | AP-SNAP | −559,9 ± 246,5 | 90,7 | 242,8 | 2351 |
| acrobot ±1 | estático | −110,2 ± 7,0 | 7,8 | 44,1 | 367 |
| acrobot ±1 | AP | −114,8 ± 8,8 | 4,5 | 33,5 | 323 |
| acrobot ±1 | AP-SNAP | **−105,5 ± 4,6** | 48,5 | 138,2 | 620 |

Figuras: `~/tesis/runs/reunion2_figuras/curvas_aprendizaje.{png,pdf}`. Análisis estadístico por entorno en
`~/tesis/runs/reunion2_*/analysis/report.md`.

## Conclusión y límites

- En pendulum, AP no mejora al estático. Con el crédito de SNAP-NEAT, además, infla las redes (~6× más ocultas) y es 3–4× más
  lento.
- En acrobot ±1, AP-SNAP da el mejor test, pero con redes mucho más grandes.
- El crédito es una variable crítica de AP. En la reunión del 2026-09-29 se acordó explorarlo sistemáticamente, guiándose por
  la taxonomía de Pei et al. (2025).
- **Límites:**
  - entornos simples (acrobot ±1 resultó después un ancla, con score 0,94);
  - versión de RLtools anterior a la fijada el 2026-10-03.

## Datos y reproducción

- `~/tesis/runs/reunion2_pendulum`, `reunion2_acrobot`, `reunion2_figuras`; resumen en `reunion2_resumen.txt`.
- `configs/reunion2_pendulum.json`, `reunion2_acrobot.json`, `reunion2_snap_*.json`. Lanzador `scripts/run_grid.py`.

## Documentos relacionados

- `docs/diseno_neat_ap.md` §15 (acuerdos de la reunión) y §15.1 (taxonomía de AOS y dónde caen estas variantes).
