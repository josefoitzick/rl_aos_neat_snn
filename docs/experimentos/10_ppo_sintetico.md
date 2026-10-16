# 10 · PPO en un AOS sintético (paso 1 del agente)

| | |
|---|---|
| Semana | S04 (2026-10-13) |
| Estado | terminado |
| Capítulo de la tesis | Metodología: diseño y calibración del agente PPO · Anexo: verificación |

## Pregunta

Antes de gastar clúster en NEAT real:
1. ¿Funciona la integración con el PPO de RLtools (lote propio, contabilidad de corridas)?
2. ¿Aprende PPO con un ruido como el nuestro?
3. ¿Cuántas corridas de entrenamiento hacen falta?
4. ¿Qué hiperparámetros de PPO convienen?

## Diseño

- **MDP del agente** (`docs/diseno_ppo.md` §2):
  - T = 50 decisiones por corrida;
  - acción a ∈ ℝᴷ → P = softmax(log P₀ + c·tanh(a/c)), con c = 3 y piso de P₀ = 0,002;
  - recompensa = escala · [(Φ' − Φ) + (w/T)·Φ'], con w = 0,5 y escala 10, donde Φ es el score del campeón.
- **AOS sintético** (`src/experiment/aos_toy.hpp`):
  - K = 8 operadores y 30 observaciones;
  - un oráculo con estado que supera a la mejor P fija;
  - calibrado para que la desviación estándar entre semillas y la ganancia del oráculo sean del orden de las de NEAT.
- **Lote propio de RLtools** (`ppo_lote::Lote`): 16 corridas en paralelo. El núcleo de RLtools fija un lote secuencial.
- **Corridas:** 4 baterías, 290 entrenamientos, 10 semillas de PPO por configuración.
- **Medición:** Δ = score de PPO − score de P₀, en las mismas 1920–2000 corridas de evaluación, con la política determinista.
- **Controles:**
  - recompensa barajada (sin señal);
  - entorno con óptimo estático (no hay nada que aprender).

## Resultados

**Calibración del sintético** (4000 corridas):

| Política | Score | DE |
|---|---|---|
| Uniforme | 0,256 | — |
| Mejor P fija | 0,681 | 0,046 |
| Oráculo con estado | 0,734 | 0,045 |

Ganancia del oráculo sobre la mejor P fija: +0,054.

**Δ de PPO** según las corridas de entrenamiento:

| Configuración | 96 | 192 | 384 | 768 | Semillas que aprenden a 768 |
|---|---|---|---|---|---|
| Inicial (γ = 1, λ = 0,95, 10 épocas) | −0,003 | −0,010 | +0,008 | +0,038 | 10/10 |
| γ = 0,99, λ = 1 | +0,003 | +0,008 | +0,018 | +0,040 | 10/10 |
| … + 20 épocas | +0,007 | +0,018 | +0,033 | +0,046 | 10/10 |
| **… + crítico con lr 1e-3 (final)** | **+0,009** | **+0,018** | **+0,033** | **+0,046** | **10/10** |
| Control: recompensa barajada (inicial) | −0,006 | −0,024 | −0,070 | −0,081 | 0/10 |
| Control: recompensa barajada (final) | −0,003 | −0,002 | −0,011 | −0,009 | 3/10 |
| Control: óptimo estático (final) | 0,000 | 0,000 | 0,000 | −0,0004 | — |

**Variantes:**
- **Empeoran:**
  - lr 1e-4: −0,033 a 768 corridas;
  - 4 épocas;
  - N = 8 corridas en paralelo: −0,077 a 192;
  - σ inicial 0,3: más lento;
  - λ = 0,9.
- **Sin efecto:** w = 0 frente a 0,5, precalentar el crítico, N = 32 y recorte 0,3.

Figura: `~/tesis/runs/ppo_sintetico/curvas_sintetico.png`.

## Conclusión y límites

- **La integración funciona y PPO aprende** con un ruido como el nuestro: 10/10 semillas, ~85 % de la ganancia alcanzable a
  768 corridas.
- **192 corridas no alcanzan** (a lo más ~33 % de la ganancia). El presupuesto de entrenamiento pasa a **≥ 400 corridas por
  entorno**, idealmente ~800.
- **Configuración final:**

  | Parámetro | Valor |
  |---|---|
  | γ, λ | 0,99, 1 |
  | Épocas | 20 |
  | Corridas en paralelo × pasos por iteración | 16 × 25 |
  | Minilote | 100 |
  | Actor | 2 × 64 (tanh rápida), lr 3e-4, σ inicial 0,5, última capa × 0,01 |
  | Crítico | 2 × 128, lr 1e-3 |

- **Sin señal, PPO deriva.** La configuración final deriva poco (−0,01). Aun así, validar contra P₀ y un pre-chequeo de
  señal siguen siendo necesarios.
- **Límite:** el sintético es una caricatura calibrada del ruido de NEAT, no NEAT.

## Datos y reproducción

- `~/tesis/runs/ppo_sintetico/` (`lanzar*.sh`, `logs*`, `resumen.py`, `figura.py`, `bin/`).
- `src/aos/action_map.hpp`, `src/experiment/aos_toy.hpp`, `src/experiment/ppo_lote.hpp`, `src/experiment/aos_toy_ppo.cpp`
  (modos `calibrar`, `evaluar`, `entrenar`, `verificar`); `tests/ppo_tests.cpp`.

## Documentos relacionados

- `docs/diseno_ppo.md` §2 (MDP), §4 (entrenamiento) y §5.1 (resultados).
