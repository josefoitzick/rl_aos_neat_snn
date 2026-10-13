#!/usr/bin/env python3
"""Resumen de la referencia RL (docs/diseno_neat_ap.md §15.4, pasos 3 y 4).

Lee ~/tesis/runs/ref_rl/evaluacion.csv (rl_ref_eval vía scripts/ref_rl/evaluar_zoo.sh), zoo_propio.csv y
~/tesis/runs/analisis_barrido_ab.json (métricas de NEAT, scripts/analisis_barrido_ab.py).

Por (entorno, torque) y entrenador: media ± sd entre semillas RL del retorno en NUESTRO entorno (cada semilla RL
promediada sobre los 5 × 100 episodios finales de NEAT). Controles: acción cero y aleatoria con los mismos
episodios; trivial = el mejor de los dos. Referencia RL = el mejor entrenador (o el óptimo analítico).

Score de cada variante de NEAT = (NEAT − trivial) / (RL − trivial):
  ≥ ~0,8 NEAT resuelve (ancla) · ~0,2–0,8 NEAT lucha (objeto de estudio del AOS) · ≤ ~0,2 fuera de alcance.
Uso: resumen_ref.py [v2|v1]. Escribe ~/tesis/runs/ref_rl/score[_v2].json.
"""
import csv, json, os, re, statistics as st, sys
from collections import defaultdict

V = sys.argv[1] if len(sys.argv) > 1 else 'v2'          # v2 (OpenBLAS, 10 semillas) o v1
SUF = '' if V == 'v1' else '_' + V

R = os.path.expanduser('~/tesis/runs')
ev = list(csv.DictReader(open(f'{R}/ref_rl/evaluacion{SUF}.csv')))
neat = {r['v']: r for r in json.load(open(f'{R}/analisis_barrido_ab.json'))}

pol = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # (env, ts) → entrenador → semilla RL → medias
ctrl = defaultdict(lambda: defaultdict(list))                       # (env, ts) → zero/random → medias
for r in ev:
    clave = (r['env'], float(r['torque_scale']))
    nombre = os.path.basename(r['policy'])
    if nombre in ('zero', 'random'):
        ctrl[clave][nombre].append(float(r['mean']))
    elif nombre == 'memory_oracle.so':
        pol[clave]['óptimo analítico'][0].append(float(r['mean']))
    else:
        m = re.match(r'(.+)_seed(\d+)\.so$', nombre)
        pol[clave][m.group(1)][int(m.group(2))].append(float(r['mean']))


def variantes_neat(env, ts):
    if env == 'acrobot':
        return ['acrobot_t5'] if ts == 1 else ['acrobot_t1']
    return sorted(k for k in neat if k == env or k.startswith(env + '_'))


filas = []
print(f"{'entorno':22s} {'entrenador':34s} {'n':>2s} {'retorno (nuestro)':>20s}")
for clave in sorted(pol):
    env, ts = clave
    etiqueta = env + ('' if ts == 1 else f' (×{ts:g})')
    mejor = None
    for ent, por_semilla in sorted(pol[clave].items()):
        ms = [st.mean(v) for v in por_semilla.values()]
        sd = st.stdev(ms) if len(ms) > 1 else 0.0
        print(f"{etiqueta:22s} {ent:34s} {len(ms):2d} {st.mean(ms):11.4g} ± {sd:<7.3g}")
        if mejor is None or st.mean(ms) > mejor[1]:
            mejor = (ent, st.mean(ms), sd, len(ms))
    cero, aleat = st.mean(ctrl[clave]['zero']), st.mean(ctrl[clave]['random'])
    triv = max(cero, aleat)
    print(f"{etiqueta:22s} {'(controles: cero / aleatoria)':34s}    {cero:11.4g} / {aleat:.4g}")
    for v in variantes_neat(env, ts):
        n = neat[v]
        score = (n['test'] - triv) / (mejor[1] - triv)
        filas.append(dict(entorno=etiqueta, variante=v, neat_test=n['test'], neat_sd=n['sd'], rl=mejor[0],
                          rl_mean=mejor[1], rl_sd=mejor[2], rl_n=mejor[3], trivial=triv, score=score,
                          score_sd=n['sd'] / (mejor[1] - triv)))

print(f"\n{'variante NEAT':36s} {'NEAT':>9s} {'RL (mejor)':>10s} {'entrenador':30s} {'trivial':>9s} {'score':>12s}")
for f in sorted(filas, key=lambda f: f['score']):
    print(f"{f['variante']:36s} {f['neat_test']:9.4g} {f['rl_mean']:10.4g} {f['rl']:30s} {f['trivial']:9.4g} "
          f"{f['score']:6.2f} ± {f['score_sd']:.2f}")
json.dump(filas, open(f'{R}/ref_rl/score{SUF}.json', 'w'), indent=1, ensure_ascii=False)
