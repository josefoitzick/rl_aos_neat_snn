#!/usr/bin/env python3
"""Análisis del barrido largo (docs/diseno_neat_ap.md §15.4.1): ¿NEAT cierra la brecha con más presupuesto? y
¿cuánto del mal desempeño es sobreajuste a los episodios fijos?

Fuentes por variante y semilla (1–5 por defecto):
  - test final (100 episodios, stream 2, no interviene en ninguna selección) del summary.txt de la corrida larga
    (2000 generaciones; ant 1000), para _fijo y _rot: comparación sin sesgo entre protocolos;
  - corte 500 del protocolo fijo = test final del barrido A (B para moving-gate): las corridas _fijo reproducen bit a bit las primeras
    500 generaciones del barrido A (mismo binario equivalente y semilla; se verifica aquí en todas las semillas);
  - curva de validación (champion_test: 10 episodios, stream 1) en los cortes 500/1000/2000:
      fijo: validación del mejor genoma en entrenamiento hasta el corte (el que entregaría pop.bestEver()), no la
            usa ninguna selección → sin sesgo, pero con 10 episodios;
      rot:  máximo de la validación hasta el corte (el genoma que entrega --resample-train) → OPTIMISTA, porque la
            misma validación lo elige; el sesgo se mide al final como best_validation_test − final_test.

Score = (x − trivial) / (RL − trivial) con el trivial y la referencia RL de ~/tesis/runs/ref_rl/score_v2.json
(scripts/ref_rl/resumen_ref.py). Las variantes sin referencia RL (flag, moving-gate) van en retorno crudo.
Corridas sin terminar: entran a la curva hasta donde lleguen y al test final solo si tienen summary.txt; la
columna n indica cuántas semillas aportan.

Uso: analisis_barrido_largo.py [SEMILLAS=1-5]. Escribe ~/tesis/runs/analisis_barrido_largo.json y .md.
"""
import csv, json, os, statistics as st, sys

R = os.path.expanduser('~/tesis/runs')
a, b = (sys.argv[1] if len(sys.argv) > 1 else '1-5').split('-')
SEMILLAS = range(int(a), int(b) + 1)

ref = {f['variante']: f for f in json.load(open(f'{R}/ref_rl/score_v2.json'))}


def kv(f):
    return dict(l.strip().split('=', 1) for l in open(f) if '=' in l)


def media(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0, len(xs)) if xs else (float('nan'), 0.0, 0)


def corrida(d):
    """Curva (best_train, champion_test) y summary de una corrida, o None si no existe."""
    if not os.path.exists(d + '/generations.csv'):
        return None
    g = list(csv.DictReader(open(d + '/generations.csv')))
    s = kv(d + '/summary.txt') if os.path.exists(d + '/summary.txt') else None
    return dict(train=[float(x['best_train']) for x in g], val=[float(x['champion_test']) for x in g], sum=s,
                filas=g)


def corte(c, n, rot):
    """Validación del genoma entregado si la corrida se detuviera en n generaciones."""
    if len(c['val']) < n:
        return None
    if rot:
        return max(c['val'][:n])
    tr = c['train'][:n]
    return c['val'][tr.index(max(tr))]


def barrido_500(base, s):
    """Directorio de la corrida de 500 generaciones (barrido A, o B para los visuales) de esa semilla."""
    for e in ('a', 'b'):
        d = f'{R}/barrido_etapa_{e}/{base}/seed_{s:02d}'
        if os.path.exists(d):
            return d
    return None


def variantes():
    for barrido in ('barrido_largo', 'barrido_largo_ant'):
        cfg = json.load(open(f'{R}/{barrido}/config.json'))
        gens = int(cfg['base'][cfg['base'].index('--generations') + 1])
        for v in cfg['variants']:
            if v.endswith('_fijo'):
                yield barrido, v[:-5], gens


filas, problemas, verificadas = [], [], 0
for barrido, base, G in variantes():
    cortes = [c for c in (500, 1000, 2000) if c <= G]
    r = dict(variante=base, G=G)
    for prot in ('fijo', 'rot'):
        fin, val, sesgo, tr = [], {c: [] for c in cortes}, [], []
        for s in SEMILLAS:
            c = corrida(f'{R}/{barrido}/{base}_{prot}/seed_{s:02d}')
            if c is None:
                continue
            for k in cortes:
                x = corte(c, k, prot == 'rot')
                if x is not None:
                    val[k].append(x)
            if c['sum']:
                fin.append(float(c['sum']['final_test']))
                tr.append(float(c['sum']['best_train']))
                if prot == 'rot':
                    sesgo.append(float(c['sum']['best_validation_test']) - float(c['sum']['final_test']))
            if prot == 'fijo':   # verificación: las primeras 500 generaciones son las del barrido A
                d5 = barrido_500(base, s)
                fa = d5 and d5 + '/generations.csv'
                if fa and os.path.exists(fa) and len(c['filas']) >= 500:
                    verificadas += 1
                    ga = list(csv.DictReader(open(fa)))[:500]
                    quitar = lambda x: {k: v for k, v in x.items() if k != 'seconds'}
                    if [quitar(x) for x in ga] != [quitar(x) for x in c['filas'][:500]]:
                        problemas.append(f'{base} seed {s}: las primeras 500 generaciones difieren del barrido A')
        r[prot] = dict(final=media(fin), train=media(tr), sesgo=media(sesgo),
                       val={k: media(v) for k, v in val.items()})
    # corte 500 sin sesgo del protocolo fijo: test final del barrido A
    a500 = [float(kv(barrido_500(base, s) + '/summary.txt')['final_test']) for s in SEMILLAS if barrido_500(base, s)]
    r['fijo_500_final'] = media(a500)
    f = ref.get(base)
    r['ref'] = dict(trivial=f['trivial'], rl=f['rl_mean'], entrenador=f['rl']) if f else None
    filas.append(r)


def sc(r, x):
    if r['ref'] is None or x != x:
        return float('nan')
    return (x - r['ref']['trivial']) / (r['ref']['rl'] - r['ref']['trivial'])


def fmt(x, d=2):
    return '  —  ' if x != x else f'{x:5.{d}f}'


# Tabla 1: scores con test final (sin sesgo)
lin = ['## Score con test final (100 episodios, sin sesgo)', '',
       '| variante | G | fijo 500 (barrido A) | fijo G | rot G | Δ fijo 500→G | Δ rot−fijo (G) | n fijo/rot |',
       '|---|---|---|---|---|---|---|---|']
for r in sorted(filas, key=lambda r: sc(r, r['fijo_500_final'][0]) if r['ref'] else 9):
    s500, sf, sr = sc(r, r['fijo_500_final'][0]), sc(r, r['fijo']['final'][0]), sc(r, r['rot']['final'][0])
    if r['ref'] is None:   # sin referencia RL: retorno crudo
        s500, sf, sr = r['fijo_500_final'][0], r['fijo']['final'][0], r['rot']['final'][0]
    lin.append(f"| {r['variante']}{'' if r['ref'] else ' (retorno, sin RL)'} | {r['G']} | {fmt(s500)} | {fmt(sf)} | "
               f"{fmt(sr)} | {fmt(sf - s500)} | {fmt(sr - sf)} | {r['fijo']['final'][2]}/{r['rot']['final'][2]} |")

# Tabla 2: curva de validación (score) en los cortes
lin += ['', '## Curva de validación por corte (score; rot es optimista)', '',
        '| variante | fijo 500 | fijo 1000 | fijo 2000 | rot 500 | rot 1000 | rot 2000 | sesgo rot (val − test, score) | n (corte mayor) fijo/rot |',
        '|---|---|---|---|---|---|---|---|---|']
for r in sorted(filas, key=lambda r: sc(r, r['fijo_500_final'][0]) if r['ref'] else 9):
    k = lambda p, c: sc(r, r[p]['val'][c][0]) if c in r[p]['val'] else float('nan')
    sesgo = r['rot']['sesgo'][0] / (r['ref']['rl'] - r['ref']['trivial']) if r['ref'] else float('nan')
    cmax = max(r['fijo']['val'])
    lin.append(f"| {r['variante']} | " + ' | '.join(fmt(k(p, c)) for p in ('fijo', 'rot') for c in (500, 1000, 2000))
               + f" | {fmt(sesgo)} | {r['fijo']['val'][cmax][2]}/{r['rot']['val'][cmax][2]} |")

lin += ['', f'Verificación: {verificadas} corridas _fijo comparadas con el barrido de 500 generaciones; '
        f'{len(problemas)} difieren.']
if problemas:
    lin += ['', '## PROBLEMAS', ''] + ['- ' + p for p in problemas]
texto = '\n'.join(lin)
print(texto)
open(f'{R}/analisis_barrido_largo.md', 'w').write(texto + '\n')
json.dump(filas, open(f'{R}/analisis_barrido_largo.json', 'w'), indent=1, ensure_ascii=False)
