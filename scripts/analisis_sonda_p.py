#!/usr/bin/env python3
"""Análisis de la sonda de P (docs/diseno_neat_ap.md §16, configs/sonda_p.json).

Por entorno y protocolo (fijo, rotación), con las semillas terminadas de cada configuración:
  - test final (100 episodios, stream 2) → score = (test − trivial) / (RL − trivial), con el trivial y la RL de
    ~/tesis/runs/ref_rl/score_v2_s1-10.json (mismos episodios, semillas 1–10);
  - tamaño del genoma final (conexiones, ocultas) y horas por corrida;
  - validación del campeón en la generación 250 (antes del cambio de P) y 500, para leer los cambios a mitad de camino.
Preguntas:
  1. ¿Importa P? Kruskal–Wallis entre los 7 P sin ablación (5 fijos + 2 con cambio).
  2. ¿Un cambio a mitad de camino supera al mejor P fijo? Mann–Whitney (dos colas) del mejor cambio contra el mejor
     P fijo; el "mejor" se elige con los mismos datos, así que la comparación es optimista para ambos lados.
  3. ¿Sobra algún operador? Mann–Whitney de cada ablación contra Stanley.
  Las comparaciones contra Stanley se corrigen por Holm dentro de cada entorno y protocolo.
Control: stanley + fijo debe reproducir bit a bit las primeras 500 generaciones del barrido largo (sin la columna
seconds), en cada semilla.

Uso (scipy está en el entorno conda tesis-analisis):
  ~/miniconda3/envs/tesis-analisis/bin/python scripts/analisis_sonda_p.py [entorno ...]
Escribe ~/tesis/runs/analisis_sonda_p.md y .json.
"""
import csv, json, os, statistics as st, sys
from scipy.stats import kruskal, mannwhitneyu

R = os.path.expanduser('~/tesis/runs')
cfg = json.load(open(f'{R}/sonda_p/config.json'))
ref = {r['variante']: r for r in json.load(open(f'{R}/ref_rl/score_v2_s1-10.json'))}
ENTORNOS = sys.argv[1:] or ['bottleneck', 'acrobot-swingup', 'l2f']
CONFIG = ['stanley', 'uniforme', 'pesos', 'estructura', 'nodos', 'estr_a_pesos', 'pesos_a_estr',
          'sin_conexion', 'sin_nodo', 'sin_pesos']
FIJOS = ['stanley', 'uniforme', 'pesos', 'estructura', 'nodos']
CAMBIOS = ['estr_a_pesos', 'pesos_a_estr']
ABL = ['sin_conexion', 'sin_nodo', 'sin_pesos']
P = {v: x[x.index('--p') + 1] + (f" → {x[x.index('--p2') + 1]} (gen. {x[x.index('--p-switch') + 1]})" if '--p2' in x else '')
     for v, x in cfg['variants'].items()}


def kv(f):
    return dict(l.strip().split('=', 1) for l in open(f) if '=' in l)


def corrida(e, prot, c, s):
    d = f'{R}/sonda_p/{e}_{prot}_{c}/seed_{s:02d}'
    if not os.path.exists(d + '/summary.txt'):
        return None
    k = kv(d + '/summary.txt')
    g = list(csv.DictReader(open(d + '/generations.csv')))
    return dict(test=float(k['final_test']), conns=int(k['best_conns']), hidden=int(k['best_hidden']),
                horas=float(k['seconds']) / 3600, v250=float(g[249]['champion_test']), v500=float(g[-1]['champion_test']),
                mean_conns=float(g[-1]['mean_conns']), filas=g, dir=d)


def holm(ps):
    orden = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, m, prev = [0.0] * len(ps), len(ps), 0.0
    for r, i in enumerate(orden):
        prev = max(prev, min(1.0, (m - r) * ps[i]))
        adj[i] = prev
    return adj


def control(e):
    """stanley + fijo vs barrido largo: primeras 500 generaciones idénticas (sin seconds)."""
    ok = tot = 0
    for s in range(1, 11):
        a = f'{R}/sonda_p/{e}_fijo_stanley/seed_{s:02d}/generations.csv'
        b = f'{R}/barrido_largo/{e}_fijo/seed_{s:02d}/generations.csv'
        if not (os.path.exists(a) and os.path.exists(b) and os.path.exists(a.replace('generations.csv', 'summary.txt'))):
            continue
        q = lambda f: [{k: v for k, v in r.items() if k != 'seconds'} for r in list(csv.DictReader(open(f)))[:500]]
        tot += 1
        ok += q(a) == q(b)
    return ok, tot


salida, lin = {}, []
f2 = lambda x: f'{x:.2f}'.replace('.', ',').replace('-', '−')
for e in ENTORNOS:
    r = ref[e]
    sc = lambda x: (x - r['trivial']) / (r['rl_mean'] - r['trivial'])
    ok, tot = control(e)
    lin += [f'## {e}', '', f'Referencia: trivial {r["trivial"]:.4g}, RL {r["rl_mean"]:.4g} ({r["rl"]}). '
            f'Control stanley+fijo = barrido largo (500 gen.): {ok}/{tot} semillas idénticas.', '']
    salida[e] = {'control': [ok, tot]}
    for prot in ('fijo', 'rot'):
        D = {c: [x for s in range(1, 11) if (x := corrida(e, prot, c, s))] for c in CONFIG}
        S = {c: [sc(x['test']) for x in D[c]] for c in CONFIG}
        lin += [f'### {e} · {"episodios fijos" if prot == "fijo" else "rotación"}', '',
                '| P | P (pesos/conex./nodo) | n | score ± sd | mediana | Δ vs Stanley | p (Holm) | val. gen 250 → 500 | conexiones | horas |',
                '|---|---|---|---|---|---|---|---|---|---|']
        base = S['stanley']
        otros = [c for c in CONFIG if c != 'stanley' and len(S[c]) >= 3 and len(base) >= 3]
        praw = [mannwhitneyu(S[c], base, alternative='two-sided').pvalue for c in otros]
        padj = dict(zip(otros, holm(praw)))
        fila = {}
        for c in CONFIG:
            v = S[c]
            if not v:
                lin.append(f'| {c} | {P[f"{e}_{prot}_{c}"]} | 0 | — | — | — | — | — | — | — |')
                continue
            m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            v25 = st.mean(sc(x['v250']) for x in D[c]); v50 = st.mean(sc(x['v500']) for x in D[c])
            con = st.mean(x['conns'] for x in D[c]); h = st.mean(x['horas'] for x in D[c])
            d = m - st.mean(base) if base else float('nan')
            p = padj.get(c)
            lin.append(f'| {c} | {P[f"{e}_{prot}_{c}"]} | {len(v)} | {f2(m)} ± {f2(sd)} | {f2(st.median(v))} | '
                       f'{"" if c == "stanley" else f2(d)} | {"" if p is None else f"{p:.3f}"} | {f2(v25)} → {f2(v50)} | '
                       f'{con:.0f} | {h:.1f} |')
            fila[c] = dict(n=len(v), score=m, sd=sd, mediana=st.median(v), delta=d, p_holm=p, val250=v25, val500=v50,
                           conns=con, horas=h, scores=v)
        # Preguntas
        g = [S[c] for c in FIJOS + CAMBIOS if len(S[c]) >= 3]
        kw = kruskal(*g).pvalue if len(g) >= 2 else float('nan')
        fij = [c for c in FIJOS if len(S[c]) >= 3]
        cam = [c for c in CAMBIOS if len(S[c]) >= 3]
        mejor_f = max(fij, key=lambda c: st.mean(S[c])) if fij else None
        mejor_c = max(cam, key=lambda c: st.mean(S[c])) if cam else None
        pc = (mannwhitneyu(S[mejor_c], S[mejor_f], alternative='two-sided').pvalue if mejor_f and mejor_c else float('nan'))
        lin += ['', f'- **¿Importa P?** Kruskal–Wallis entre {len(g)} P sin ablación: p = {kw:.2g}.',
                f'- **¿Cambiar a mitad de camino supera al mejor P fijo?** mejor fijo = {mejor_f} '
                f'({f2(st.mean(S[mejor_f])) if mejor_f else "—"}), mejor cambio = {mejor_c} '
                f'({f2(st.mean(S[mejor_c])) if mejor_c else "—"}); Mann–Whitney p = {pc:.2g}.',
                '- **Ablaciones vs Stanley:** ' + '; '.join(
                    f'{c} Δ {f2(fila[c]["delta"])} (p Holm {fila[c]["p_holm"]:.3f})' for c in ABL
                    if c in fila and fila[c]['p_holm'] is not None), '']
        salida[e][prot] = dict(filas=fila, kruskal_p=kw, mejor_fijo=mejor_f, mejor_cambio=mejor_c, p_cambio=pc)

texto = '\n'.join(lin)
print(texto)
open(f'{R}/analisis_sonda_p.md', 'w').write(texto + '\n')
json.dump(salida, open(f'{R}/analisis_sonda_p.json', 'w'), indent=1, ensure_ascii=False)
