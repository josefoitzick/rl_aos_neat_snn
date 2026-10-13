#!/usr/bin/env python3
"""Métricas de complejidad de los barridos A y B (docs/entornos_rltools.md §7).

Por variante (5 semillas): retorno trivial (máx. de aleatorio y acción cero, de los probe.csv), test final ± sd,
mejor train, z del test respecto del trivial, brecha train→test normalizada, g90 (generación en que el máximo
acumulado de best_train alcanza el 90 % de su mejora), fracción de la mejora en la 2ª mitad, tamaño, horas y,
por ventanas de 50 generaciones, la tasa de éxito de cada operador (W = pesos, C = conexión, N = nodo).
Escribe ~/tesis/runs/analisis_barrido_ab.json.
"""
import os,json,csv,glob,itertools,statistics as st
R=os.path.expanduser('~/tesis/runs')
probe={}
for f in glob.glob(R+'/env_probe_*/probe.csv'):
    for r in csv.DictReader(open(f)): probe[r['env']]=r
OPS=['mutate_weights','add_connection','add_node']
rows=[]
for sweep in ['barrido_etapa_a','barrido_etapa_b']:
    cfg=json.load(open(f'{R}/{sweep}/config.json'))
    for name in cfg['variants']:
        env=name.replace('_memoria','').replace('_reactivo','')
        if env.startswith('acrobot_t'): env='acrobot'
        T=[];TR=[];H=[];C=[];S=[];G90=[];LATE=[]
        win=[[ [0,0] for _ in OPS] for _ in range(10)]
        for s in range(1,6):
            d=f'{R}/{sweep}/{name}/seed_{s:02d}'
            kv=dict(l.strip().split('=',1) for l in open(d+'/summary.txt') if '=' in l)
            T.append(float(kv['final_test']));TR.append(float(kv['best_train']));H.append(int(kv['best_hidden']));C.append(int(kv['best_conns']));S.append(float(kv['seconds'])/3600)
            g=list(csv.DictReader(open(d+'/generations.csv')))
            b=list(itertools.accumulate([float(x['best_train']) for x in g],max)); b0,bN=b[0],b[-1]; gain=bN-b0
            if gain>1e-9:
                G90.append(next(i for i,x in enumerate(b) if x-b0>=0.9*gain))
                LATE.append((bN-b[250])/gain)
            else: G90.append(0); LATE.append(0)
            for i,x in enumerate(g):
                w=min(i//50,9)
                for k,op in enumerate(OPS):
                    n=float(x['n_'+op]); win[w][k][0]+=n*float(x['improved_'+op]); win[w][k][1]+=n
        rate=[[ (a/n if n else 0) for a,n in w] for w in win]
        best=[max(range(3),key=lambda k:r[k]) for r in rate]
        changes=sum(1 for i in range(1,10) if best[i]!=best[i-1])
        # rank of non-weight ops relative to weights: early vs late ratio
        p=probe.get(env,{})
        rnd=float(p.get('random_mean','nan')); zer=float(p.get('zero_mean','nan'))
        if name=='acrobot_t5': rnd=zer=float('nan')
        triv=max(rnd,zer)
        tm=st.mean(T); sd=st.stdev(T); trm=st.mean(TR)
        rows.append(dict(v=name,ops=p.get('obs_dim','?'),act=p.get('act_dim','?'),ag=p.get('agents','?'),
            triv=triv,test=tm,sd=sd,train=trm,
            dtriv=(tm-triv)/sd if sd>0 and triv==triv else float('nan'),
            gap=(trm-tm)/(trm-triv) if triv==triv and trm-triv>0 else float('nan'),
            g90=st.mean(G90),late=st.mean(LATE),H=st.mean(H),C=st.mean(C),h=st.mean(S),
            changes=changes,best=''.join('WCN'[k] for k in best),
            early=rate[0],late_r=rate[9]))
print(f"{'variante':34s} obs act ag  trivial     test±sd       train   z_triv  brecha g90 tardío  ocu  conns  h   cambios mejor_op(10 vent.)  éxito W/C/N ini→fin")
for r in rows:
    print(f"{r['v']:34s} {r['ops']:>4} {r['act']:>2} {r['ag']:>2} {r['triv']:8.4g} {r['test']:8.4g}±{r['sd']:<6.3g} {r['train']:8.4g} {r['dtriv']:6.1f} {r['gap']:6.2f} {r['g90']:4.0f} {r['late']:5.2f} {r['H']:4.1f} {r['C']:6.0f} {r['h']:4.1f} {r['changes']:3d}  {r['best']}  "
          + "/".join(f"{x:.3f}" for x in r['early'])+" → "+"/".join(f"{x:.3f}" for x in r['late_r']))
json.dump(rows,open(R+'/analisis_barrido_ab.json','w'),indent=1)
