#!/usr/bin/env python3
"""Estudio de hiperparámetros de NEAT estático con Optuna (docs/diseno_neat_ap.md §17).

Un estudio por entorno (bottleneck, acrobot-swingup, l2f). Cada trial corre `neat_control` con protocolo de
rotación (--resample-train) en las semillas de AJUSTE (101–103 por defecto), nunca en las de evaluación (1–10):
la mejor configuración se evalúa después en las semillas 1–10, así que su resultado no queda optimista.

- Presupuesto fijo: población × generaciones = 75 000 evaluaciones de genoma (150 × 500). Si Optuna cambia la
  población, las generaciones se ajustan para mantener el presupuesto.
- Objetivo: score medio en el test final (100 episodios, stream 2) de las semillas de ajuste,
  score = (test − trivial) / (RL − trivial) con ~/tesis/runs/ref_rl/score_v2_s1-10.json.
- Poda: cada minuto se lee generations.csv; en cada 10 % del presupuesto se informa a Optuna la media, entre
  semillas, del mejor score de validación hasta ahí (lo que elige --resample-train). MedianPruner corta los trials
  que van por debajo de la mediana de los anteriores en el mismo punto del presupuesto.
- El primer trial de cada estudio es la configuración actual (valores por defecto + P de Stanley), como referencia.
- Varios procesos trabajadores comparten el estudio a través de un JournalStorage en archivo; MaxTrialsCallback
  limita el total de trials (completos + podados).

Uso (Optuna está en el entorno conda tdi):
  ~/miniconda3/envs/tdi/bin/python scripts/optuna_neat.py --env acrobot-swingup --trials 100
  (lanzarlo con scripts/lote_optuna_neat.sh, que arranca los trabajadores en tmux con candado y binario congelado)
"""
import argparse, csv, json, os, signal, subprocess, time
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

R = os.path.expanduser('~/tesis/runs')
EVALS = 75000
DEFECTO = dict(pop=150, target_species=10, compat_weight_scale=10.0, survival=0.2, weight_power=5.0,
               weight_replace_prob=0.02, add_node_weight=30.0, weight_init_range=30.0, weight_max_factor=2.0,
               mutation_only=0.25, species_stagnation=15, input_gain=40.0, spikes_full=5.0, step_ms=50,
               p_w=0.8, p_c=0.05, p_n=0.03)


def espacio(t):
    return dict(
        pop=t.suggest_int('pop', 50, 300, step=10),
        target_species=t.suggest_int('target_species', 3, 25),
        compat_weight_scale=t.suggest_float('compat_weight_scale', 2.0, 30.0, log=True),
        survival=t.suggest_float('survival', 0.1, 0.5),
        weight_power=t.suggest_float('weight_power', 0.5, 15.0, log=True),
        weight_replace_prob=t.suggest_float('weight_replace_prob', 0.0, 0.2),
        add_node_weight=t.suggest_float('add_node_weight', 15.0, 50.0),
        weight_init_range=t.suggest_float('weight_init_range', 10.0, 50.0),
        weight_max_factor=t.suggest_float('weight_max_factor', 1.0, 3.0),   # peso máximo = factor × rango inicial
        mutation_only=t.suggest_float('mutation_only', 0.1, 0.6),
        species_stagnation=t.suggest_int('species_stagnation', 10, 40),
        input_gain=t.suggest_float('input_gain', 10.0, 100.0, log=True),
        spikes_full=t.suggest_float('spikes_full', 2.0, 15.0),
        step_ms=t.suggest_int('step_ms', 20, 100, step=10),
        p_w=t.suggest_float('p_w', 0.05, 1.0),
        p_c=t.suggest_float('p_c', 0.005, 0.5, log=True),
        p_n=t.suggest_float('p_n', 0.005, 0.5, log=True),
    )


def argumentos(env, h, seed, out, evals=EVALS):
    gens = max(2, round(evals / h['pop']))
    return ['--env', env, '--selector', 'static', '--resample-train', '--seed', str(seed),
            '--episodes', '10', '--test-episodes', '10', '--final-episodes', '100', '--print-every', '1000',
            '--pop', str(h['pop']), '--generations', str(gens), '--target-species', str(h['target_species']),
            '--compat-weight-scale', f"{h['compat_weight_scale']:.6g}", '--survival', f"{h['survival']:.6g}",
            '--weight-power', f"{h['weight_power']:.6g}", '--weight-replace-prob', f"{h['weight_replace_prob']:.6g}",
            '--add-node-weight', f"{h['add_node_weight']:.6g}", '--weight-init-range', f"{h['weight_init_range']:.6g}",
            '--weight-max', f"{h['weight_init_range'] * h['weight_max_factor']:.6g}",
            '--mutation-only', f"{h['mutation_only']:.6g}", '--species-stagnation', str(h['species_stagnation']),
            '--input-gain', f"{h['input_gain']:.6g}", '--spikes-full', f"{h['spikes_full']:.6g}",
            '--step-ms', str(h['step_ms']), '--p', f"{h['p_w']:.6g},{h['p_c']:.6g},{h['p_n']:.6g}",
            '--out', out], gens


def leer_validacion(f):
    try:
        return [float(r['champion_test']) for r in csv.DictReader(open(f))]
    except (FileNotFoundError, KeyError, ValueError):
        return []


def objetivo_factory(a, ref):
    sc = lambda x: (x - ref['trivial']) / (ref['rl_mean'] - ref['trivial'])

    def objetivo(trial):
        h = espacio(trial)
        base = f'{R}/{a.dir}/{a.env}/trial_{trial.number:04d}'
        procs, gens = [], None
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
        for s in a.seeds:
            out = f'{base}/seed_{s}'
            os.makedirs(out, exist_ok=True)
            args, gens = argumentos(a.env, h, s, out, a.evals)
            log = open(f'{out}/log.txt', 'w')
            procs.append((subprocess.Popen(['nice', '-n', '10', a.binary] + args, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True), out))
        trial.set_user_attr('generaciones', gens)
        informado = 0
        try:
            while True:
                vivos = [p for p, _ in procs if p.poll() is None]
                curvas = [leer_validacion(f'{o}/generations.csv') for _, o in procs]
                hecho = min(len(c) for c in curvas) / gens
                while informado + 10 <= int(hecho * 100) and informado < 90:
                    informado += 10
                    k = max(1, int(informado / 100 * gens))
                    v = sum(sc(max(c[:k])) for c in curvas) / len(curvas)
                    trial.report(v, informado)
                    if trial.should_prune():
                        raise optuna.TrialPruned(f'podado al {informado} % del presupuesto (validación {v:.3f})')
                if not vivos:
                    break
                time.sleep(a.intervalo)
            res = []
            for p, o in procs:
                if p.returncode != 0 or not os.path.exists(f'{o}/summary.txt'):
                    raise RuntimeError(f'neat_control falló en {o} (código {p.returncode})')
                kv = dict(l.strip().split('=', 1) for l in open(f'{o}/summary.txt') if '=' in l)
                res.append(dict(score=sc(float(kv['final_test'])), conns=int(kv['best_conns']),
                                horas=float(kv['seconds']) / 3600))
            trial.set_user_attr('scores', [r['score'] for r in res])
            trial.set_user_attr('conexiones', [r['conns'] for r in res])
            trial.set_user_attr('horas', max(r['horas'] for r in res))
            return sum(r['score'] for r in res) / len(res)
        finally:
            for p, _ in procs:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            for p, _ in procs:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)

    return objetivo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--trials', type=int, default=100, help='total de trials del estudio (completos + podados)')
    ap.add_argument('--seeds', default='101,102,103')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--binary', default=f'{R}/optuna_neat_bin/neat_control')
    ap.add_argument('--evals', type=int, default=EVALS, help='presupuesto (solo para pruebas; el estudio usa 75 000)')
    ap.add_argument('--dir', default='optuna_neat', help='carpeta en ~/tesis/runs')
    ap.add_argument('--max-fallos', type=int, default=5, help='trials fallidos tras los cuales se detiene el estudio')
    ap.add_argument('--intervalo', type=float, default=60, help='segundos entre lecturas de generations.csv')
    a = ap.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(',')]
    ref = {r['variante']: r for r in json.load(open(f'{R}/ref_rl/score_v2_s1-10.json'))}[a.env]
    os.makedirs(f'{R}/{a.dir}/{a.env}', exist_ok=True)
    storage = JournalStorage(JournalFileBackend(f'{R}/{a.dir}/{a.env}/estudio.log'))
    study = optuna.create_study(
        study_name=f'neat_{a.env}', storage=storage, direction='maximize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=10))
    if not study.trials:
        study.enqueue_trial(DEFECTO)
    def demasiados_fallos(study, trial):
        # Un fallo aislado no detiene al trabajador (catch), pero si se acumulan, el estudio se corta: un error
        # sistemático no debe quemar trials en silencio.
        if sum(t.state == TrialState.FAIL for t in study.trials) >= a.max_fallos:
            print(f'Estudio detenido: {a.max_fallos} trials fallidos. Revisar los log.txt de los trials.', flush=True)
            study.stop()

    study.optimize(objetivo_factory(a, ref), catch=(Exception,),
                   callbacks=[optuna.study.MaxTrialsCallback(a.trials, states=(TrialState.COMPLETE, TrialState.PRUNED)),
                              demasiados_fallos])


if __name__ == '__main__':
    main()
