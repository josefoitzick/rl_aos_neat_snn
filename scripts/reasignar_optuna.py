#!/usr/bin/env python3
"""Reasigna hebras entre los estudios de Optuna de NEAT (docs/diseno_neat_ap.md §17).

Mantiene a lo más MAX trabajadores en total (cada uno = un trial = 3 semillas × 4 hilos = 12 hilos; 25 × 12 = 300).
Cada 5 minutos: si un estudio ya llegó a su número de trials (completos + podados), sus trabajadores terminan solos;
los cupos libres se dan a los estudios que siguen, en orden de prioridad (el más caro primero). No toca trials en
curso: solo lanza trabajadores nuevos, que se suman al mismo estudio. Termina cuando todos los estudios terminan.

Uso: tmux new -d -s optuna_reasignar '~/miniconda3/envs/tdi/bin/python scripts/reasignar_optuna.py'
"""
import os, subprocess, time
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

R = os.path.expanduser('~/tesis/runs')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser('~/miniconda3/envs/tdi/bin/python')
BIN = f'{R}/optuna_neat_bin/neat_control'
PRIORIDAD = ['bottleneck', 'acrobot-swingup', 'l2f']
TRIALS = 100
MAX = 25
LOG = f'{R}/optuna_neat/logs/reasignar.log'
optuna.logging.set_verbosity(optuna.logging.WARNING)


def log(m):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%F %T ') + m + '\n')


def hechos(env):
    s = optuna.load_study(study_name=f'neat_{env}',
                          storage=JournalStorage(JournalFileBackend(f'{R}/optuna_neat/{env}/estudio.log')))
    return sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in s.trials)


def trabajadores(env):
    out = subprocess.run(['pgrep', '-u', str(os.getuid()), '-f', f'optuna_neat.py --env {env} '],
                         capture_output=True, text=True).stdout.split()
    return len(out)


log(f'inicio, MAX={MAX} trabajadores')
extra = 0
while True:
    pend = [e for e in PRIORIDAD if hechos(e) < TRIALS]
    if not pend:
        log('todos los estudios terminaron; fin')
        break
    activos = {e: trabajadores(e) for e in PRIORIDAD}
    libres = MAX - sum(activos.values())
    for e in pend:
        if libres <= 0:
            break
        # no tiene sentido lanzar más trabajadores que trials por hacer
        n = min(libres, TRIALS - hechos(e) - activos[e])
        for _ in range(max(0, n)):
            extra += 1
            subprocess.Popen([PY, 'scripts/optuna_neat.py', '--env', e, '--trials', str(TRIALS), '--binary', BIN],
                             cwd=REPO, stdout=open(f'{R}/optuna_neat/logs/{e}_extra{extra}.log', 'w'),
                             stderr=subprocess.STDOUT, start_new_session=True)
            libres -= 1
            log(f'+1 trabajador en {e} (extra {extra}); activos antes: {activos}')
            time.sleep(5)
    time.sleep(300)
