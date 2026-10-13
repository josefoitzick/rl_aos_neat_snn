#!/usr/bin/env python3
"""Genera una copia de src/rl/zoo/zoo.cpp de RLtools con un gancho para configuraciones propias
(docs/diseno_neat_ap.md §15.4, paso 2: entornos sin entrenador en el zoo).

Cambios (los únicos; se verifica que cada marcador aparezca exactamente una vez):
  1. La cadena que elige LOOP_CORE_CONFIG según las macros del zoo queda envuelta en
     `#ifndef TESIS_REF_CONFIG`; con la macro definida se incluye ese archivo (segunda etapa de la
     configuración propia, ver src/ref_rl/README.md).
  2. La cadena que fija el nombre del entorno queda envuelta en `#ifndef TESIS_REF_ENV_NAME`.
El resto del entrenador (evaluaciones, checkpoints, exportación del actor, curriculum de l2f con SAC si se
define RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F) queda igual.

Uso: generar_zoo_tesis.py <rl-tools>/src/rl/zoo/zoo.cpp <salida.cpp>
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split('\n')


def unico(pred, desc, desde=0):
    idx = [i for i, l in enumerate(lines) if i >= desde and pred(l)]
    if len(idx) != 1:
        sys.exit(f'generar_zoo_tesis: se esperaba 1 "{desc}", hay {len(idx)}')
    return idx[0]


# 1. Cadena de LOOP_CORE_CONFIG: desde el primer "#if defined(RL_TOOLS_RL_ZOO_ALGORITHM_SAC)" que precede a
#    "using LOOP_CORE_CONFIG" hasta el "#endif" que sigue a '#error "RLtools Zoo: Algorithm not defined"' (el primero).
err_alg = [i for i, l in enumerate(lines) if '#error "RLtools Zoo: Algorithm not defined"' in l]
if len(err_alg) != 2:
    sys.exit(f'generar_zoo_tesis: se esperaban 2 "Algorithm not defined", hay {len(err_alg)}')
fin1 = err_alg[0] + 1
assert lines[fin1].strip() == '#endif', lines[fin1]
primer_using = unico(lambda l: l.startswith('using LOOP_CORE_CONFIG = rlt::rl::zoo::pendulum_v1::sac::'), 'using pendulum_v1::sac')
ini1 = primer_using - 2
assert lines[ini1].strip() == '#if defined(RL_TOOLS_RL_ZOO_ALGORITHM_SAC)', lines[ini1]

# 2. Cadena del nombre del entorno
ini2 = unico(lambda l: l.strip() == 'std::string environment = "pendulum-v1";', 'environment = pendulum-v1') - 1
assert lines[ini2].strip() == '#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)', lines[ini2]
err_env = unico(lambda l: '#error "RLtools Zoo: Environment not defined"' in l, 'Environment not defined')
fin2 = err_env + 1
assert lines[fin2].strip() == '#endif', lines[fin2]
assert ini1 < fin1 < ini2 < fin2

out = (['// GENERADO por scripts/ref_rl/generar_zoo_tesis.py a partir de ' + src + ' (no editar)']
       + lines[:ini1]
       + ['#ifdef TESIS_REF_CONFIG', '#define TESIS_REF_STAGE2', '#include TESIS_REF_CONFIG', '#else']
       + lines[ini1:fin1 + 1]
       + ['#endif  // TESIS_REF_CONFIG']
       + lines[fin1 + 1:ini2]
       + ['#ifdef TESIS_REF_ENV_NAME', 'std::string environment = TESIS_REF_ENV_NAME;', '#else']
       + lines[ini2:fin2 + 1]
       + ['#endif  // TESIS_REF_ENV_NAME']
       + lines[fin2 + 1:])
open(dst, 'w').write('\n'.join(out))
print(f'ok: cadena de configuración {ini1 + 1}-{fin1 + 1}, nombre {ini2 + 1}-{fin2 + 1} → {dst}')
