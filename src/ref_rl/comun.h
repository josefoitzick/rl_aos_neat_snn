// Configuraciones propias de la referencia RL (docs/diseno_neat_ap.md §15.4, paso 2): entornos nuestros sin
// entrenador en el zoo de RLtools. Cada header de src/ref_rl se usa DOS veces al compilar la copia generada
// del zoo (scripts/ref_rl/generar_zoo_tesis.py, scripts/ref_rl/compilar_propias.sh):
//   1. con -include, antes de todo zoo.cpp (etapa 1): operaciones de entornos que el zoo no incluye y
//      sobrecargas que deben declararse antes que las plantillas de RLtools;
//   2. vía TESIS_REF_CONFIG, en lugar de la cadena de configuraciones del zoo (etapa 2, con
//      TESIS_REF_STAGE2 definida): LOOP_CORE_CONFIG y LOOP_EVALUATION_PARAMETER_OVERWRITES.
// Criterio: hiperparámetros de la configuración del zoo (o de RLtools) más cercana, sin ajustarlos; solo se
// cambia el entorno y, cuando difiere, el horizonte del episodio por el nuestro (rltools_envs.cpp).
#pragma once
#include <rl_tools/operations/cpu_mux.h>
