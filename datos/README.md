# Datos numéricos de los experimentos

Resultados numéricos de los experimentos de la tesis (NEAT + SNN con selección adaptativa de operadores). Cada carpeta corresponde
a un experimento descrito en las fichas `docs/experimentos/` del repositorio del código. Contenido:

- `summary.txt` y `generations.csv` de cada corrida (resultado final y curva por generación);
- `decisiones.csv`, `corridas.csv` y `ppo.csv` de los entrenamientos y evaluaciones del agente;
- `analisis*resultado*.txt` y `resultado*.txt`: salidas de los análisis con reglas fijadas antes de mirar;
- `score*.json`: referencias (trivial y RL) para el score normalizado;
- `estado*.tar` / `estado*.txt`: puntos de control de los agentes PPO, para retomar el entrenamiento.

Notas:
- Una corrida está completa si su carpeta tiene `summary.txt` (o `FIN` en el planificador original). Las corridas detenidas al
  cerrar una ventana de cómputo dejan `generations.csv` parcial y sin `summary.txt`; los análisis las excluyen.
- Las rutas a archivos del clúster se reemplazaron por `~` en la copia.
