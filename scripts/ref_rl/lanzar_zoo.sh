#!/usr/bin/env bash
# Entrena las referencias RL del zoo de RLtools (docs/diseno_neat_ap.md §15.4, paso 2):
# cada binario de ~/tesis/runs/ref_rl/bin (scripts/ref_rl/compilar_zoo.sh) × semillas 0..4,
# un proceso de un hilo por corrida, en ~/tesis/runs/ref_rl/zoo/<binario>/seed_<k>.
#
# Protecciones (como los barridos de NEAT):
#   - candado ~/tesis/runs/ref_rl/zoo.lock (no se lanza dos veces);
#   - marca .running atómica por corrida (O_EXCL vía mkdir) y "FIN" con el código de salida;
#   - reanudable: se salta lo que ya tiene return.json.set (corrida completa);
#   - sha256 de los binarios en el log.
#
# Uso: tmux new -d -s ref_rl 'bash scripts/ref_rl/lanzar_zoo.sh [JOBS]'
set -u
BIN=~/tesis/runs/ref_rl/bin
OUT=~/tesis/runs/ref_rl/zoo
JOBS=${1:-80}
mkdir -p "$OUT"
exec 8> ~/tesis/runs/ref_rl/zoo.lock
if ! flock -n 8; then echo "Ya hay un lanzador de ref_rl en curso."; exit 1; fi

{
  echo "inicio: $(date)  $(cat "$BIN/VERSION")  jobs: $JOBS"
  cat "$BIN/SHA256SUMS"
} >> "$OUT/lanzador.log"

correr() {
  local bin=$1 seed=$2
  local nombre=${bin#zoo_}
  local dir=$OUT/$nombre/seed_$seed
  mkdir -p "$dir"
  if compgen -G "$dir/experiments/*/*/*/*/return.json.set" > /dev/null; then return 0; fi
  mkdir "$dir/.running" 2>/dev/null || { echo "OMITIDA $nombre seed $seed (otra en curso)"; return 0; }
  rm -rf "$dir/experiments"
  local t0=$SECONDS
  nice -n 10 "$BIN/$bin" -s "$seed" -e "$dir/experiments" > "$dir/log.txt" 2>&1
  local rc=$?
  echo "$rc $((SECONDS - t0))" > "$dir/FIN"
  rmdir "$dir/.running"
  if [[ $rc -eq 0 ]]; then echo "ok $nombre seed $seed ($((SECONDS - t0)) s)"; else echo "ERROR $nombre seed $seed (código $rc)"; fi
}
export -f correr
export BIN OUT

for b in $(cd "$BIN" && ls zoo_* | grep -v '\.log$'); do
  for s in 0 1 2 3 4; do echo "$b $s"; done
done | xargs -P "$JOBS" -L 1 bash -c 'correr $0 $1' >> "$OUT/progreso.log" 2>&1

echo "fin: $(date)" | tee -a "$OUT/lanzador.log" > "$OUT/FIN"
