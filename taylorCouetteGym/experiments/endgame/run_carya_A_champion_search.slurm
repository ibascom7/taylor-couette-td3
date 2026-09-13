#!/bin/bash
# =====================================================================
# Carya: ENDGAME A -- fair static champion search (GP/EI over (T+, T-)).
#
# Asynchronous Bayesian optimization over the duty_v3 waveform family
# (burst T+ in [1,5] s, idle T- in [0,5] s, trough 0, fixed mean 300 rpm)
# on the WARMED short reactor, 130 s episodes of 5 x 26 s tau-blocks,
# objective = last-block reward r5 = X_[104,130] - P/31.94 (the duty_v3
# gate metric). 44 threaded worker cases on one node; the GP refits on
# every finished episode and proposes the next point (Kriging believer
# for the in-flight ones), so no worker idles. Budget: 44 design + 170 GP
# + replicates of the top 3 (x3, +-0.5 rpm jitter) = ~220 episodes.
#
# COST: duty_v3 ran ~7 episodes per worker per 24 h on Carya cores
# (~3.4 h per 130 s episode) -> ~220/44 x 3.4 h ~ 17-20 h. -t 30 h.
# The run is RESUMABLE: evaluations.csv is append-only and --resume
# reloads it, so a walltime kill loses only the in-flight episodes:
#     sbatch experiments/endgame/run_carya_A_champion_search.slurm
#
# Outputs: experiments/endgame/results/A/ (evaluations.csv, episodes/,
# A_surface.png, A_slices.png, A_idle_collapse.png, A_convergence.png,
# top_table.csv, replicates.csv, summary.json). Replot anywhere with
#     python3 experiments/endgame/a_champion_search.py --analyze-only
#
# Submit FROM taylorCouetteGym/ (paths are $SLURM_SUBMIT_DIR-relative):
#     cd /project/.../taylor-couette-td3/taylorCouetteGym
#     sbatch experiments/endgame/run_carya_A_champion_search.slurm
# Monitor: tail -f experiments/endgame/logs/endgame_A_*.out
# =====================================================================

#SBATCH -J endgame_A
#SBATCH -o experiments/endgame/logs/%x_%j.out
#SBATCH -e experiments/endgame/logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # 44 workers + main thread
#SBATCH --mem=8G
#SBATCH -t 30:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

N_WORKERS=44
SEED=0

OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/endgame/logs
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
WORKER_ROOT="${TMPDIR:-/tmp/$USER}/endgame_A_workers"

echo "=== ENDGAME A: champion search, workers=$N_WORKERS host=$(hostname) cores=${SLURM_CPUS_PER_TASK:-?} ==="
"$PYTHON" experiments/endgame/a_champion_search.py \
    --n_workers "$N_WORKERS" --seed "$SEED" \
    --worker_root "$WORKER_ROOT" \
    --results_dir experiments/endgame/results/A \
    --n_init 44 --n_bo 170 --replicate_top 3 --n_rep 3 \
    --resume
echo "=== A done -> experiments/endgame/results/A ==="
