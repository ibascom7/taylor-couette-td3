#!/bin/bash
# =====================================================================
# Carya: fine-mesh CONVERGENCE STUDY of the resolved-gradient side-outlet wall.
#
# Runs the SAME fine_mesh_sweep.py used locally, as a JOB ARRAY with one task per
# uniform mesh resolution (NR = 16 / 32 / 64 / 128). Each task sweeps its mesh over
# {0,250,500,750,1000} rpm x 150 s, 5 serial pimpleFoam episodes running concurrently
# (1 core each). Splitting by mesh means the cheap meshes finish in ~an hour and the
# expensive NR=128 mesh gets its OWN long-walltime job instead of risking the whole run.
#
# NOTE ON SPEED: the wall-clock of each task is set by its slowest SINGLE episode
# (1000 rpm), i.e. single-core speed -- more cores do not help. Carya cores are usually
# a bit slower per-core than a desktop 7800X3D, so expect these to run AS SLOW OR SLOWER
# than locally; the win of running here is freeing your own machine, not speed.
#
# Est. per-task wall-clock (scales with per-core speed): NR16 ~<1 h, NR32 ~2-4 h,
# NR64 ~8-12 h, NR128 ~1.5-2 days. VERIFY your partition's max walltime (sinfo /
# cluster docs); if it is < the NR128 task, submit --array=0-2 (drop NR=128) or use a
# long partition.
#
# PREREQ (two-checkout sync): commit + push the new scripts and the
#   taylor_couette_mixing/cases/side_outlet_case_r{16,32,64,128}
# case dirs from your /home clone, then `git pull` in the /project clone before sbatch.
#
# Submit (from the taylorCouetteGym/ dir):
#   mkdir -p experiments/steady_state_conversion/logs
#   sbatch experiments/steady_state_conversion/run_carya_fine_mesh.slurm
# Monitor:
#   squeue -u $USER ; tail -f experiments/steady_state_conversion/logs/*.out
# After ALL tasks finish, regenerate the combined table + convergence plot:
#   <env-python> experiments/steady_state_conversion/fine_mesh_sweep.py --analyze-only
# =====================================================================

#SBATCH -J tc_fine_conv
#SBATCH -o experiments/steady_state_conversion/logs/%x_%A_%a.out
#SBATCH -e experiments/steady_state_conversion/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6                      # 5 episodes (one mesh x 5 speeds) + headroom; 1 core each
#SBATCH --mem=8G                  # ~5 fine solvers; 15488-cell NR=128 is the biggest. Plenty of margin.
#SBATCH -t 48:00:00              # sized for the NR=128 task (the long pole). Short tasks finish early
                                  #   and release. If your partition caps below this, see NOTE above.
#SBATCH --array=0-3              # one task per resolution: 0=NR16 1=NR32 2=NR64 3=NR128.
                                  #   Use --array=0-2 for the overnight 3-point study (drops NR=128).
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -uo pipefail

# array index -> radial cell count / case dir
NRS=(16 32 64 128)
NR=${NRS[${SLURM_ARRAY_TASK_ID:-0}]}
CASE="side_outlet_case_r${NR}"

# ---- OpenFOAM + python env (hard-won recipe; see the carya-environment memory) ------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME        # OpenFOAM's SciPy-bundle leaks EasyBuild py3.13 -> breaks conda numpy
PYTHON=/project/mwang/ibascom/envs/tc/bin/python   # absolute path: EasyBuild python shadows conda on PATH
export OMP_NUM_THREADS=1           # each pimpleFoam is serial; don't let libs grab extra cores
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"             # = taylorCouetteGym/
EXP_DIR="$SLURM_SUBMIT_DIR/experiments/steady_state_conversion"
CASE_DIR="$SLURM_SUBMIT_DIR/taylor_couette_mixing/cases/$CASE"
LOGDIR="$EXP_DIR/logs"
mkdir -p "$LOGDIR"

echo "=== fine-mesh convergence task ${SLURM_ARRAY_TASK_ID:-0}: $CASE (NR=$NR) ==="
echo "    host=$(hostname)  cores=${SLURM_CPUS_PER_TASK:-?}  submit_dir=$SLURM_SUBMIT_DIR"

# Build this task's mesh on the node (blockMeshDict is committed; polyMesh may not be;
# blockMesh is deterministic so re-running is harmless if it is committed). 0/ is committed.
( cd "$CASE_DIR" && blockMesh > "$LOGDIR/blockMesh_${CASE}.log" 2>&1 )

# Run only THIS mesh's sweep. fine_mesh_sweep.py resolves its own paths relative to the
# script, writes to results_fine/, and its final analyze() rebuilds the table +
# convergence plot from ALL logs on disk -> partial results appear as tasks finish.
cd "$EXP_DIR"
"$PYTHON" fine_mesh_sweep.py --cases "$CASE" --workers 5

echo "=== done $CASE. For the FINAL combined figures once every task is finished, run:"
echo "    $PYTHON $EXP_DIR/fine_mesh_sweep.py --analyze-only"
