#!/bin/bash
# Launch B (3 runs), C (2 runs) and E (6 runs) on the LOCAL machine in the
# background -- 11 single-core pimpleFoam episodes, ~1-2.5 days each.
# Leaves 5 cores free on the 16-core box. Run A on Carya (needs 44 workers).
#
#   cd taylorCouetteGym && bash experiments/endgame/launch_local.sh
#   tail -f experiments/endgame/logs/{B,C_stretch,C_wrap,E}_local.log
set -euo pipefail
cd "$(dirname "$0")/../.."          # taylorCouetteGym/
source /usr/lib/openfoam/openfoam2506/etc/bashrc
PYTHON=/home/ibascom/research/taylor-couette/.venv/bin/python
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p experiments/endgame/logs
E=experiments/endgame

nohup "$PYTHON" -u $E/gamma30_statics.py --group B --workers 3 --resume > $E/logs/B_local.log 2>&1 &
echo "B  (3 cores) pid $!"
nohup "$PYTHON" -u $E/c_continue_transfer.py --source s2_stretch --clock hold --resume > $E/logs/C_stretch_local.log 2>&1 &
echo "C0 (1 core)  pid $!"
nohup "$PYTHON" -u $E/c_continue_transfer.py --source s2_wrap --clock wrap --resume > $E/logs/C_wrap_local.log 2>&1 &
echo "C1 (1 core)  pid $!"
nohup "$PYTHON" -u $E/gamma30_statics.py --group E --workers 6 --resume > $E/logs/E_local.log 2>&1 &
echo "E  (6 cores) pid $!"
