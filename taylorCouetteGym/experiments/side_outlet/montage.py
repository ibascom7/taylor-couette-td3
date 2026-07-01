"""Assemble a 4-up montage (start / breakthrough / mid / steady) from the
ParaView frames produced by visualize.py.

Run:  python montage.py FRAMES_DIR OUT_PNG [t1 t2 t3 t4]
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frames")
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else frames_dir.parent / "montage.png"

idx = frames_dir / "frames_index.txt"
pairs = []
for line in idx.read_text().splitlines():
    t, fp = line.split(maxsplit=1)
    pairs.append((float(t), fp))
pairs.sort()
times = [t for t, _ in pairs]

# target times: start, ~breakthrough, mid, steady (overridable on cmd line)
if len(sys.argv) > 6:
    targets = [float(x) for x in sys.argv[3:7]]
else:
    tmax = times[-1]
    targets = [times[0], 4.0, tmax * 0.5, tmax]


def nearest(tt):
    return min(pairs, key=lambda p: abs(p[0] - tt))


fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
labels = ["start", "early burst", "mid-run", "steady (end)"]
for ax, tt, lab in zip(axes, targets, labels):
    t, fp = nearest(tt)
    ax.imshow(mpimg.imread(fp))
    ax.set_title(f"{lab}\nt = {t:.0f} s", fontsize=11)
    ax.axis("off")
fig.suptitle("Reactant concentration c on the r–z wedge face "
             "(top inlet c0=50 → bottom side-outlet; outer wall catalytic c=0)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out_png, dpi=130)
print("wrote", out_png)
