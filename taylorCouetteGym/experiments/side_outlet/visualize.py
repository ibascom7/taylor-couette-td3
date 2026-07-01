"""ParaView (pvbatch) visualization of Yuhe's side-outlet case.

Renders the concentration field c on the wedge's r-z front face for every
written time directory, so you can watch the dye enter at the top inlet and
advect down to the bottom side-outlet. Saves one PNG per frame plus a 4-up
montage (start / breakthrough / mid / steady) for slides.

Run:  pvbatch visualize.py CASE_DIR OUT_DIR
(CASE_DIR must contain a <name>.foam file; snapshot_frames() / `touch case.foam`
creates one.)
"""
import sys
import os
import glob
from paraview.simple import *  # noqa: F401,F403

case_dir = sys.argv[1] if len(sys.argv) > 1 else "."
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(case_dir, "frames")
os.makedirs(out_dir, exist_ok=True)

foam = glob.glob(os.path.join(case_dir, "*.foam"))
if not foam:
    # create one so the reader opens the case in place
    foam_path = os.path.join(case_dir, "case.foam")
    open(foam_path, "a").close()
    foam = [foam_path]
foam_path = foam[0]

print("Opening", foam_path)
reader = OpenFOAMReader(FileName=foam_path)
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = ["c", "U", "p"]
reader.UpdatePipeline()

# available time steps
tk = reader.TimestepValues
times = list(tk) if tk is not None else [0.0]
print("times:", times[0], "...", times[-1], "n=", len(times))

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [900, 760]
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]
view.CameraParallelProjection = 1

disp = Show(reader, view)
ColorBy(disp, ("CELLS", "c"))
disp.SetScalarBarVisibility(view, True)
disp.Representation = "Surface"

# fixed color range 0..50 (inlet c0) so frames are comparable
ctf = GetColorTransferFunction("c")
ctf.RescaleTransferFunction(0.0, 50.0)
ctf.ApplyPreset("Viridis (matplotlib)", True)
bar = GetScalarBar(ctf, view)
bar.Title = "c [mmol/m3]"
bar.ComponentTitle = ""

# camera: look along -y at the x-z (r-z) plane; +z up
cx = 0.5 * (0.0254 + 0.03175)
disp.Representation = "Surface With Edges"
view.CameraFocalPoint = [cx, 0.0, 0.0]
view.CameraPosition = [cx, -0.2, 0.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelScale = 0.022  # half-height of view (~ device half-height + margin)
Render()

frame_paths = []
for i, t in enumerate(times):
    view.ViewTime = t
    reader.UpdatePipeline(t)
    Render()
    fp = os.path.join(out_dir, "frame_%04d_t%07.2f.png" % (i, t))
    SaveScreenshot(fp, view, ImageResolution=[900, 760])
    frame_paths.append((t, fp))
    print("saved", fp)

# write an index so the montage step (matplotlib) can pick frames
with open(os.path.join(out_dir, "frames_index.txt"), "w") as f:
    for t, fp in frame_paths:
        f.write("%g %s\n" % (t, fp))
print("done:", len(frame_paths), "frames ->", out_dir)
