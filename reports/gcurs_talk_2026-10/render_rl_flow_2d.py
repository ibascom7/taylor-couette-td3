"""ParaView render of the 2D (axisymmetric r-z) simulation domain with the flow
under the learned RL policy (duty_v3 seed 2, short reactor, warmed evaluation,
mid-burst at t = 113 s): reactant concentration c/c0 (default) or radial
velocity as colour, plus in-plane streamlines.

Produces a tall, narrow image of just the fluid gap (inner wall on the left,
catalytic outer wall on the right, true 6.35 x 38.1 mm aspect). The slide
draws the axis, inner cylinder, inlet and side-outlet marks around it.

    pvbatch reports/gcurs_talk_2026-10/render_rl_flow_2d.py <case_dir> <time> <out.png> [c|ur] [lines|nolines]
Concentration uses blue (low: reactant used up, i.e. product) to red (high: fresh reactant).
"""
import os
import sys

from paraview.simple import (OpenFOAMReader, MergeBlocks, CellDatatoPointData, Calculator, Slice,
                             StreamTracer, Transform, Show, GetActiveViewOrCreate, ColorBy,
                             GetColorTransferFunction, SaveScreenshot)

HERE = os.path.dirname(os.path.abspath(__file__))
case = sys.argv[1]
time = float(sys.argv[2]) if len(sys.argv) > 2 else 113.0
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "figures", "rl_flow_2d.png")
field = sys.argv[4] if len(sys.argv) > 4 else "c"          # "c" (concentration) or "ur"
show_lines = (sys.argv[5] if len(sys.argv) > 5 else "lines") != "nolines"
RI, RO, H = 0.0254, 0.03175, 0.0381

foam = os.path.join(case, "case.foam")
if not os.path.exists(foam):
    open(foam, "w").close()
reader = OpenFOAMReader(FileName=foam)
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = ["U", "c"]
reader.UpdatePipeline(time)
pts = CellDatatoPointData(Input=MergeBlocks(Input=reader))
ur = Calculator(Input=pts)
ur.ResultArrayName = "ur"
ur.Function = "(U_X*coordsX+U_Y*coordsY)/sqrt(coordsX^2+coordsY^2)*1000"          # mm/s
cn = Calculator(Input=ur)
cn.ResultArrayName = "cn"
cn.Function = "c/50"                                                                 # c / c0
uplane = Calculator(Input=cn)
uplane.ResultArrayName = "Uplane"                                                  # meridional velocity
uplane.Function = "(U_X*coordsX+U_Y*coordsY)/sqrt(coordsX^2+coordsY^2)*iHat + U_Z*kHat"

mid = Slice(Input=uplane)                     # the r-z mid-plane of the wedge (y = 0)
mid.SliceType = "Plane"
mid.SliceType.Origin = [0.5 * (RI + RO), 0.0, 0.0]
mid.SliceType.Normal = [0.0, 1.0, 0.0]

lines = StreamTracer(Input=mid, SeedType="Line")
lines.Vectors = ["POINTS", "Uplane"]
lines.SeedType.Point1 = [RI + 0.0006, 0.0, -0.5 * H + 0.0004]
lines.SeedType.Point2 = [RO - 0.0006, 0.0, 0.5 * H - 0.0004]
lines.SeedType.Resolution = 28
lines.MaximumStreamlineLength = 0.012   # about one loop per line
lines.IntegrationDirection = "BOTH"
front = Transform(Input=lines)                # lift the lines just in front of the coloured plane
front.Transform = "Transform"
front.Transform.Translate = [0.0, -0.0002, 0.0]

view = GetActiveViewOrCreate("RenderView")
view.ViewTime = time
view.ViewSize = [420, 2400]
view.Background = [0.984, 0.984, 0.973]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.OrientationAxesVisibility = 0

d = Show(mid, view)
if field == "c":
    ColorBy(d, ("POINTS", "cn"))
    lut = GetColorTransferFunction("cn")
    lut.ApplyPreset("Cool to Warm", True)     # blue = product (used up), red = reactant
    lut.RescaleTransferFunction(0.45, 0.88)   # bulk spans ~0.65-0.86; the wall layer saturates dark
else:
    ColorBy(d, ("POINTS", "ur"))
    lut = GetColorTransferFunction("ur")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(-150.0, 150.0)
d.SetScalarBarVisibility(view, False)

ld = Show(front, view)
ld.Visibility = 1 if show_lines else 0
ld.ColorArrayName = ["POINTS", ""]
line_rgb = [0.20, 0.19, 0.19]
ld.DiffuseColor = line_rgb
ld.AmbientColor = line_rgb
ld.LineWidth = 1.2

view.CameraParallelProjection = 1
view.CameraFocalPoint = [0.5 * (RI + RO), 0.0, 0.0]
view.CameraPosition = [0.5 * (RI + RO), -0.2, 0.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelScale = 0.5 * H            # fit the height exactly

os.makedirs(os.path.dirname(out), exist_ok=True)
SaveScreenshot(out, view, ImageResolution=[420, 2400], TransparentBackground=0)
print("wrote", out)
