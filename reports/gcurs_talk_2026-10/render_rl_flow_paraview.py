"""ParaView render of the flow under the learned RL policy (duty_v3 seed 2,
short reactor, deterministic warmed evaluation), mid-burst at t = 113 s.

The CFD is a 5-degree axisymmetric wedge; it is replicated about the axis
into a half annulus (180 degrees) viewed face-on to the cut plane, so the
Taylor cells show as bands on the two cut faces either side of the (half)
inner cylinder. Colour = radial velocity (red: toward the catalytic outer wall).

Run with ParaView's batch interpreter:
    pvbatch reports/gcurs_talk_2026-10/render_rl_flow_paraview.py <case_dir> <time> <out.png>
<case_dir> is an OpenFOAM case (needs constant/, system/ and the time folder
with U); a copy is fine. Default: the committed eval case, t = 113.
"""
import os
import sys

from paraview.simple import (OpenFOAMReader, CellDatatoPointData, Calculator, Transform,
                             AppendDatasets, MergeBlocks, Cylinder, Show, GetActiveViewOrCreate,
                             ColorBy, GetColorTransferFunction, GetScalarBar, SaveScreenshot,
                             UpdatePipeline, ExtractSurface, GetActiveCamera, Clip, Plane, Disk)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CASE = os.path.join(HERE, "..", "..", "taylorCouetteGym", "experiments", "modulation_rl",
                            "results", "warmed_eval", "duty_v3_s2", "case")
case = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE
time = float(sys.argv[2]) if len(sys.argv) > 2 else 113.0
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "figures", "rl_flow_paraview.png")

foam = os.path.join(case, "case.foam")
if not os.path.exists(foam):
    open(foam, "w").close()

reader = OpenFOAMReader(FileName=foam)
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = ["U"]
reader.UpdatePipeline(time)
pts = CellDatatoPointData(Input=MergeBlocks(Input=reader))   # AppendDatasets drops multiblock input
ur = Calculator(Input=pts)
ur.ResultArrayName = "ur"
ur.Function = "(U_X*coordsX+U_Y*coordsY)/sqrt(coordsX^2+coordsY^2)*1000"   # mm/s

# 36 copies of the 5-degree wedge (centred on +x) = the half annulus y >= 0
copies = []
for k in range(36):
    tr = Transform(Input=ur)
    tr.Transform = "Transform"
    tr.Transform.Rotate = [0.0, 0.0, 5.0 * k + 2.5]
    copies.append(tr)
annulus = MergeBlocks(Input=AppendDatasets(Input=copies))
surf = ExtractSurface(Input=annulus)

view = GetActiveViewOrCreate("RenderView")
view.ViewTime = time                      # otherwise the view asks for t = 0 (no data)
view.ViewSize = [2400, 1800]
view.Background = [0.984, 0.984, 0.973]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.OrientationAxesVisibility = 0

disp = Show(surf, view)
ColorBy(disp, ("POINTS", "ur"))
lut = GetColorTransferFunction("ur")
lut.ApplyPreset("Cool to Warm", True)
lut.RescaleTransferFunction(-150.0, 150.0)
disp.SetScalarBarVisibility(view, True)
bar = GetScalarBar(lut, view)
bar.Title = "radial velocity [mm/s]"
bar.ComponentTitle = ""
bar.TitleColor = [0.14, 0.12, 0.13]
bar.LabelColor = [0.14, 0.12, 0.13]
bar.TitleFontSize = 22
bar.LabelFontSize = 20
bar.ScalarBarLength = 0.40
bar.Orientation = "Horizontal"
bar.WindowLocation = "Any Location"
bar.Position = [0.30, 0.03]
bar.ScalarBarThickness = 20
bar.RangeLabelFormat = "%.0f"

# spinning inner cylinder (solid grey), slightly inside the fluid's inner radius
cyl = Cylinder(Radius=0.02535, Height=0.0381, Resolution=120, Center=[0.0, 0.0, 0.0])
cylt = Transform(Input=cyl)
cylt.Transform = "Transform"
cylt.Transform.Rotate = [90.0, 0.0, 0.0]          # cylinder source is along y; turn it to z
half = Clip(Input=cylt)                            # keep the y >= 0 half, like the annulus
half.ClipType = "Plane"
half.ClipType.Origin = [0.0, 0.0, 0.0]
half.ClipType.Normal = [0.0, 1.0, 0.0]
half.Invert = 0
cdisp = Show(half, view)
cdisp.DiffuseColor = [0.72, 0.71, 0.68]
cdisp.AmbientColor = [0.72, 0.71, 0.68]
cdisp.ColorArrayName = ["POINTS", ""]
# solid face where the cut plane slices the inner cylinder (hides the hollow inside)
face = Plane(Origin=[-0.02535, 0.0, -0.01905], Point1=[0.02535, 0.0, -0.01905], Point2=[-0.02535, 0.0, 0.01905])
fdisp = Show(face, view)
fdisp.DiffuseColor = [0.80, 0.79, 0.76]
fdisp.AmbientColor = [0.80, 0.79, 0.76]
fdisp.ColorArrayName = ["POINTS", ""]
# top cap of the half cylinder
cap = Clip(Input=Disk(InnerRadius=0.0, OuterRadius=0.02535, CircumferentialResolution=120))
cap.ClipType = "Plane"
cap.ClipType.Origin = [0.0, 0.0, 0.0]
cap.ClipType.Normal = [0.0, 1.0, 0.0]
cap.Invert = 0
capt = Transform(Input=cap)
capt.Transform = "Transform"
capt.Transform.Translate = [0.0, 0.0, 0.01907]   # just above the cylinder top (avoid z-fighting)
kdisp = Show(capt, view)
kdisp.DiffuseColor = [0.80, 0.79, 0.76]
kdisp.AmbientColor = [0.80, 0.79, 0.76]
kdisp.ColorArrayName = ["POINTS", ""]

view.CameraPosition = [0.0, -0.16, 0.06]
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelProjection = 0
view.ResetCamera(False)                   # fit the data, keep this viewing direction
GetActiveCamera().Dolly(1.2)

os.makedirs(os.path.dirname(out), exist_ok=True)
SaveScreenshot(out, view, ImageResolution=[2400, 1800], TransparentBackground=0)
print("wrote", out)
