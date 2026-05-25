from taylor_couette_mixing.envs.helpers import Helpers
import numpy as np

omega = 52.4  # rad/sec
time_step = 1 # seconds

helpers = Helpers("taylor_couette_mixing/cases/tc_mixing_case")
results = helpers.do_simulation(omega, time_step)

print(results[0])
print(results[0].keys())

# Mz_kin is torque / density given by OpenFOAM
powers = []
times = []
for result in results:
    Mz = result["Mz_kin"] * 1000
    powers.append(-Mz * omega)
    times.append(result["t"])
    E = np.trapezoid(powers, times) # Energy consumption of this time step

print(f"The energy consumption is {E} joules")