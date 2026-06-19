"""Electric-motor power model from Lopez-Guajardo et al., Chem. Eng. J. 489 (2024)
151174, Section 3.4 "Power consumption" (Eqs. 18-23).

Computes the electrical power P_e(t) drawn by the motor that drives the inner
cylinder, given the angular-speed waveform omega(t). This is the energy metric
the paper uses to show that pulsating modulation is CHEAPER than constant rotation
at the SAME conversion -- a result a purely mechanical (viscous-drag) energy
cannot reproduce, because the real drivetrain is dominated by bearing dry friction
(LINEAR in speed) and includes regenerative braking, not the convex drag-work term
alone.

Structure (matches the paper):
  T_mot = J*dw/dt + T0*sign(w) + (beta2*w^2*sign(w) + beta1*w)        (Eq 18)
          inertia   bearings      drag (full-machine polynomial fit)
  i_mot = T_mot / Kt                                                   (Eq 20)
  e_mot = L*di_mot/dt + R*i_mot + Ke*w                                 (Eq 21)
  P_mot = e_mot * i_mot                                                (Eq 22)
  P_e   = P_mot/eta  (motoring, P_mot>=0)  |  P_mot*eta  (regen, P_mot<0)  (Eq 23)

The drag torque uses the paper's full-machine polynomial (beta2, beta1), so P_e is
a pure function of the commanded omega(t) -- exactly as in the paper, where the CFD
supplies CONVERSION and this model supplies POWER. (If you ever want power tied to
*your* CFD drag instead, swap the drag term for rho*Mz_kin from the env.)

All angles in rad, angular speeds in rad/s, times in s, torque in N.m, power in W.
"""
import numpy as np

# --- drivetrain + motor constants (paper Section 3.4, p. 8) ------------------
J_TOT = 1.66e-4    # kg.m^2        moment of inertia of the rotating parts (Eq 18)
T0    = 0.1        # N.m           bearing dry-friction torque       (Eq 18, T_bear)
BETA2 = 7.19e-8    # N.m.s^2/rad^2 quadratic drag coefficient        (Eq 18, T_drag)
BETA1 = 1.51e-5    # N.m.s/rad     linear drag coefficient           (Eq 18, T_drag)
KT    = 0.0931     # N.m/A         torque constant (93.1 mNm/A)      (Eq 20)
KE    = 0.0931     # V.s/rad       back-EMF constant (~= KT)         (Eq 21)
L     = 0.28e-3    # H             winding inductance                (Eq 21)
R     = 0.178      # ohm           winding resistance                (Eq 21)
ETA   = 0.90       # -             motor-controller efficiency       (Eq 23)


def motor_torque(t, omega):
    """T_mot(t) from Eq 18 (inertia + bearings + hydrodynamic drag)."""
    t = np.asarray(t, float)
    omega = np.asarray(omega, float)
    omega_dot = np.gradient(omega, t)
    sgn = np.sign(omega)
    T_in = J_TOT * omega_dot                              # inertia
    T_bear = T0 * sgn                                     # bearing dry friction
    T_drag = BETA2 * omega**2 * sgn + BETA1 * omega       # viscous + pressure drag
    return T_in + T_bear + T_drag


def electrical_power(t, omega):
    """P_e(t): motor electrical power with controller efficiency (Eqs 20-23).

    P_e > 0 is power drawn from the supply (motoring); P_e < 0 is power returned
    by regenerative braking. Time-average / integrate this for the paper's metric.
    """
    t = np.asarray(t, float)
    omega = np.asarray(omega, float)
    T_mot = motor_torque(t, omega)
    i_mot = T_mot / KT                                    # Eq 20
    di_dt = np.gradient(i_mot, t)
    e_mot = L * di_dt + R * i_mot + KE * omega            # Eq 21
    P_mot = e_mot * i_mot                                 # Eq 22
    # Eq 23: pay 1/eta when motoring, recover only eta when braking (regen).
    return np.where(P_mot >= 0.0, P_mot / ETA, P_mot * ETA)


def average_power(t, omega):
    """Time-average electrical power over [t[0], t[-1]] (W)."""
    t = np.asarray(t, float)
    if t[-1] <= t[0]:
        return float("nan")
    return float(np.trapezoid(electrical_power(t, omega), t) / (t[-1] - t[0]))


def energy(t, omega):
    """Total electrical energy over the trace (J)."""
    t = np.asarray(t, float)
    return float(np.trapezoid(electrical_power(t, omega), t))


def drag_power(omega):
    """Mechanical viscous-drag power ONLY (the term our CFD env currently
    integrates), for contrast with the full motor model:
        P_drag = (beta2*w^2 + beta1*|w|) * |w|
    Convex in omega -> penalizes bursts; this is why drag-only energy makes
    modulation look expensive. A pure function of omega (no derivatives)."""
    omega = np.asarray(omega, float)
    return (BETA2 * omega**2 + BETA1 * np.abs(omega)) * np.abs(omega)


def average_drag_power(t, omega):
    """Time-average of drag_power over the trace (W)."""
    t = np.asarray(t, float)
    if t[-1] <= t[0]:
        return float("nan")
    return float(np.trapezoid(drag_power(omega), t) / (t[-1] - t[0]))
