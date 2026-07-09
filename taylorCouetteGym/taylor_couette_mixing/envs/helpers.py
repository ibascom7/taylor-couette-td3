"""Helper functions for Taylor-Couette simulation in OpenFOAM

At each time step 
1. The sim will be paused
2. Omega will be changed
3. The sim will continue with updated omega
"""

import numpy as np
from pathlib import Path
import shutil
import subprocess

class Helpers():
    def __init__(self, case_path):
        self.case_path = case_path

    def _get_latest_time(self):
        """Gets the latest simulation time folder"""
        current_t = -1.0
        current_name = "0"
        for p in Path(self.case_path).iterdir():
            try:
                t = float(p.name)
                if t > current_t:
                    current_t = t
                    current_name = p.name
            except ValueError:
                pass
        return current_name
    
    def _set_omega(self, chosen_omega):
        """Changes the set angular velocity in the OpenFOAM case"""
        latest_time = self._get_latest_time()
        # Uses foamDictionary which is a command line tool for OpenFOAM
        subprocess.run(
            ["foamDictionary",
             "-entry", "boundaryField.inner_wall.omega",
             "-set", str(chosen_omega),
             f"{latest_time}/U"],
             cwd=self.case_path, check=True,
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    def _set_omega_ramp(self, omega_from, omega_to, ramp_time, time_step):
        """Set inner-wall omega as a LINEAR RAMP from omega_from to omega_to over
        the first `ramp_time` seconds of this step, then hold omega_to.

        Avoids a large instantaneous change in wall velocity (which spikes the
        Courant number on the first timestep after an RL action and can crash the
        solver at high omega). Implemented as a tabulated Function1 omega on the
        rotatingWallVelocity BC -- the same form make_case.py writes for the
        prescribed square wave, so it is known to parse in this OpenFOAM version.

        Times are ABSOLUTE simulation time: the table spans [t0, t_end] (t0 = the
        latest time dir we restart from), so the BC is never evaluated outside the
        table and out-of-bounds handling is irrelevant. The final point sits past
        t_end so the hold value is unambiguous.
        """
        latest_time = self._get_latest_time()
        t0 = float(latest_time)
        t_end = t0 + time_step
        pts = [
            (t0, omega_from),
            (t0 + ramp_time, omega_to),
            (t_end + 1.0, omega_to),
        ]
        table = "table (" + " ".join(f"({t:.6f} {w:.6f})" for t, w in pts) + ")"
        subprocess.run(
            ["foamDictionary",
             "-entry", "boundaryField.inner_wall.omega",
             "-set", table,
             f"{latest_time}/U"],
             cwd=self.case_path, check=True,
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._clear_omega_coeffs(latest_time)
        return True

    def _clear_omega_coeffs(self, latest_time):
        """Delete any stale `omegaCoeffs` sub-dict left in the inner_wall BC.

        CRITICAL: rotatingWallVelocity's omega is a Function1. When we `-set` the
        inline form `omega table ( ... )`, pimpleFoam READS it, then on write
        SERIALISES it back as the sub-dict form `omega table;` + `omegaCoeffs {
        values ( ... ); }`. On the NEXT step foamDictionary again `-set`s the inline
        `omega`, but OpenFOAM v2506 reads the coefficients from the coexisting
        `omegaCoeffs` sub-dict and IGNORES the inline data -- so the wall FREEZES at
        the first step's omega Function1 for the rest of the run (the env's per-step
        omega updates are silently shadowed). Removing omegaCoeffs after every set
        forces pimpleFoam to re-read the fresh inline table. `-remove` is a no-op
        (exit 0) when the sub-dict is absent, so this is safe on the first step too.
        """
        subprocess.run(
            ["foamDictionary",
             "-entry", "boundaryField.inner_wall.omegaCoeffs",
             "-remove",
             f"{latest_time}/U"],
             cwd=self.case_path, check=False,
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    def _parse_metrics(self, line):
        """Parses RL_metrics log line from system/conrolDict"""
        parts = dict(p.split("=") for p in line.split()[1:])
        return {k: float(v) for k, v in parts.items()}
    
    def _update_end_time(self, time_step):
        """Updates the endTime in controlDict to continue for (time_step) more seconds"""
        latest_time = float(self._get_latest_time())
        new_end = latest_time + time_step
        subprocess.run(
            ["foamDictionary",
             "-entry", "endTime",
             "-set", str(new_end),
             "system/controlDict"],
             cwd=self.case_path, check=True,
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    def set_write_interval(self, dt):
        """Set controlDict writeInterval so a field dir is written at EVERY step
        boundary. The env continues from the latest WRITTEN time each step, so with
        `writeControl adjustableRunTime` the step boundary (endTime = latest+time_step)
        is only written when it is a multiple of writeInterval. time_step < writeInterval
        (e.g. the freeform 0.5 s step vs writeInterval 1) -> nothing is written at
        0.5, 1.5, ... and _get_latest_time never advances. So writeInterval == time_step."""
        subprocess.run(
            ["foamDictionary", "-entry", "writeInterval", "-set", repr(float(dt)),
             "system/controlDict"],
            cwd=self.case_path, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    def reset_case(self, mode="hard"):
        """Reset the OpenFOAM case between RL episodes.

        mode="hard": restore 0/ from a cached IC snapshot. Prefers 0.warmed/
                     (post-spin-up state) when present, else 0.orig/ (pristine).
        mode="soft": promote the latest completed time directory to 0/ so
                     the next episode continues from the previous state.

        The first call ever (when 0.orig/ doesn't exist yet) snapshots the
        current 0/ into 0.orig/ so the true IC is preserved for future
        hard resets.
        """
        case = Path(self.case_path)
        zero = case / "0"
        orig = case / "0.orig"
        warmed = case / "0.warmed"

        if not zero.is_dir():
            raise RuntimeError(
                f"Refusing to reset: {zero} is missing. "
                "reset_case would leave the case with no initial fields."
            )

        # One-time snapshot of the pristine initial condition.
        if not orig.is_dir():
            shutil.copytree(zero, orig)

        hard_source = warmed if warmed.is_dir() else orig

        # Collect numeric time directories.
        time_dirs = []
        for p in case.iterdir():
            if not p.is_dir():
                continue
            try:
                time_dirs.append((float(p.name), p))
            except ValueError:
                continue

        if mode == "soft":
            non_zero = [(t, p) for t, p in time_dirs if p.name != "0"]
            if non_zero:
                _, latest = max(non_zero, key=lambda x: x[0])
                shutil.rmtree(zero)
                latest.rename(zero)
            # else: no later time exists yet, soft == hard degenerates to
            # "leave 0/ alone", which is already correct.
            for _, p in time_dirs:
                if p.exists() and p.name != "0":
                    shutil.rmtree(p)

        elif mode == "hard":
            for _, p in time_dirs:
                if p.name != "0":
                    shutil.rmtree(p)
            shutil.rmtree(zero)
            shutil.copytree(hard_source, zero)

        else:
            raise ValueError(f"Unknown reset mode: {mode!r}")

        pp = case / "postProcessing"
        if pp.is_dir():
            shutil.rmtree(pp)

        subprocess.run(
            ["foamDictionary",
             "-entry", "endTime",
             "-set", "0",
             "system/controlDict"],
            cwd=self.case_path, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    def _warmup_case(self, warmup_omega_rad, warmup_duration):
        """Spin the case up once and cache the result as 0.warmed/.

        Runs pimpleFoam at warmup_omega_rad for warmup_duration seconds
        starting from the pristine IC, then promotes the final time dir
        to 0.warmed/ so future hard resets begin from this post-spin-up
        state. Idempotent: returns False without doing work if 0.warmed/
        already exists.
        """
        case = Path(self.case_path)
        warmed = case / "0.warmed"
        if warmed.is_dir():
            return False

        # Start from pristine IC; reset_case uses 0.orig/ since 0.warmed/
        # doesn't exist yet.
        self.reset_case(mode="hard")
        self.do_simulation(warmup_omega_rad, warmup_duration)

        latest_name = self._get_latest_time()
        if latest_name == "0":
            raise RuntimeError("Warmup did not advance simulation time.")
        shutil.copytree(case / latest_name, warmed)
        return True

    def snapshot_frames(self, dest):
        """Copy the case's mesh + all numeric time dirs into dest for offline
        ParaView viewing, and drop an empty .foam file so it opens directly.

        Captures everything the OpenFOAM ParaView reader needs (constant/ mesh,
        system/controlDict, and every written time directory including 0/) so a
        single episode's trajectory can be animated. dest is overwritten if it
        already exists. ~9 MB for a 60 s run at writeInterval 1.
        """
        case = Path(self.case_path)
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        shutil.copytree(case / "constant", dest / "constant")
        shutil.copytree(case / "system", dest / "system")
        for p in case.iterdir():
            if not p.is_dir():
                continue
            try:
                float(p.name)  # numeric time directory (a frame)
            except ValueError:
                continue
            shutil.copytree(p, dest / p.name)

        (dest / f"{dest.name}.foam").touch()
        return True

    def do_simulation(self, chosen_omega, time_step, ramp_from=None, ramp_time=0.0):
        """Runs pimpleFoam for (time_step) seconds with chosen angular velocity.

        If ramp_from is not None and ramp_time > 0, the inner-wall omega is
        LINEARLY RAMPED from ramp_from to chosen_omega over the first ramp_time
        seconds of the step (then held), instead of jumping instantly -- this
        keeps the wall acceleration finite so a big RL action does not spike the
        Courant number on the first timestep (see _set_omega_ramp). Otherwise the
        omega is set instantly as a scalar (the original behavior, used by the
        mixing/constant envs and the warmup).

        Returns a list of the metrics for each time interval of the step
        {time, Mz_kin, concentrations}
        """
        if ramp_from is not None and ramp_time > 0:
            self._set_omega_ramp(ramp_from, chosen_omega, ramp_time, time_step)
        else:
            self._set_omega(chosen_omega)
        self._update_end_time(time_step)
        result = subprocess.run(
            ["pimpleFoam"],
             cwd=self.case_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pimpleFoam failed (exit {result.returncode}) at omega={chosen_omega} rad/s\n"
                f"--- stderr (tail) ---\n{result.stderr[-2000:]}\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
            )
        metric_lines= [l for l in result.stdout.splitlines() if l.startswith("METRICS")]
        if not metric_lines:
            raise RuntimeError(f"No METRICS in pimpleFoam output:\n{result.stderr[-500:]}")
        step_metrics = [self._parse_metrics(l) for l in metric_lines]
        return step_metrics

    def do_simulation_table(self, points, time_step):
        """Like do_simulation but drives inner-wall omega from an explicit
        tabulated Function1 -- `points` = list of (absolute_time, omega_rad) --
        over the next `time_step` seconds. Used by the waveform env to run a full
        square-wave segment per control step. Continues from the latest time dir
        (controlDict startFrom latestTime), same step-loop contract as do_simulation."""
        latest = self._get_latest_time()
        table = "table (" + " ".join(f"({t:.6f} {w:.6f})" for t, w in points) + ")"
        subprocess.run(
            ["foamDictionary",
             "-entry", "boundaryField.inner_wall.omega",
             "-set", table, f"{latest}/U"],
            cwd=self.case_path, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._clear_omega_coeffs(latest)   # else the sticky omegaCoeffs freezes omega (see _clear_omega_coeffs)
        self._update_end_time(time_step)
        result = subprocess.run(
            ["pimpleFoam"], cwd=self.case_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pimpleFoam failed (exit {result.returncode}) on waveform table\n"
                f"--- stderr (tail) ---\n{result.stderr[-2000:]}\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
            )
        metric_lines = [l for l in result.stdout.splitlines() if l.startswith("METRICS")]
        if not metric_lines:
            raise RuntimeError(f"No METRICS in pimpleFoam output:\n{result.stderr[-500:]}")
        return [self._parse_metrics(l) for l in metric_lines]
    
        

