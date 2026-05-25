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

    def do_simulation(self, chosen_omega, time_step):
        """Runs pimpleFoam for (time_step) seconds with chosen angular velocity.
        
        Returns a list of the metrics for each time interval of the step
        {time, Mz_kin, concentrations}
        """
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
    
        

