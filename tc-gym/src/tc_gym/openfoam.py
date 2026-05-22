"""OpenFOAM case driver for the Taylor-Couette mixing environment.

This module wraps the file/IO contract between the RL loop and an
OpenFOAM `pimpleFoam` case: setting the inner-wall angular velocity,
advancing the simulation by a fixed duration, parsing per-step
``METRICS`` log lines emitted by the in-case `rlMetrics` function
object, and restoring the case between RL episodes.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


METRICS_PREFIX = "METRICS"


class OpenFOAMCase:
    """Thin driver for an OpenFOAM pimpleFoam case used as a Gym backend."""

    def __init__(self, case_path: str | Path):
        self.case_path = Path(case_path)
        if not self.case_path.is_dir():
            raise FileNotFoundError(f"Case directory not found: {self.case_path}")

    # ------------------------------------------------------------------ #
    # Time-directory bookkeeping
    # ------------------------------------------------------------------ #

    def _numeric_time_dirs(self) -> List[tuple[float, Path]]:
        out: List[tuple[float, Path]] = []
        for p in self.case_path.iterdir():
            if not p.is_dir():
                continue
            try:
                out.append((float(p.name), p))
            except ValueError:
                continue
        return out

    def latest_time_name(self) -> str:
        """Name (string) of the most recent numeric time directory."""
        dirs = self._numeric_time_dirs()
        if not dirs:
            return "0"
        return max(dirs, key=lambda kv: kv[0])[1].name

    # ------------------------------------------------------------------ #
    # foamDictionary helpers
    # ------------------------------------------------------------------ #

    def _foam_dict_set(self, entry: str, value: str, file_rel: str) -> None:
        subprocess.run(
            ["foamDictionary", "-entry", entry, "-set", value, file_rel],
            cwd=self.case_path,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def set_inner_wall_omega(self, omega_rad: float) -> None:
        """Write the inner-wall angular velocity (rad/s) into latest `U` file."""
        latest = self.latest_time_name()
        self._foam_dict_set(
            "boundaryField.inner_wall.omega", str(omega_rad), f"{latest}/U"
        )

    def set_end_time(self, end_time: float) -> None:
        """Set `endTime` in `system/controlDict`."""
        self._foam_dict_set("endTime", str(end_time), "system/controlDict")

    # ------------------------------------------------------------------ #
    # Reset semantics
    # ------------------------------------------------------------------ #

    def reset(self, mode: str = "hard") -> None:
        """Restore the case between RL episodes.

        mode="hard": replace `0/` with a cached IC snapshot. Prefers
                     `0.warmed/` (post-spin-up state) when present, else
                     `0.orig/` (pristine).
        mode="soft": promote the latest time directory to `0/` so the
                     next episode continues from where the last one
                     ended.

        On first ever call, snapshots the current `0/` into `0.orig/`
        so the pristine IC is preserved for future hard resets.
        """
        case = self.case_path
        zero = case / "0"
        orig = case / "0.orig"
        warmed = case / "0.warmed"

        if not zero.is_dir():
            raise RuntimeError(
                f"Refusing to reset: {zero} is missing. Cannot recover "
                "without an initial-condition directory."
            )

        if not orig.is_dir():
            shutil.copytree(zero, orig)

        time_dirs = self._numeric_time_dirs()

        if mode == "hard":
            for _, p in time_dirs:
                if p.name != "0":
                    shutil.rmtree(p)
            shutil.rmtree(zero)
            source = warmed if warmed.is_dir() else orig
            shutil.copytree(source, zero)

        elif mode == "soft":
            non_zero = [(t, p) for t, p in time_dirs if p.name != "0"]
            if non_zero:
                _, latest = max(non_zero, key=lambda kv: kv[0])
                shutil.rmtree(zero)
                latest.rename(zero)
            for _, p in time_dirs:
                if p.exists() and p.name != "0":
                    shutil.rmtree(p)

        else:
            raise ValueError(f"Unknown reset mode: {mode!r}")

        pp = case / "postProcessing"
        if pp.is_dir():
            shutil.rmtree(pp)

        self.set_end_time(0.0)

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_metrics_line(line: str) -> Dict[str, float]:
        parts = dict(p.split("=") for p in line.split()[1:])
        return {k: float(v) for k, v in parts.items()}

    def simulate(self, omega_rad: float, duration: float) -> List[Dict[str, float]]:
        """Run pimpleFoam for `duration` seconds at the given inner-wall omega.

        Returns the list of per-interval metric dicts parsed from
        `METRICS` log lines (one per write interval).
        """
        self.set_inner_wall_omega(omega_rad)
        start_time = float(self.latest_time_name())
        self.set_end_time(start_time + duration)

        result = subprocess.run(
            ["pimpleFoam"],
            cwd=self.case_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pimpleFoam failed (exit {result.returncode}) at "
                f"omega={omega_rad:.4f} rad/s\n"
                f"--- stderr (tail) ---\n{result.stderr[-2000:]}\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
            )

        metric_lines = [
            ln for ln in result.stdout.splitlines() if ln.startswith(METRICS_PREFIX)
        ]
        if not metric_lines:
            raise RuntimeError(
                "pimpleFoam produced no METRICS lines. Check the rlMetrics "
                "function object in system/controlDict.\n"
                f"--- stderr (tail) ---\n{result.stderr[-500:]}"
            )
        return [self._parse_metrics_line(ln) for ln in metric_lines]

    def warmup(self, omega_rad: float, duration: float) -> bool:
        """Spin the case up once and cache the result as `0.warmed/`.

        Idempotent: returns ``False`` without doing work if `0.warmed/`
        already exists. Otherwise, hard-resets to pristine, runs for
        `duration` seconds at `omega_rad`, then promotes the final
        time directory to `0.warmed/` so future hard resets start
        from the post-spin-up state.
        """
        warmed = self.case_path / "0.warmed"
        if warmed.is_dir():
            return False

        self.reset(mode="hard")
        self.simulate(omega_rad, duration)

        latest = self.latest_time_name()
        if latest == "0":
            raise RuntimeError("Warmup did not advance simulation time.")
        shutil.copytree(self.case_path / latest, warmed)
        return True
