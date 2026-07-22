"""Landscape-emulator twin of TaylorCouetteModulationEnv (fixed-mean w_b=300).

NOT a CFD emulator: a learner test-rig. It reproduces only what the learner
experiences -- the 7-D obs, 5 blocks, the reward magnitudes and noise, and the
(w_low, D, T) reward landscape -- via k-NN regression over 6,600 real CFD
block-samples (build_dataset.py). Decode is INHERITED from the real env class,
so the action mapping is identical by construction.

Known deliberate approximations (fine for learner testing, do not use for
physics claims): obs slot 0 (wallFlux) is proxied by X_block (wallFlux was
never logged per block); cross-block memory is first-order (X_prev feature);
X noise is sigma=0.005 (per-block share of the measured +-0.01 episode floor).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

from taylor_couette_mixing.envs.taylor_couette_modulation import (  # noqa: E402
    TaylorCouetteModulationEnv,
)


class KNN:
    """Inverse-distance-weighted k-NN in standardized feature space."""

    def __init__(self, feats, targs, k=25):
        self.mu = feats.mean(axis=0)
        self.sd = feats.std(axis=0) + 1e-12
        self.z = (feats - self.mu) / self.sd
        self.targs = targs
        self.k = k

    def predict(self, f):
        zq = (np.asarray(f, float) - self.mu) / self.sd
        d = np.sqrt(((self.z - zq) ** 2).sum(axis=1))
        idx = np.argpartition(d, self.k)[: self.k]
        w = 1.0 / (d[idx] + 1e-6)
        return (self.targs[idx] * w[:, None]).sum(axis=0) / w.sum()


class SurrogateModulationEnv(TaylorCouetteModulationEnv):
    """Same interface + decode as the real env; dynamics from the k-NN."""

    def __init__(self, dataset=os.path.join(HERE, "dataset.npz"),
                 noise_x=0.005, seed=0,
                 w_b_rpm=300.0, duty_min=0.6, duty_max=1.0,
                 idle_min_rpm=0.0, idle_max_rpm=None,
                 period_min=0.5, period_max=5.0,
                 episode_duration=50.0, block_dt=10.0, p_max_watt=31.94):
        # Deliberately NOT calling super().__init__ (it builds OpenFOAM helpers).
        self.w_b = None if w_b_rpm is None else float(w_b_rpm)
        self.block_dt = float(block_dt)
        self.max_steps = int(round(episode_duration / block_dt))
        self.episode_duration = self.max_steps * self.block_dt
        self.duty_min, self.duty_max = float(duty_min), float(duty_max)
        self.idle_min = float(idle_min_rpm)
        if self.w_b is None:
            self.idle_max = self.idle_min
        else:
            self.idle_max = self.w_b if idle_max_rpm is None else float(idle_max_rpm)
        self.nom_min, self.nom_max = 0.0, 500.0
        self._logTmin = float(np.log(period_min))
        self._logTmax = float(np.log(period_max))
        self.ramp_time = 0.05
        self.p_max = float(p_max_watt)
        self.wallflux_max = 1.32e-8
        self.capture_episodes, self.capture_dir = set(), None
        self.episode_count = 0
        self.step_count = 0

        import gymnasium.spaces as spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,),
                                            dtype=np.float64)

        data = np.load(dataset)
        self.knn = KNN(data["feats"], data["targs"])
        self.noise_x = float(noise_x)
        self.rng = np.random.default_rng(seed)
        self._zero_state()

    def _get_obs(self):
        # wallFlux proxy: X_block (see module docstring).
        return np.array([
            self.x_block,
            self.delta_x,
            self.x_block,
            (self.step_count * self.block_dt) / self.episode_duration,
            self.prev_action[0],
            self.prev_action[1],
            self.prev_action[2],
        ], dtype=np.float64)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_count += 1
        self.step_count = 0
        self._zero_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        duty, w_low, period, w_hi = self._decode(action)
        x_hat, p_hat = self.knn.predict([w_hi, w_low, np.log(period), self.x_block])
        x_block = float(max(0.0, x_hat + self.rng.normal(0.0, self.noise_x)))
        reward = x_block - p_hat / self.p_max

        self.delta_x = x_block - self.x_block
        self.x_block = x_block
        self.wf_block = x_block * self.wallflux_max   # keep _get_info consistent
        self.p_block = float(p_hat)
        self.prev_action = np.clip(np.asarray(action, float).ravel(), -1.0, 1.0)
        self.last_params = (duty, w_low, period, w_hi)
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), False, truncated, self._get_info()


def scan_static(knn=None, n=21, p_max=31.94):
    """Ground truth BY CONSTRUCTION: roll every static (D, w_low, T) policy
    through the surrogate (noise-free) and return the best by episode return."""
    if knn is None:
        data = np.load(os.path.join(HERE, "dataset.npz"))
        knn = KNN(data["feats"], data["targs"])
    best = None
    grid_D = np.linspace(0.6, 1.0, n)
    grid_W = np.linspace(0.0, 300.0, n)
    grid_T = np.exp(np.linspace(np.log(0.5), np.log(5.0), n))
    for D in grid_D:
        for W in grid_W:
            for T in grid_T:
                x, ret, last_r = 0.0, 0.0, 0.0
                whi = (300.0 - (1.0 - D) * W) / D
                for _ in range(5):
                    xh, ph = knn.predict([whi, W, np.log(T), x])
                    last_r = xh - ph / p_max
                    ret += last_r
                    x = xh
                if best is None or ret > best[0]:
                    best = (ret, last_r, D, W, T)
    return {"return": best[0], "final_R": best[1],
            "duty": best[2], "w_low": best[3], "period": best[4]}


if __name__ == "__main__":
    opt = scan_static()
    print("surrogate grid-scan optimum (static policies, noise-free):")
    print(f"  D={opt['duty']:.3f}  w_low={opt['w_low']:.0f} rpm  T={opt['period']:.2f} s"
          f"  -> episode return {opt['return']:.4f}, final-block R {opt['final_R']:.4f}")
    env = SurrogateModulationEnv()
    obs, _ = env.reset(seed=0)
    ret = 0.0
    for _ in range(5):
        obs, r, te, tr, info = env.step([0.5, -1.0, 0.075])  # champion D=0.8 wlo=0 T~2.5
        ret += r
    print(f"  sanity: champion static episode on surrogate -> return {ret:.4f}, "
          f"final-block R {r:.4f} (real CFD benchmark: 0.270)")
