import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from taylor_couette_mixing.envs.helpers import Helpers

# Helpers is finished
# need to write reset and step
# figure out render


class TaylorCouetteMixingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        case_path,
        render_mode=None,
        omega_min=-300.0,
        omega_max=300.0,
        omega_start=0.0,
        max_steps=60,
        warmup_omega_rpm=100.0,
        warmup_duration=10.0,
    ):
        self.helpers = Helpers(case_path)

        # Spin the case up once so dye reaches the measurement bins.
        # Cached as 0.warmed/; subsequent env instances skip this work.
        warmup_omega_rad = (warmup_omega_rpm * 2 * np.pi) / 60
        self.helpers._warmup_case(warmup_omega_rad, warmup_duration)

        self.max_steps = max_steps

        self.omega_start = omega_start
        self.omega = omega_start
        # Note that I and E will be [0,1].
        self.alpha = 1.0 # How much we care about mixing
        self.beta = 1.0 # How much we care about energy consumption
        self.I_current = 1.0 # Initially unmixed
        self.I_threshold = 0.01
        self.E_current = 0.0 # 0 initial energy consumption
        self.time_step = 1 # Number of seconds that the sim runs between steps
        self.step_count = 0

        self.omega_min  = omega_min
        self.omega_max = omega_max
        self.observation_space = spaces.Dict(
            {
                "omega": spaces.Box(low=omega_min, high=omega_max, shape=(1,), dtype=np.float64)
            }
        )

        # action[0] in [-1, 1] selects omega within [mean - amp, mean + amp]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)


        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        # """
        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        # """
        # self.window = None
        # self.clock = None

    def _get_obs(self):
        return {"omega": self.omega}

    def _get_info(self):
        return {
            "step_count": self.step_count,
            "mixing_index": self.I_current,
            "energy_consumption": self.E_current
            
        }
    
    def calculate_mixing_index(self, concentrations):
        """Calculating Intensity of Segregation.
        I_mix = (Variance in Concentration) / (Maximum Variance in Concentration)
        Over 20 radial bins we find the variance in concentration """
        r_in = 25.4 # radius of inner cylinder (mm)
        r_out = 31.75 # radious of outer cylinder (mm)
        nBins = 20

        C = np.asarray(concentrations, dtype=float)

        dr = (r_out - r_in) / nBins
        r_mids = r_in + (np.arange(nBins) + 0.5) * dr
        weights = r_mids / r_mids.sum()

        C_bar = np.sum(weights * C)
        sigma2 = np.sum(weights * (C - C_bar) ** 2)
        sigma2_max = C_bar * (1 - C_bar) + 0.0000000000000001 # avoid div by 0 error
        return sigma2 / sigma2_max

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # options={"reset_mode": "hard"|"soft"} picks whether the case
        # reverts to pristine 0.orig/ or continues from the last time dir.
        mode = (options or {}).get("reset_mode", "hard")
        self.helpers.reset_case(mode=mode)

        self.step_count = 0
        self.omega = self.omega_start
        self.E_current = 0.0
        self.I_current = 1.0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        self.omega = float(np.clip(self.omega + action[0]*50, self.omega_min, self.omega_max))
        omega_rad = (self.omega * (2 * np.pi)) / 60

        results = self.helpers.do_simulation(omega_rad, self.time_step)

        # Mz_kin is torque / density given by OpenFOAM
        powers = []
        times = []
        for result in results:
             Mz = result["Mz_kin"] * 1000
             powers.append(Mz * omega_rad)
             times.append(result["t"])
        E = -np.trapezoid(powers, times) # Energy consumption of this time step
        E_norm = E / 0.0011017031875434 # This is from a run with 52.4 rad/sec for 1 second AKA E_max in joules.

        # OpenFOAM simulates a r by h wedge of the annulus
        # There are 20 equally spaced bins between r_in and r_out at the bottom with concentrations at each.
        final_result = results[-1]
        concentrations = [final_result[f"C{i}"] for i in range(20)]
        # print("concentrations:", concentrations)
        mixing_index = self.calculate_mixing_index(concentrations)

        terminated = False
        truncated = (self.step_count >= self.max_steps)
        reward = -(self.alpha * mixing_index) - (self.beta * E_norm)
        observation = self._get_obs()

        self.E_current = self.E_current + E
        self.I_current = mixing_index
        self.step_count += 1
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()

    # def _render_frame(self):
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()

    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((255, 255, 255))
    #     pix_square_size = (
    #         self.window_size / self.size
    #     )  # The size of a single grid square in pixels

    #     # First we draw the target
    #     pygame.draw.rect(
    #         canvas,
    #         (255, 0, 0),
    #         pygame.Rect(
    #             pix_square_size * self._target_location,
    #             (pix_square_size, pix_square_size),
    #         ),
    #     )
    #     # Now we draw the agent
    #     pygame.draw.circle(
    #         canvas,
    #         (0, 0, 255),
    #         (self._agent_location + 0.5) * pix_square_size,
    #         pix_square_size / 3,
    #     )

    #     # Finally, add some gridlines
    #     for x in range(self.size + 1):
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (self.window_size, pix_square_size * x),
    #             width=3,
    #         )
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, self.window_size),
    #             width=3,
    #         )

    #     if self.render_mode == "human":
    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()

    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to
    #         # keep the framerate stable.
    #         self.clock.tick(self.metadata["render_fps"])
    #     else:  # rgb_array
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #         )

    # def close(self):
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()
