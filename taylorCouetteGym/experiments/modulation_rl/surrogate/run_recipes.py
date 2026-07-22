"""Run the REAL TD3 learner against the surrogate env: reproduce the Carya
corner-pinning in seconds, then test fix recipes (UTD, exploration, TD3+BC).

Faithfully mirrors parallel_train.py's dynamics: random actions for the first
start_timesteps env steps, then policy + clipped Gaussian noise; done stored as
(terminated or truncated) [the fixed convention]; the learner paces `utd`
gradient steps per env step once past start_timesteps.

Usage (torch venv):
  python run_recipes.py --utd 1 --seed 0            # the Carya-equivalent recipe
  python run_recipes.py --utd 32 --seed 0 1 2       # a fix candidate, 3 seeds
  python run_recipes.py --utd 32 --bc 0.05 --seed 0 # + TD3+BC actor regularizer
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

import TD3  # noqa: E402  (the repo's actual learner)
from surrogate_env import SurrogateModulationEnv, scan_static  # noqa: E402


class ReplayBuffer:
    """Copy of train.py's buffer (import avoided: train.py drags in the envs)."""

    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size, self.ptr, self.size = max_size, 0, 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        t = lambda x: torch.FloatTensor(x).to(self.device)
        return (t(self.state[ind]), t(self.action[ind]), t(self.next_state[ind]),
                t(self.reward[ind]), t(self.not_done[ind]))


class TD3BC(TD3.TD3):
    """TD3 with the minimal TD3+BC actor regularizer (Fujimoto 2021): the actor
    also imitates the sampled batch actions, weighted so the Q term keeps scale.
    bc_alpha=0 -> exactly stock TD3."""

    def __init__(self, *a, bc_alpha=0.0, **k):
        super().__init__(*a, **k)
        self.bc_alpha = float(bc_alpha)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise
                           ).clamp(-self.max_action, self.max_action)
            tq1, tq2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(tq1, tq2)
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic.Q1(state, pi)
            lam = 1.0 / q.abs().mean().detach() if self.bc_alpha > 0 else 1.0
            actor_loss = -lam * q.mean() + self.bc_alpha * F.mse_loss(pi, action)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


def eval_policy(policy, env, episodes=3):
    """Deterministic rollouts; returns mean return and the last episode's
    decoded per-block params + raw actions."""
    rets, params, raws = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)
        done, ret = False, 0.0
        params, raws = [], []
        while not done:
            a = policy.select_action(np.asarray(obs, np.float32))
            obs, r, te, tr, info = env.step(a)
            ret += r
            done = te or tr
            params.append([info["duty"], info["w_low_rpm"], info["period_s"]])
            raws.append(np.clip(a, -1, 1))
        rets.append(ret)
    return float(np.mean(rets)), np.array(params), np.array(raws)


def run(seed, utd, episodes, expl_noise, start_timesteps, batch_size, bc_alpha,
        noise_x, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = SurrogateModulationEnv(seed=seed, noise_x=noise_x)
    policy = TD3BC(7, 3, 1.0, bc_alpha=bc_alpha)
    buffer = ReplayBuffer(7, 3)
    total = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            if total < start_timesteps:
                a = env.action_space.sample()
            else:
                a = (policy.select_action(np.asarray(obs, np.float32))
                     + rng.normal(0, expl_noise, 3)).clip(-1, 1)
            nobs, r, te, tr, _ = env.step(a)
            buffer.add(np.asarray(obs, np.float32), a, np.asarray(nobs, np.float32),
                       r, float(te or tr))
            obs, done, total = nobs, te or tr, total + 1
            if total >= start_timesteps and buffer.size >= batch_size:
                for _ in range(utd):
                    policy.train(buffer, batch_size)
    ret, params, raws = eval_policy(policy, env)
    pinned = bool((np.abs(raws).mean(axis=0) > 0.95).any())
    return ret, params, raws, pinned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--utd", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--expl_noise", type=float, default=0.2)
    ap.add_argument("--start_timesteps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--bc", type=float, default=0.0, help="TD3+BC alpha (0 = stock)")
    ap.add_argument("--noise_x", type=float, default=0.005)
    args = ap.parse_args()

    opt = scan_static()
    print(f"[target] grid-scan optimum: D={opt['duty']:.3f} w_low={opt['w_low']:.0f} "
          f"T={opt['period']:.2f} -> return {opt['return']:.4f}")
    for s in args.seed:
        ret, params, raws, pinned = run(
            s, args.utd, args.episodes, args.expl_noise, args.start_timesteps,
            args.batch_size, args.bc, args.noise_x)
        gap = opt["return"] - ret
        mp = params.mean(axis=0)
        print(f"[seed {s}] utd={args.utd} bc={args.bc}: eval return {ret:.4f} "
              f"(gap to optimum {gap:+.4f}) "
              f"policy D={mp[0]:.2f} w_low={mp[1]:.0f} T={mp[2]:.2f} "
              f"| raw means {np.round(raws.mean(axis=0), 2)} "
              f"| {'PINNED' if pinned else 'interior'}")


if __name__ == "__main__":
    main()
