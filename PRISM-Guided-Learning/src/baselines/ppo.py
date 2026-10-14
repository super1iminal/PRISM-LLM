"""Reinforcement learning that generalizes across instances: PPO (Schulman et al. 2017) trained on the training
instances, the usual way to learn one policy for a family of MDPs (procedurally generated levels, MiniGrid-style),
then frozen and certified on every target like the other baselines. Gridworld only.

Environment: the numpy gridworld (gridworld_mdp.GridMDP), one training instance per parallel environment. Reward:
+1 for each goal reached in order, -1 and the episode ends when a later goal is entered early or the agent meets
the moving obstacle, -0.01 per step; an episode also ends when every goal is reached or after `horizon` steps.

Observation (`observation`):
  features   the rule-language features of all four actions and the goal flags: the information the rules and
             1-2-3-Go! with our features get;
  view       MiniGrid-style: a (2r+1)^2 window around the agent (walls and grid edge, patrol cells, the moving
             obstacle now, the next goal, later goals), the sign of the row and column offset to the next goal,
             and the goal flags.

The frozen policy takes its most likely action in every state; on a target it is written as rules over the cells
(decision_tree.policy_rules) and certified.

params:
  observation: features | view      (default features)
  radius: 2                         view window radius
  steps: 300000                     environment steps of training
  envs: 32                          parallel environments
  horizon: 200                      episode length cap
  seed: 0
  sets: null                        training sets as lists of ids, one policy each; null: one from all training ids
"""
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from baselines.base import Baseline, Rules
from baselines.decision_tree import policy_rules
from baselines.gridworld_mdp import FEATURES, GridMDP
from core.domain import Instance


def observations(mdp: GridMDP, kind: str, radius: int = 2) -> np.ndarray:
    """One row per state of `mdp`: the policy's input."""
    stage = np.array([s for (_, _, s, _) in mdp.states])
    flags = np.stack([stage > k for k in range(mdp.K)], axis=1).astype(np.float32)
    if kind == "features":
        return np.hstack([mdp.features[f].astype(np.float32) for f in FEATURES] + [flags])
    if kind != "view":
        raise ValueError(f"unknown observation {kind!r} (features | view)")
    patrol = set(mdp.cycle)
    rows = []
    for (r, c, s, o) in mdp.states:
        nxt = mdp.goals[s] if s < mdp.K else None
        later = set(mdp.goals[s + 1:]) if s < mdp.K else set()
        window = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                cell = (r + dr, c + dc)
                inside = 0 <= cell[0] < mdp.n and 0 <= cell[1] < mdp.n
                window += [float(not inside or cell in mdp.static), float(cell in patrol),
                           float(bool(mdp.cycle) and cell == mdp.cycle[o]), float(cell == nxt), float(cell in later)]
        offset = [np.sign(nxt[0] - r), np.sign(nxt[1] - c)] if nxt else [0.0, 0.0]
        rows.append(window + offset)
    return np.hstack([np.array(rows, dtype=np.float32), flags])


class _Grid:
    """A training instance as arrays: observations, successors and the reward and end of each transition."""

    def __init__(self, data: dict, kind: str, radius: int):
        self.mdp = mdp = GridMDP(data)
        self.obs = torch.tensor(observations(mdp, kind, radius))
        stage = np.array([s for (_, _, s, _) in mdp.states])
        cell_goal = {g: k for k, g in enumerate(mdp.goals)}
        self.bad = np.array([cell_goal.get((r, c), -1) > s or (bool(mdp.cycle) and s < mdp.K and (r, c) == mdp.cycle[o])
                             for (r, c, s, o) in mdp.states])
        self.stage = stage
        self.done = stage == mdp.K


class _Net(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.pi, self.v = nn.Linear(64, 4), nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def train(grids: List[_Grid], steps: int, envs: int, horizon: int, seed: int, log=print) -> _Net:
    """PPO with GAE on the training grids, `envs` environments in parallel (each bound to one grid)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = _Net(grids[0].obs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    which = np.arange(envs) % len(grids)
    state = np.array([grids[g].mdp.initial for g in which])
    t = np.zeros(envs, dtype=int)
    rollout, gamma, lam = 128, 0.99, 0.95
    returns_log, ep_return = [], np.zeros(envs)
    for update in range(max(1, steps // (rollout * envs))):
        obs_b, act_b, logp_b, val_b, rew_b, end_b = [], [], [], [], [], []
        for _ in range(rollout):
            obs = torch.stack([grids[g].obs[s] for g, s in zip(which, state)])
            with torch.no_grad():
                logits, value = net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
            a = act.numpy()
            nxt = np.empty(envs, dtype=int)
            rew = np.full(envs, -0.01)
            end = np.zeros(envs, dtype=bool)
            for e in range(envs):
                grid = grids[which[e]]
                m = rng.choice(3, p=grid.mdp.prob[a[e]])
                nxt[e] = grid.mdp.succ[state[e], a[e], m]
                rew[e] += grid.stage[nxt[e]] - grid.stage[state[e]]
                if grid.bad[nxt[e]]:
                    rew[e] -= 1.0
                end[e] = grid.bad[nxt[e]] or grid.done[nxt[e]] or t[e] + 1 >= horizon
            obs_b.append(obs); act_b.append(act); logp_b.append(dist.log_prob(act)); val_b.append(value)
            rew_b.append(torch.tensor(rew, dtype=torch.float32)); end_b.append(torch.tensor(end, dtype=torch.float32))
            ep_return += rew
            for e in np.flatnonzero(end):
                returns_log.append(ep_return[e])
                ep_return[e], t[e] = 0.0, 0
                nxt[e] = grids[which[e]].mdp.initial
            t[~end] += 1
            state = nxt
        with torch.no_grad():
            _, last = net(torch.stack([grids[g].obs[s] for g, s in zip(which, state)]))
        adv, gae = [], torch.zeros(envs)
        for k in reversed(range(rollout)):
            nv = last if k == rollout - 1 else val_b[k + 1]
            delta = rew_b[k] + gamma * nv * (1 - end_b[k]) - val_b[k]
            gae = delta + gamma * lam * (1 - end_b[k]) * gae
            adv.insert(0, gae)
        adv = torch.stack(adv).reshape(-1)
        ret = adv + torch.stack(val_b).reshape(-1)
        obs_all, act_all, logp_all = torch.cat(obs_b), torch.cat(act_b), torch.cat(logp_b)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(4):
            for idx in torch.randperm(len(act_all)).split(512):
                logits, value = net(obs_all[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act_all[idx]) - logp_all[idx])
                pg = -torch.min(ratio * adv[idx], torch.clamp(ratio, 0.8, 1.2) * adv[idx]).mean()
                loss = pg + 0.5 * ((value - ret[idx]) ** 2).mean() - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if update % 20 == 0 and returns_log:
            log(f"  update {update}: mean return of the last episodes {np.mean(returns_log[-200:]):.3f}")
    return net


class PPO(Baseline):

    def __init__(self, domain, params):
        super().__init__(domain, params)
        self.kind = params.get("observation", "features")
        self.radius = params.get("radius", 2)
        self.steps, self.envs = params.get("steps", 300000), params.get("envs", 32)
        self.horizon, self.seed = params.get("horizon", 200), params.get("seed", 0)
        self.sets = params.get("sets")
        self.nets: Dict[str, _Net] = {}

    def fit(self, train_instances: Sequence[Instance]) -> None:
        by_id = {int(i.id): i for i in train_instances}
        for k, ids in enumerate(self.sets or [sorted(by_id)]):
            print(f"PPO ({self.kind}), training set {ids}:", flush=True)
            grids = [_Grid(by_id[i].data, self.kind, self.radius) for i in ids]
            self.nets[f"ppo_{self.kind}" + (f"_set{k}" if self.sets else "")] = train(
                grids, self.steps, self.envs, self.horizon, self.seed + k, log=lambda s: print(s, flush=True))

    def actions(self, target: Instance, name: str = None) -> np.ndarray:
        """The frozen policy's most likely action in every state of `target` (for the fast model)."""
        net = self.nets[name] if name else next(iter(self.nets.values()))
        mdp = GridMDP(target.data)
        with torch.no_grad():
            logits, _ = net(torch.tensor(observations(mdp, self.kind, self.radius)))
        return logits.argmax(dim=1).numpy()

    def rules(self, target: Instance) -> Dict[str, Rules]:
        mdp = GridMDP(target.data)
        return {name: policy_rules(mdp, self.actions(target, name)) for name in self.nets}
