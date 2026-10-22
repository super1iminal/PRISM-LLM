"""Reinforcement learning that generalizes across instances: PPO (Schulman et al. 2017) trained on the training
instances, the usual way to learn one policy for a family of MDPs (procedurally generated levels, MiniGrid-style),
then frozen and certified on every target like the other baselines. This class is the gridworld version;
`params.model: explicit` gives `ExplicitPPO` (below), on any domain.

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
  features: null                    with observation: features, the features (null: gridworld_mdp.FEATURES)
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from baselines.base import Baseline, Rules
from baselines.decision_tree import policy_rules
from baselines.gridworld_mdp import FEATURES, GridMDP
from core.domain import Instance


def observations(mdp: GridMDP, kind: str, radius: int = 2, features: Sequence[str] = FEATURES) -> np.ndarray:
    """One row per state of `mdp`: the policy's input."""
    stage = np.array([s for (_, _, s, _) in mdp.states])
    flags = np.stack([stage > k for k in range(mdp.K)], axis=1).astype(np.float32)
    if kind == "features":
        return np.hstack([mdp.features[f].astype(np.float32) for f in features] + [flags])
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

    def __init__(self, data: dict, kind: str, radius: int, features: Sequence[str] = FEATURES):
        self.mdp = mdp = GridMDP(data)
        self.obs = torch.tensor(observations(mdp, kind, radius, features))
        stage = np.array([s for (_, _, s, _) in mdp.states])
        cell_goal = {g: k for k, g in enumerate(mdp.goals)}
        self.bad = np.array([cell_goal.get((r, c), -1) > s or (bool(mdp.cycle) and s < mdp.K and (r, c) == mdp.cycle[o])
                             for (r, c, s, o) in mdp.states])
        self.stage = stage
        self.done = stage == mdp.K


class _Net(nn.Module):
    def __init__(self, inputs: int, actions: int = 4):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.pi, self.v = nn.Linear(64, actions), nn.Linear(64, 1)

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

    def __new__(cls, domain=None, params=None):
        if cls is PPO and (params or {}).get("model") == "explicit":
            cls = ExplicitPPO
        return super().__new__(cls)

    def __init__(self, domain, params):
        super().__init__(domain, params)
        self.kind = params.get("observation", "features")
        self.radius = params.get("radius", 2)
        self.steps, self.envs = params.get("steps", 300000), params.get("envs", 32)
        self.horizon, self.seed = params.get("horizon", 200), params.get("seed", 0)
        self.sets = params.get("sets")
        self.features = params.get("features") or list(FEATURES)
        self.nets: Dict[str, _Net] = {}

    def fit(self, train_instances: Sequence[Instance]) -> None:
        by_id = {int(i.id): i for i in train_instances}
        for k, ids in enumerate(self.sets or [sorted(by_id)]):
            print(f"PPO ({self.kind}), training set {ids}:", flush=True)
            grids = [_Grid(by_id[i].data, self.kind, self.radius, self.features) for i in ids]
            self.nets[f"ppo_{self.kind}" + (f"_set{k}" if self.sets else "")] = train(
                grids, self.steps, self.envs, self.horizon, self.seed + k, log=lambda s: print(s, flush=True))

    def actions(self, target: Instance, name: str = None) -> np.ndarray:
        """The frozen policy's most likely action in every state of `target` (for the fast model)."""
        net = self.nets[name] if name else next(iter(self.nets.values()))
        mdp = GridMDP(target.data)
        with torch.no_grad():
            logits, _ = net(torch.tensor(observations(mdp, self.kind, self.radius, self.features)))
        return logits.argmax(dim=1).numpy()

    def rules(self, target: Instance) -> Dict[str, Rules]:
        mdp = GridMDP(target.data)
        return {name: policy_rules(mdp, self.actions(target, name)) for name in self.nets}


# ---------------------------------------------------------------- any domain, on the explicit model

class _Env:
    """The training instances' explicit models side by side (states renumbered one after another), sampled in
    numpy for many environments at once. The policy acts at decision states; every other step samples a choice
    uniformly (forced moves, and nondeterminism the policy does not control, such as Pac-Man's idle step) and then
    a successor. For the requirement `φ U ψ` (or `F ψ`): entering ψ gives -1 when the requirement bounds the
    probability from above (+1 from below) and ends the episode; so do leaving φ, a state from which ψ is
    unreachable, an absorbing state, and the instance's transition limit (`Domain.horizon`, else 1000), with 0."""

    def __init__(self, mdps, requirement: str, observe):
        self.actions = mdps[0].actions
        A = len(self.actions)
        offsets = np.cumsum([0] + [m.S for m in mdps])
        c_off = np.cumsum([0] + [m.C for m in mdps])
        self.initial = np.array([o + m.initial for o, m in zip(offsets, mdps)])
        self.horizon = np.array([m.domain.horizon(m.instance) or 1000 for m in mdps])
        self.choice_start = np.r_[np.concatenate([m.choice_start[:-1] + c for m, c in zip(mdps, c_off)]), c_off[-1]]
        self.n_choices = np.diff(self.choice_start)
        act = np.concatenate([m.choice_action for m in mdps])
        lengths = np.concatenate([np.diff(m.trans_start) for m in mdps])
        self.target = np.concatenate([m.target + o for m, o in zip(mdps, offsets)])
        prob = np.concatenate([m.prob for m in mdps])
        starts, cum = np.cumsum(lengths) - lengths, np.cumsum(prob)
        within = cum - np.repeat(cum[starts] - prob[starts], lengths)          # cumulative within each choice
        within /= np.repeat(within[starts + lengths - 1], lengths)
        self.key = np.repeat(np.arange(lengths.size), lengths) + within        # choice c's successors: keys in (c, c + 1]
        self.decision = np.concatenate([m.decision for m in mdps])
        stop, goal, sign = [], [], []
        for m in mdps:
            req = m.objective_requirement if requirement == "objective" else \
                next(r for r in m.spec.requirements if requirement in (None, r.name))
            stay, psi = m.path_sets(req.formula)
            reach = m.reach(req.formula, maximize=True) > 0
            stop.append((~stay & ~psi) | ~reach | m.absorbing)
            goal.append(psi)
            sign.append(-1.0 if req.bound == "<=" else 1.0)
        self.stop, self.goal, self.sign = np.concatenate(stop), np.concatenate(goal), sign[0]
        state = np.repeat(np.arange(offsets[-1]), self.n_choices)
        mine = act >= 0
        order = np.lexsort((act[mine], state[mine]))
        self.by_action = np.flatnonzero(mine)[order]
        pairs = state[mine][order] * A + act[mine][order]                       # sorted (state, action)
        self.first = np.full(offsets[-1] * A, -1)
        self.count = np.zeros(offsets[-1] * A, dtype=np.int64)
        np.add.at(self.count, pairs, 1)
        keys, firsts = np.unique(pairs, return_index=True)
        self.first[keys] = firsts
        if (self.count.reshape(-1, A)[self.decision] == 0).any():
            raise NotImplementedError("a decision state where some policy action is not enabled")
        rows = [observe(m, np.flatnonzero(m.decision)) for m in mdps]
        self.obs = torch.zeros((offsets[-1], rows[0].shape[1]), dtype=torch.float32)
        for m, o, r in zip(mdps, offsets, rows):
            self.obs[o + np.flatnonzero(m.decision)] = torch.tensor(r, dtype=torch.float32)

    def _move(self, envs, choices, state, t, which, reward, end, rng):
        nxt = self.target[np.searchsorted(self.key, choices + rng.random(choices.size), side="right")]
        state[envs], t[envs] = nxt, t[envs] + 1
        reward[envs] += np.where(self.goal[nxt], self.sign, 0.0)
        end[envs] |= self.goal[nxt] | self.stop[nxt] | (t[envs] >= self.horizon[which[envs]])

    def _advance(self, envs, state, t, which, reward, end, rng):
        """Move `envs` through states that are not decisions until a decision state or the end of the episode."""
        while True:
            moving = envs[~end[envs] & ~self.decision[state[envs]]]
            if not moving.size:
                return
            s = state[moving]
            self._move(moving, self.choice_start[s] + (rng.random(s.size) * self.n_choices[s]).astype(np.int64),
                       state, t, which, reward, end, rng)

    def reset(self, envs, state, t, which, rng, returns):
        """Start new episodes in `envs` and run them to their first decision state (episodes that end before any
        decision go to `returns`)."""
        while envs.size:
            state[envs], t[envs] = self.initial[which[envs]], 0
            reward, end = np.zeros(state.size), np.zeros(state.size, dtype=bool)
            self._advance(envs, state, t, which, reward, end, rng)
            returns += reward[envs[end[envs]]].tolist()
            envs = envs[end[envs]]

    def step(self, state, t, which, actions, rng):
        """Every environment takes its action at its decision state and runs to its next decision or end:
        (reward, end) per environment."""
        envs = np.arange(state.size)
        k = state * len(self.actions) + actions
        pick = self.first[k] + (rng.random(state.size) * self.count[k]).astype(np.int64)
        reward, end = np.zeros(state.size), np.zeros(state.size, dtype=bool)
        self._move(envs, self.by_action[pick], state, t, which, reward, end, rng)
        self._advance(envs, state, t, which, reward, end, rng)
        return reward, end


def train_explicit(env: _Env, steps: int, envs: int, seed: int, gamma: float, log=print) -> _Net:
    """PPO with GAE on `env`, `envs` environments, each bound to one training instance in turn; `steps` decisions."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = _Net(env.obs.shape[1], len(env.actions))
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    which = np.arange(envs) % len(env.initial)
    state, t = np.zeros(envs, dtype=np.int64), np.zeros(envs, dtype=np.int64)
    returns, ep_return = [], np.zeros(envs)
    env.reset(np.arange(envs), state, t, which, rng, returns)
    rollout, lam = 128, 0.95
    for update in range(max(1, steps // (rollout * envs))):
        obs_b, act_b, logp_b, val_b, rew_b, end_b = [], [], [], [], [], []
        for _ in range(rollout):
            obs = env.obs[state]
            with torch.no_grad():
                logits, value = net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
            rew, end = env.step(state, t, which, act.numpy(), rng)
            obs_b.append(obs); act_b.append(act); logp_b.append(dist.log_prob(act)); val_b.append(value)
            rew_b.append(torch.tensor(rew, dtype=torch.float32)); end_b.append(torch.tensor(end, dtype=torch.float32))
            ep_return += rew
            done = np.flatnonzero(end)
            returns += ep_return[done].tolist()
            ep_return[done] = 0.0
            env.reset(done, state, t, which, rng, returns)
        with torch.no_grad():
            _, last = net(env.obs[state])
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
        if update % 20 == 0 and returns:
            log(f"  update {update}: mean return of the last episodes {np.mean(returns[-2000:]):.3f}")
    return net


class ExplicitPPO(PPO):
    """PPO on any domain (`params.model: explicit`), sampling the training instances' explicit models
    (explicit_mdp.py, `_Env`): remaining nondeterminism, such as Pac-Man's idle step, is resolved uniformly in
    training. Observations at decision states: `raw`, the policy-visible variables scaled by their declared
    ranges over the training instances (fixed, so longer horizons extrapolate); or `features`, the named
    rule-language features (core evaluator), integer ones scaled by their largest value in training. Frozen as
    the most likely action in every decision state, a per-state policy certified through each target's explicit
    model. Sources: one per training set and seed.

    params:
      model: explicit
      observation: raw | features
      features: null          with observation: features, the feature names (null: every feature of the domain)
      steps: 1000000          decisions of training per seed
      envs: 32
      seeds: [0]
      gamma: 1.0              the requirement is an undiscounted probability
      sets: null              training sets (a mapping name -> ids, a list, or null: all training ids)
      requirement: null       the requirement whose target the reward marks (null: the first; objective: the
                              domain's objective model and requirement, Domain.objective)
      certify: explicit       explicit: through each target's explicit model; rules: as one rule per decision
                              state (`rules`), certified like any rule list (requirements explicit_mdp cannot check,
                              such as expected rewards)
    """
    explicit = True

    def __init__(self, domain, params):
        Baseline.__init__(self, domain, params)
        self.kind = params.get("observation", "raw")
        if self.kind not in ("raw", "features"):
            raise ValueError(f"unknown observation {self.kind!r} (raw | features)")
        self.feature_names = params.get("features")
        self.steps, self.envs = params.get("steps", 1000000), params.get("envs", 32)
        self.seeds, self.gamma = params.get("seeds", [0]), params.get("gamma", 1.0)
        self.sets, self.requirement = params.get("sets"), params.get("requirement")
        if params.get("certify", "explicit") == "rules":
            self.explicit = False       # certified as rules (`rules`), e.g. for reward requirements
        self.nets: Dict[str, _Net] = {}
        self.ids: Dict[str, List[int]] = {}
        self.scale = None

    def observe(self, mdp, states: np.ndarray) -> np.ndarray:
        """The network's input on `states` of `mdp` (an explicit_mdp.ExplicitMDP)."""
        if self.kind == "raw":
            x = mdp.obs[states].astype(np.float32)
        else:
            x = mdp.features(states, self.feature_names or [f.name for f in mdp.spec.features])[0].astype(np.float32)
        if self.scale is None:
            raise RuntimeError("observe before fit")
        return (x - self.scale[0]) / self.scale[1]

    def fit(self, train: Sequence[Instance]) -> None:
        from baselines.base import named_sets
        from baselines.explicit_mdp import ExplicitMDP
        from config import load_config
        by_id = {int(i.id): i for i in train}
        config = self.config or load_config()
        mdps = {i: ExplicitMDP(self.domain, inst, config, objective=self.requirement == "objective")
                for i, inst in by_id.items()}
        if self.kind == "raw":
            spec_vars = [m.spec.variables for m in mdps.values()]
            low = np.array([min(vs[j].low for vs in spec_vars) for j in range(len(spec_vars[0]))], dtype=np.float32)
            high = np.array([max(vs[j].high for vs in spec_vars) for j in range(len(spec_vars[0]))], dtype=np.float32)
            self.scale = (low, np.maximum(high - low, 1.0))
        else:
            names = self.feature_names or [f.name for f in next(iter(mdps.values())).spec.features]
            top = np.max(np.vstack([m.features(np.flatnonzero(m.decision), names)[0] for m in mdps.values()]), axis=0)
            self.scale = (np.zeros_like(top, dtype=np.float32), np.maximum(top, 1).astype(np.float32))
        for name, ids in named_sets(self.sets, list(by_id)).items():
            env = _Env([mdps[i] for i in ids], self.requirement, self.observe)
            for seed in self.seeds:
                source = f"ppo_{self.kind}" + (f"_{name}" if name else "") + f"_seed{seed}"
                print(f"PPO {source}, training set {ids}:", flush=True)
                self.nets[source] = train_explicit(env, self.steps, self.envs, seed, self.gamma,
                                                   log=lambda s: print(s, flush=True))
                self.ids[source] = ids

    def policies(self, mdp) -> Dict[str, np.ndarray]:
        states = np.flatnonzero(mdp.decision)
        x = torch.tensor(self.observe(mdp, states))
        out = {}
        for source, net in self.nets.items():
            policy = np.full(mdp.S, -1, dtype=np.int64)
            with torch.no_grad():
                policy[states] = torch.cat([net(b)[0].argmax(dim=1) for b in x.split(65536)]).numpy()
            out[source] = policy
        return out

    def rules(self, target: Instance) -> Dict[str, Rules]:
        """One rule per decision state (explicit_mdp.per_state_rules): small targets only."""
        from baselines.explicit_mdp import ExplicitMDP, per_state_rules
        from config import load_config
        mdp = ExplicitMDP(self.domain, target, self.config or load_config())
        return {s: per_state_rules(mdp, p) for s, p in self.policies(mdp).items()}

    def training(self, source: str) -> Optional[List[int]]:
        return self.ids[source]
