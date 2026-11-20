"""loom.py — LOOM: the unified budgeted compute allocator.

A small policy network maps SENSE features + remaining budget to a factored joint action:
  depth k in {1..L}   (how many layers to run for the next token)
  retrieve r in {0,1} (query THREAD before prefill — decided at step 0)
  verify   v in {0,1} (run ORACLE on the next token)
  spec     w in {0,2,4} (SHUTTLE block width for the next step; 0 = greedy)

Learned ONLINE from verified outcomes: REINFORCE with an EMA baseline, plus a dual price
(lambda_B) on budget usage that acts as the "price of compute" in the internal market.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class LoomPolicy(nn.Module):
    def __init__(self, feat_dim, n_layers, spec_choices=(0, 2, 4), hidden=48):
        super().__init__()
        self.n_layers = n_layers
        self.spec_choices = list(spec_choices)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.depth_head = nn.Linear(hidden, n_layers)
        self.retrieve_head = nn.Linear(hidden, 2)
        self.verify_head = nn.Linear(hidden, 2)
        self.spec_head = nn.Linear(hidden, len(self.spec_choices))

    def forward(self, feat):
        h = self.net(feat)
        return (Categorical(logits=self.depth_head(h)),
                Categorical(logits=self.retrieve_head(h)),
                Categorical(logits=self.verify_head(h)),
                Categorical(logits=self.spec_head(h)))


class Loom:
    """Policy + online learning (REINFORCE + dual price)."""

    def __init__(self, feat_dim, n_layers, lr=3e-3, spec_choices=(0, 2, 4),
                 cost_depth=1.0, cost_retrieve=2.0, cost_verify=3.0, cost_spec_unit=0.5):
        self.policy = LoomPolicy(feat_dim, n_layers, spec_choices)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.lambda_B = 0.05          # dual price of budget
        self.costs = dict(depth=cost_depth, retrieve=cost_retrieve,
                          verify=cost_verify, spec_unit=cost_spec_unit)
        self.spec_choices = list(spec_choices)

    # ---- action selection -------------------------------------------------------
    def act(self, feat, budget_left, budget_total, training=False):
        """Returns (action dict, logp tensor, cost). logp keeps its graph when called under
        enable_grad so the engine can run REINFORCE through it."""
        f = torch.tensor(feat, dtype=torch.float32)
        d, r, v, s = self.policy(f)
        if training:
            di, ri, vi, si = d.sample(), r.sample(), v.sample(), s.sample()
            logp = d.log_prob(di) + r.log_prob(ri) + v.log_prob(vi) + s.log_prob(si)
        else:
            di, ri, vi, si = d.probs.argmax(-1), r.probs.argmax(-1), v.probs.argmax(-1), s.probs.argmax(-1)
            logp = d.log_prob(di) + r.log_prob(ri) + v.log_prob(vi) + s.log_prob(si)
        # budget guard: cannot overspend
        depth = int(di) + 1
        spec = self.spec_choices[int(si)]
        retrieve = int(ri)
        verify = int(vi)
        cost = self.cost(depth, retrieve, verify, spec)
        if cost > budget_left:                     # market clearing: too expensive -> cheapest
            depth, spec, retrieve, verify = 1, 0, 0, 0
        return dict(depth=depth, spec=spec, retrieve=retrieve, verify=verify), logp, cost

    def cost(self, depth, retrieve, verify, spec):
        c = self.costs
        return depth * c["depth"] + retrieve * c["retrieve"] + verify * c["verify"] + spec * c["spec_unit"]

    # ---- online learning ----------------------------------------------------------
    def update(self, trajectory_logp, reward, spent, budget):
        """REINFORCE with EMA baseline + dual price update on the budget constraint."""
        adv = reward - self.baseline
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        loss = -trajectory_logp * adv
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # dual ascent on budget multiplier: spent > budget -> raise price of compute
        self.lambda_B = max(0.0, self.lambda_B + 0.01 * (spent - budget))
        return float(adv)
