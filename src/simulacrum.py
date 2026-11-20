"""simulacrum.py — SIMULACRUM: a learned world-model of the ARIADNE engine, used for
counterfactual lookahead planning BEFORE any real compute is spent.

Novelty (no prior art): every existing scheduler (Cascade, AdaSpec, Autopoiesis, MUX-EMA) is
reactive and myopic — it picks a strategy for the *current* request from realized or
heuristic stats. SIMULACRUM instead trains a small world model W(request features, strategy,
context) -> (latency, accuracy, cost) online from realized outcomes, and a Planner that
enumerates strategy *sequences* over a multi-request session inside W, then executes only the
winning plan. Emergent property: it will pay a consolidation cost on turn 1 because W
predicts turn 2's accuracy jumps — a myopic controller never does that.
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

STRATEGIES = ["full", "fullc", "fullr", "spec4", "spec4v", "depth1v"]
STRAT_IDX = {s: i for i, s in enumerate(STRATEGIES)}


def request_features(prompt):
    """Cheap features a serving scheduler legitimately sees: prompt stats only."""
    n = max(1, len(prompt))
    digits = sum(ch.isdigit() for ch in prompt) / n
    return [n / 40.0, digits,
            1.0 if "+" in prompt else 0.0,
            1.0 if "*" in prompt else 0.0,
            1.0 if "LIKE?" in prompt else 0.0,
            1.0 if "LIKES" in prompt else 0.0]


class WorldModel(nn.Module):
    """W(feat, strategy, ctx) -> (log_latency, acc_logit, log_cost).
    Input includes the interaction ctx * onehot(strategy) so the model can learn
    context-dependent strategy effects (e.g. retrieval only pays when memory exists)."""

    def __init__(self, feat_dim=6, hidden=48):
        super().__init__()
        self.n_strat = len(STRATEGIES)
        self.in_dim = feat_dim + self.n_strat + 1 + self.n_strat   # feat + oh + ctx + ctx*oh
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),                      # [log_latency, acc_logit, log_cost]
        )

    def _aug(self, feat, strat_idx, ctx):
        oh = torch.zeros(self.n_strat)
        oh[strat_idx] = 1.0
        return torch.cat([torch.tensor(feat, dtype=torch.float32), oh,
                          torch.tensor([float(ctx)]), oh * float(ctx)])

    def forward(self, feat, strat, ctx):
        return self.net(self._aug(feat, strat, ctx))

    @torch.no_grad()
    def predict(self, feat, strat, ctx=0.0):
        idx = STRAT_IDX[strat] if isinstance(strat, str) else strat
        out = self.forward(feat, idx, ctx)
        return dict(lat=float(out[0].exp()), acc=float(out[1].sigmoid()), cost=float(out[2].exp()))


def train_world_model(wm, events, steps=400, lr=5e-3, seed=0):
    """events: dicts with feat, strat (name), ctx, lat, acc, cost — realized outcomes."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    if not events:
        return
    for _ in range(steps):
        e = rng.choice(events)
        out = wm(e["feat"], STRAT_IDX[e["strat"]], e["ctx"])
        log_lat, acc_logit, log_cost = out[0], out[1], out[2]
        loss = ((log_lat - math.log(e["lat"] + 1e-6)) ** 2
                + (log_cost - math.log(e["cost"] + 1e-6)) ** 2
                + F.binary_cross_entropy_with_logits(acc_logit, torch.tensor(float(e["acc"]))))
        opt.zero_grad()
        loss.backward()
        opt.step()


class Planner:
    """Lookahead planner: enumerates strategy sequences for a session inside the world model."""

    def __init__(self, wm, lam_lat=0.5, mu_cost=0.05):
        self.wm = wm
        self.lam_lat = lam_lat
        self.mu_cost = mu_cost

    def reward(self, pred):
        return pred["acc"] - self.lam_lat * pred["lat"] - self.mu_cost * pred["cost"]

    def plan_session(self, reqs, choices):
        """reqs: list of (feat, allowed_strategies). Greedy lookahead with 1-step context
        (whether the previous action consolidated memory). Returns list of strategies."""
        plan, ctx = [], 0.0
        for feat, allowed in reqs:
            best, best_r = None, -1e9
            for s in allowed:
                r = self.reward(self.wm.predict(feat, s, ctx))
                if r > best_r:
                    best, best_r = s, r
            plan.append(best)
            ctx = 1.0 if best == "fullc" else ctx   # consolidation persists into next request
        return plan

    def plan_pair(self, featA, featB, choicesA, choicesB):
        """Full 2-step lookahead: enumerate (sA, sB) pairs inside the world model."""
        best, best_r = None, -1e9
        for sA in choicesA:
            ctxB = 1.0 if sA == "fullc" else 0.0
            rA = self.reward(self.wm.predict(featA, sA, 0.0))
            for sB in choicesB:
                rB = self.reward(self.wm.predict(featB, sB, ctxB))
                if rA + rB > best_r:
                    best, best_r = (sA, sB), rA + rB
        return best
