"""sense.py — SENSE: per-token metacognitive signals from the model's own internal state.
No extra model, no labels: entropy, margin, attention entropy, hidden drift, budget/progress.
"""
import math
import torch
import torch.nn.functional as F


def entropy_of(logits):
    p = F.softmax(logits.float(), -1)
    return float(-(p * (p + 1e-12).log()).sum())


def margin_of(logits):
    top2 = torch.topk(logits.float(), 2).values
    return float(top2[0] - top2[1])


def attn_entropy(probs):
    """probs: (h, T, N) attention probs of the last active layer -> mean entropy at last pos."""
    if probs is None:
        return 0.0
    p = probs[:, -1, :].float()                      # (h, N)
    return float(-(p * (p + 1e-12).log()).sum(-1).mean())


def features(logits, probs, prev_hidden, hidden, budget_left, budget_total, step, max_steps):
    """Feature vector for LOOM. All model-internal or budget signals — no task peeking."""
    V = logits.shape[-1]
    h_norm = entropy_of(logits) / math.log(V)        # normalized next-token entropy
    m = margin_of(logits)
    m_norm = 1.0 / (1.0 + abs(m))                     # high margin -> confident -> small
    ae = attn_entropy(probs) / math.log(max(2, probs.shape[-1] if probs is not None else 2))
    if prev_hidden is not None and hidden is not None:
        drift = float((hidden - prev_hidden).norm() / (prev_hidden.norm() + 1e-9))
    else:
        drift = 0.0
    return [
        h_norm,                                       # 0: uncertainty of next token
        m_norm,                                       # 1: confidence margin
        min(ae, 1.0),                                 # 2: attention entropy
        min(drift, 3.0) / 3.0,                        # 3: hidden drift
        budget_left / max(budget_total, 1),           # 4: budget fraction left
        step / max(max_steps, 1),                     # 5: progress
    ]


FEAT_DIM = 6
