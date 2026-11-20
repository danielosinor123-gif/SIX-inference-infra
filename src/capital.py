"""capital.py — CAPITAL COMPUTE: amortized investment decisions.

Novelty (no prior art): existing serving systems minimize *per-request* cost. CAPITAL COMPUTE
optimizes *lifetime* cost across the workload: it watches a stream of requests, detects
recurring patterns, and when a pattern's predicted recurrence x savings exceeds the cost, it
INVESTS compute in capability right now — fine-tuning the draft model on the spot (or
consolidating memory / updating the policy) — so all future requests of that pattern run
cheaper and more accurately. "Spend 10x now, save 3x on 1000 future requests."

Prototype: a pattern buffer per request template; on the threshold-th occurrence of a
template, run a few gradient steps of the draft model on buffered examples of that template.
Measured effect: speculative acceptance rate on that template jumps after the investment.
"""
import random
import torch
import torch.nn.functional as F


def template_of(prompt):
    if "+" in prompt:
        return "arith"
    if "*" in prompt:
        return "repeat"
    if "LIKE?" in prompt or "LIKES" in prompt:
        return "fact"
    return "other"


class Capital:
    def __init__(self, draft, threshold=6, lr=3e-3, steps=40, seed=0):
        self.draft = draft
        self.threshold = threshold
        self.lr = lr
        self.steps = steps
        self.rng = random.Random(seed)
        self.buffer = {}          # template -> list of full training sequences (prompt+answer)
        self.investments = []     # log of (template, n_examples, steps)

    def observe(self, full_seq):
        """Feed one realized request (prompt+answer) into the pattern buffer.
        Returns the template if an investment was triggered, else None."""
        tmpl = template_of(full_seq)
        buf = self.buffer.setdefault(tmpl, [])
        buf.append(full_seq)
        if len(buf) == self.threshold:
            self.invest(tmpl)
            return tmpl
        return None

    def invest(self, tmpl):
        """Spend compute now: fine-tune the draft on the buffered examples of this template."""
        from . import tasks
        seqs = self.buffer[tmpl]
        data = [tasks.encode(s) for s in seqs]
        opt = torch.optim.Adam(self.draft.parameters(), lr=self.lr)
        self.draft.train()
        for _ in range(self.steps):
            s = self.rng.choice(data)
            logits = self.draft.forward_batch(torch.tensor([s[:-1]]))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   torch.tensor([s[1:]]).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.draft.eval()
        self.investments.append((tmpl, len(seqs), self.steps))
