"""oracle.py — ORACLE: verification with KV-checkpoint rollback and budget escalation.

The verifier here is a deterministic checker standing in for a tool/PRM: for arith queries it
evaluates a+b mod 10; for fact queries it looks up THREAD memory. On failure it rolls the KV
back to the checkpoint taken before the suspect token and re-decodes that position with a
higher budget profile (full depth + self-consistency majority vote over m samples).
"""
import torch
from collections import Counter
from .tasks import decode


class Oracle:
    def __init__(self, target, memory=None, m_vote=5, escalate_depth=None):
        self.target = target
        self.memory = memory
        self.m_vote = m_vote
        self.escalate_depth = escalate_depth   # None -> full depth
        self.stats = dict(checks=0, fails=0, rollbacks=0, fixed=0)

    def check(self, text_so_far):
        """Return True (pass), False (fail), or None (cannot verify).
        Verifies the first generated char after the last '?' (arith) or 'LIKE?' (fact)."""
        from .tasks import parse_arith, parse_fact_query
        self.stats["checks"] += 1
        if "LIKE?" in text_so_far:
            q, _, tail = text_so_far.rpartition("LIKE?")
            if tail:
                name = parse_fact_query(q + "LIKE?")
                if name and self.memory is not None:
                    expected = self.memory.lookup_fact(name)
                    if expected is not None:
                        return tail[0] == expected
            return None
        if "?" in text_so_far:
            head, _, tail = text_so_far.rpartition("?")
            if tail:                                  # first generated char after '?'
                ab = parse_arith(head + "?")
                if ab is not None:
                    a, b = ab
                    return tail[0] == str((a + b) % 10)   # wrong char OR premature EOS -> False
        return None  # no verifiable claim

    @torch.no_grad()
    def verify_token(self, gen_ckpt, sampled_token, text_so_far):
        """Verify the just-committed token. On failure: rollback to the last FULL-DEPTH
        checkpoint (gen_ckpt, taken right after prefill), recompute the next-token logits at
        full depth (budget escalation), and return the best token (greedy if it verifies,
        else self-consistency majority vote). Returns (final_token, rolled_back)."""
        verdict = self.check(text_so_far)
        if verdict is False:
            self.stats["fails"] += 1
            self.stats["rollbacks"] += 1
            self.target.rollback(gen_ckpt)
            logits2, _ = self.target.next_logits(depth=self.escalate_depth or self.target.L)
            cand = int(logits2.argmax(-1))
            if self.check(text_so_far[:-1] + decode([cand])) is True:
                final = cand
            else:
                votes = Counter()
                for _ in range(self.m_vote):
                    votes[self.target.sample(logits2, temperature=0.8)] += 1
                final = votes.most_common(1)[0][0]
            if self.check(text_so_far[:-1] + decode([final])) is True:
                self.stats["fixed"] += 1
            return final, True
        return sampled_token, False
