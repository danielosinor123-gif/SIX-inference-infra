"""shuttle.py — SHUTTLE: transactional block speculative decoding.

The draft model proposes a block of w tokens; the target verifies them in ONE forward pass
(block-causal). We accept the longest prefix matching the target's argmax (variable-length
acceptance), take the target's token at the first mismatch as a bonus, and settle the KV
transactionally via checkpoint/commit_prefix/rollback — exactly the SHUTTLE protocol in
docs/DESIGN.md §3. In this prototype the block carries tokens only; the design extends the
same transaction to memory writes and sandboxed tool calls.
"""
import torch


class Shuttle:
    def __init__(self, target, draft):
        self.target = target
        self.draft = draft
        self.stats = dict(blocks=0, drafted=0, accepted=0, bonus=0, rollbacks=0)
        self._synced = 0          # draft KV is synced to committed[:_synced]
        self.draft.reset()
        self._synced = 0

    @torch.no_grad()
    def sync(self, committed):
        """Incrementally append newly committed tokens to the draft's KV (O(new tokens))."""
        while self._synced < len(committed):
            self.draft.commit(committed[self._synced])
            self._synced += 1
        if self._synced > len(committed):     # target rolled back -> roll draft back too
            self.draft.reset()
            self._synced = 0
            self.sync(committed)

    @torch.no_grad()
    def draft_from(self, committed, width, temperature=0.0):
        """Draft `width` tokens from the draft model, whose KV is kept in sync with the
        target's committed tokens via sync(). Drafted tokens are rolled back afterwards."""
        self.sync(committed)
        ck = self.draft.checkpoint()
        out = []
        logits, _ = self.draft.next_logits()          # logits after the last committed token
        for _ in range(width):
            tok = self.draft.sample(logits, temperature)
            out.append(tok)
            logits = self.draft.commit(tok)           # append drafted tok, get next logits
        self.draft.rollback(ck)          # drafted tokens are speculative: discard from draft KV
        self._synced = ck[4]             # restore sync counter to pre-draft committed length
        return out

    @torch.no_grad()
    def verify_and_commit(self, draft_tokens, pre_logits):
        """Verify a drafted block against the target. Commits accepted prefix + bonus token.
        Returns (accepted, bonus_token or None, n_committed, next_logits)."""
        t = self.target
        acc, bonus_logits, logits, ckpt = t.verify_block(draft_tokens, pre_logits)
        # settle transaction: keep exactly `acc` drafted tokens of the KV we just wrote
        last_tok = draft_tokens[acc - 1] if acc > 0 else None
        t.commit_prefix(ckpt, acc, last_tok)
        bonus = None
        if acc < len(draft_tokens):
            bonus = int(bonus_logits.argmax(-1))
            next_logits = t.commit(bonus)             # commit bonus token at full depth
            self.stats["bonus"] += 1
        else:
            next_logits = logits[-1]                  # predicts the token after the block
        n_committed = acc + (1 if bonus is not None else 0)
        self.stats["blocks"] += 1
        self.stats["drafted"] += len(draft_tokens)
        self.stats["accepted"] += acc
        return acc, bonus, n_committed, next_logits

    def acceptance_rate(self):
        return self.stats["accepted"] / max(1, self.stats["drafted"])
