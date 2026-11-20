"""tiny_lm.py — minimal GPT-style transformer with the primitives ARIADNE needs:

  * per-layer KV cache with checkpoint()/rollback()  -> transactional KV (SHUTTLE/ORACLE)
  * commit(tok, depth) + next_logits(depth)          -> per-token adaptive depth (LOOM),
    probing logits at any depth without double-appending KV
  * verify_block(draft)                              -> block speculative verification
  * forward_seq(tokens)                              -> teacher-forced training forward
"""
import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, K, V):
        T = x.shape[0]
        qkv = self.qkv(x).view(T, 3, self.h, self.dk)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        if K is None:
            K, V = k, v
        else:
            K = torch.cat([K, k], 0)
            V = torch.cat([V, v], 0)
        N = K.shape[0]
        scores = torch.einsum("thd,nhd->htn", q, K) / math.sqrt(self.dk)
        P = N - T
        mask = torch.triu(torch.full((T, N), float("-inf")), diagonal=P + 1)
        probs = (scores + mask.unsqueeze(0)).softmax(-1)
        out = torch.einsum("htn,nhd->thd", probs, V).reshape(T, -1)
        return self.proj(out), K, V, probs

    def forward_batch(self, x):
        """x: (B, T, d) — full causal attention within each row, no KV cache (training)."""
        B, T = x.shape[0], x.shape[1]
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk)
        q, k, v = (qkv[:, :, i].permute(0, 2, 1, 3) for i in range(3))  # (B,h,T,dk)
        scores = torch.einsum("bhtd,bhnd->bhtn", q, k) / math.sqrt(self.dk)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).view(1, 1, T, T)
        probs = (scores + mask).softmax(-1)
        out = torch.einsum("bhtn,bhnd->bhtd", probs, v).permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.proj(out), probs


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, K, V):
        a, K, V, probs = self.attn(self.ln1(x), K, V)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, K, V, probs

    def forward_batch(self, x):
        a, _ = self.attn.forward_batch(self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d=64, h=4, L=4, ff=128, max_seq=96):
        super().__init__()
        self.vocab, self.L, self.max_seq = vocab, L, max_seq
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)
        self.blocks = nn.ModuleList([Block(d, h, ff) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.reset()

    # ---- transactional state ----------------------------------------------------
    def reset(self):
        self.K = [None] * self.L
        self.V = [None] * self.L
        self.n = 0                      # number of committed tokens
        self.last_x = None              # hidden of last committed position after last run layer
        self.last_h = {}                # layer idx -> hidden of last position
        self.last_probs = None

    def checkpoint(self):
        """O(L) snapshot: KV lengths + last-position probe state."""
        return ([None if k is None else k.shape[0] for k in self.K],
                self.last_x, dict(self.last_h), self.last_probs, self.n)

    def rollback(self, ckpt):
        lengths, last_x, last_h, last_probs, n = ckpt
        for l in range(self.L):
            c = lengths[l]
            if c is None:
                self.K[l] = self.V[l] = None
            elif self.K[l] is not None and self.K[l].shape[0] > c:
                self.K[l] = self.K[l][:c]
                self.V[l] = self.V[l][:c]
        self.last_x, self.last_h, self.last_probs, self.n = last_x, last_h, last_probs, n

    # ---- running tokens -----------------------------------------------------------
    def _run(self, tokens, active_layers=None):
        """Append tokens through active layers (KV written for those layers).
        Updates last_x/last_h/last_probs for the final position. Returns final logits."""
        T = len(tokens)
        idx = torch.arange(self.n, self.n + T)
        x = self.emb(torch.tensor(tokens, dtype=torch.long)) + self.pos(idx)
        probs = None
        last_h = {}
        for l, blk in enumerate(self.blocks):
            if active_layers is not None and l not in active_layers:
                continue
            x, self.K[l], self.V[l], probs = blk(x, self.K[l], self.V[l])
            last_h[l] = x[-1]
        self.n += T
        self.last_x = x[-1]
        self.last_h = last_h
        self.last_probs = probs
        return self.head(self.lnf(x))          # (T, V)

    def prefill(self, tokens):
        """Full-depth prefill; returns logits for the last prompt position (T,V) — caller
        uses [-1] to predict the first generated token."""
        return self._run(list(tokens), None)

    def commit(self, tok, depth=None):
        """Commit one generated token, running only layers 0..depth-1 (adaptive depth).
        Returns logits after this token."""
        active = None if depth is None else set(range(depth))
        return self._run([tok], active)[-1]

    def next_logits(self, depth=None):
        """Logits for the NEXT token at the given depth, probing the last committed position
        through more layers WITHOUT writing KV. Returns (logits, attn_probs_or_None)."""
        depth = depth or self.L
        if depth <= len(self.last_h):
            x = self.last_h[depth - 1]
            probs = self.last_probs if depth == len(self.last_h) else None
        else:
            with torch.no_grad():
                xx = self.last_x.view(1, -1)
                probs = None
                for l in range(len(self.last_h), depth):
                    xx, _, _, probs = self.blocks[l](xx, self.K[l], self.V[l])
                    self.last_h[l] = xx[0]
                x = xx[0]
                self.last_x = x
        return self.head(self.lnf(x)), probs

    # ---- SHUTTLE block verification ------------------------------------------------
    def verify_block(self, draft_tokens, pre_logits):
        """Run all layers over the drafted block in ONE pass. Returns (accepted, bonus_logits,
        block_logits, ckpt). KV now includes the whole block; settle with commit_prefix.

        NOTE: block_logits[i] predicts the token AFTER draft[i], so draft[i] must be compared
        against the prediction of the *previous* position: pre_logits for i=0, block_logits[i-1]
        for i>=1."""
        ckpt = self.checkpoint()
        block_logits = self._run(list(draft_tokens), None)
        preds = [pre_logits] + [block_logits[i] for i in range(len(draft_tokens) - 1)]
        argmax = [int(p.argmax(-1)) for p in preds]
        acc = 0
        for i, dt in enumerate(draft_tokens):
            if argmax[i] == dt:
                acc += 1
            else:
                break
        bonus = preds[acc] if acc < len(draft_tokens) else None
        return acc, bonus, block_logits, ckpt

    def commit_prefix(self, ckpt, keep, last_token=None):
        """Keep only `keep` drafted tokens of the KV written by verify_block (pointer commit,
        no recompute), then rebuild the probe state (last_x/last_h) for the new last position."""
        lengths, last_x, last_h, last_probs, n = ckpt
        for l in range(self.L):
            c = lengths[l] or 0
            if self.K[l] is not None:
                self.K[l] = self.K[l][: c + keep]
                self.V[l] = self.V[l][: c + keep]
        self.n = n + keep
        if keep == 0:
            self.last_x, self.last_h, self.last_probs = last_x, last_h, last_probs
            return
        with torch.no_grad():
            x = self.emb(torch.tensor([last_token])) + self.pos(torch.tensor([self.n - 1]))
            lh, probs = {}, None
            for l, blk in enumerate(self.blocks):
                K = self.K[l][:-1] if (self.K[l] is not None and self.K[l].shape[0] > 0) else None
                V = self.V[l][:-1] if (self.V[l] is not None and self.V[l].shape[0] > 0) else None
                x, _, _, probs = blk(x, K, V)
                lh[l] = x[-1]
            self.last_x, self.last_h, self.last_probs = x[-1], lh, probs

    # ---- training forward -----------------------------------------------------------
    def forward_seq(self, tokens):
        """Teacher-forced forward over a full sequence with LOCAL kv (training only).
        Returns logits (T, V)."""
        T = len(tokens)
        x = self.emb(torch.tensor(tokens, dtype=torch.long)) + self.pos(torch.arange(T))
        Ks, Vs = [None] * self.L, [None] * self.L
        for l, blk in enumerate(self.blocks):
            x, Ks[l], Vs[l], _ = blk(x, Ks[l], Vs[l])
        return self.head(self.lnf(x))

    def forward_batch(self, x):
        """Batched teacher-forced forward. x: (B, T) token ids -> logits (B, T, V)."""
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T)).unsqueeze(0)
        for blk in self.blocks:
            h = blk.forward_batch(h)
        return self.head(self.lnf(h))

    @staticmethod
    def sample(logits, temperature=0.0):
        if temperature <= 0:
            return int(logits.argmax(-1))
        return int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
