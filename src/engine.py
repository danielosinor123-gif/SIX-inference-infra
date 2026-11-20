"""engine.py — the ARIADNE runtime: closes the loop
   SENSE -> LOOM -> SHUTTLE/ORACLE -> emit -> reward -> LOOM.update, with THREAD memory.

Modes for benchmarking:
  baseline="full"      : full depth, greedy, no verify          (vanilla serving)
  baseline="earlyexit" : entropy-threshold early exit (CALM-style), greedy
  baseline="ariadne"   : LOOM policy + SHUTTLE + ORACLE + THREAD (the full system)
"""
import torch
from . import sense
from .tasks import encode, decode, STOI


class AriadneEngine:
    def __init__(self, target, draft, loom, thread, oracle, shuttle):
        self.target = target
        self.draft = draft
        self.loom = loom
        self.thread = thread
        self.oracle = oracle
        self.shuttle = shuttle
        self.eos = STOI["\n"]

    # ------------------------------------------------------------------ helpers
    def _feat(self, logits, probs, prev_h, hidden, b_left, b_total, step, max_steps):
        return sense.features(logits, probs, prev_h, hidden,
                              b_left, b_total, step, max_steps)

    # ------------------------------------------------------------------ generate
    def generate(self, prompt, budget=60.0, max_steps=12, mode="ariadne",
                 temperature=0.0, training=False, earlyexit_thr=0.25, force_retrieve=False):
        """Returns dict(text, tokens, cost, layer_tokens, rollbacks, accept_rate, reward)."""
        tgt = self.target
        ng = torch.no_grad()
        with ng:
            tgt.reset()
            prompt_tokens = encode(prompt)
            logits = tgt.prefill(prompt_tokens)[-1]           # predicts first gen token
            probs = tgt.last_probs
            hidden = tgt.last_x

        text = prompt
        committed = list(prompt_tokens)
        gen_ckpt = tgt.checkpoint()     # full-depth checkpoint right after prefill (ORACLE base)
        prev_h = None
        b_left = budget
        cost = 0.0
        layer_tokens = 0
        traj_logp = torch.zeros(())
        n_steps = 0
        rb0 = self.oracle.stats["rollbacks"]

        # step-0 action: LOOM may decide to RETRIEVE from THREAD before committing to prefill
        with torch.no_grad() if not training else torch.enable_grad():
            feat = self._feat(logits, probs, None, hidden, b_left, budget, 0, max_steps)
            if mode == "ariadne":
                action, logp, c = self.loom.act(feat, b_left, budget, training=training)
                if training:
                    traj_logp = traj_logp + logp
            else:
                action, c = dict(depth=tgt.L, spec=0, retrieve=0, verify=0), 0.0
        if (force_retrieve or (mode == "ariadne" and action["retrieve"])) and len(self.thread) > 0:
            mem = self.thread.memory_prefix(prompt)
            if mem:
                with ng:
                    tgt.reset()
                    prompt_tokens = encode(mem + prompt)
                    logits = tgt.prefill(prompt_tokens)[-1]
                    probs = tgt.last_probs
                    hidden = tgt.last_x
                text = mem + prompt
                committed = list(prompt_tokens)
                gen_ckpt = tgt.checkpoint()   # re-base after memory-augmented prefill
                cost += self.loom.costs["retrieve"]
                b_left -= cost

        while n_steps < max_steps:
            n_steps += 1
            with torch.no_grad() if not training else torch.enable_grad():
                feat = self._feat(logits, probs, prev_h, hidden, b_left, budget, n_steps, max_steps)
                if mode == "ariadne":
                    action, logp, c = self.loom.act(feat, b_left, budget, training=training)
                    if training:
                        traj_logp = traj_logp + logp
                elif mode == "earlyexit":
                    h_norm = feat[0]
                    action = dict(depth=tgt.L if h_norm > earlyexit_thr else 1,
                                  spec=0, retrieve=0, verify=0)
                    c = self.loom.cost(action["depth"], 0, 0, 0)
                elif mode == "spec4":
                    action = dict(depth=tgt.L, spec=4, retrieve=0, verify=0)
                    c = self.loom.cost(action["depth"], 0, 0, 4)
                elif mode == "spec4v":
                    action = dict(depth=tgt.L, spec=4, retrieve=0, verify=1)
                    c = self.loom.cost(action["depth"], 0, 1, 4)
                elif mode == "depth1":
                    action = dict(depth=1, spec=0, retrieve=0, verify=0)
                    c = self.loom.cost(1, 0, 0, 0)
                elif mode == "depth1v":
                    action = dict(depth=1, spec=0, retrieve=0, verify=1)
                    c = self.loom.cost(1, 0, 1, 0)
                elif mode == "fullc":                      # full depth + consolidate to memory
                    action = dict(depth=tgt.L, spec=0, retrieve=0, verify=0)
                    c = self.loom.cost(tgt.L, 0, 0, 0)
                else:
                    action = dict(depth=tgt.L, spec=0, retrieve=0, verify=0)
                    c = self.loom.cost(action["depth"], 0, 0, 0)

            if action["spec"] > 0:
                # ---- SHUTTLE: transactional speculative block ----
                with ng:
                    draft_toks = self.shuttle.draft_from(committed, action["spec"])
                    acc, bonus, n_comm, logits = self.shuttle.verify_and_commit(draft_toks, logits)
                    new_toks = draft_toks[:acc] + ([bonus] if bonus is not None else [])
                    layer_tokens += len(draft_toks)          # one full pass over the block
                    probs = tgt.last_probs
                    hidden = tgt.last_x
                committed += new_toks
                text += decode(new_toks)
                cost += c
                b_left -= c
                prev_h = hidden
                if new_toks and new_toks[-1] == self.eos:
                    break
                continue

            # ---- greedy / adaptive-depth step ----
            ckpt = tgt.checkpoint()
            with ng:
                tok = tgt.sample(logits, temperature)
                logits = tgt.commit(tok, depth=action["depth"])   # KV transaction for tok
                probs = tgt.last_probs
                hidden = tgt.last_x
                layer_tokens += action["depth"]
            rolled = False
            if action["verify"]:
                with ng:
                    tok, rolled = self.oracle.verify_token(gen_ckpt, tok, text + decode([tok]))
                    if rolled:
                        logits = tgt.commit(tok, depth=tgt.L)
                        probs = tgt.last_probs
                        hidden = tgt.last_x
                        layer_tokens += tgt.L
            committed.append(tok)
            text += decode([tok])
            cost += c
            b_left -= c
            prev_h = hidden
            if tok == self.eos:
                break

        # ---- SLEEP-on-the-spot: consolidate this session into THREAD (mode "fullc") ----
        if mode == "fullc":
            self.thread.consolidate(text)
            cost += 2.0                            # consolidation fee (capital expense)
            b_left -= 2.0

        # ---- reward (verified outcome) + closed-loop update ----
        with ng:
            verdict = self.oracle.check(text)
        if verdict is True:
            reward = 1.0
        elif verdict is False:
            reward = 0.0
        else:  # unverifiable: proxy = confidence of the final token
            reward = 0.5 * (1.0 - sense.entropy_of(logits) / 8.0)
        reward -= 0.05 * cost
        if training and mode == "ariadne":
            self.loom.update(traj_logp, reward, cost, budget)

        return dict(text=text, n_tokens=len(committed) - len(prompt_tokens),
                    cost=cost, layer_tokens=layer_tokens,
                    rollbacks=self.oracle.stats["rollbacks"] - rb0,
                    accept_rate=self.shuttle.acceptance_rate(), reward=reward,
                    cert=verdict)          # proof-carrying output: downstream may trust this
