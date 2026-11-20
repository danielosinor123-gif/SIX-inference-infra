"""thread.py — THREAD: episodic memory store + SLEEP consolidation.

Store entries: {name -> color} facts extracted from session transcripts, with a utility score
(hits + recency), usage-weighted decay between sleeps, and retrieval by token overlap with the
query. Consolidation is utility-predicted: entries below a utility threshold are forgotten
(Ebbinghaus-style), matching DESIGN.md §5.
"""
import re
import time


class Thread:
    def __init__(self, forget_threshold=0.5, decay=0.9):
        self.facts = {}          # name -> dict(color, utility, hits, created)
        self.forget_threshold = forget_threshold
        self.decay = decay
        self.sleeps = 0

    # ---- retrieval ----------------------------------------------------------------
    def lookup_fact(self, name):
        e = self.facts.get(name.strip().upper())
        return e["color"] if e else None

    def query(self, prompt, top_k=2):
        """Score entries by overlap of content tokens with the prompt + utility."""
        toks = set(re.findall(r"[A-Z]+", prompt.upper()))
        scored = []
        for name, e in self.facts.items():
            overlap = 1.0 if name in toks else (0.3 if toks & {name} else 0.0)
            scored.append((overlap * e["utility"], name, e["color"]))
        scored.sort(reverse=True)
        hits = [(n, c) for s, n, c in scored if s > 0][:top_k]
        for n, _ in hits:
            self.facts[n]["hits"] += 1
            self.facts[n]["utility"] += 0.5
        return hits

    def memory_prefix(self, prompt, top_k=2):
        """Inject retrieved facts in the *same format the LM was trained on*."""
        hits = self.query(prompt, top_k)
        if not hits:
            return ""
        return "".join(f"{n} LIKES {c}.\n" for n, c in hits)

    # ---- consolidation (SLEEP) ------------------------------------------------------
    def consolidate(self, session_text):
        """Extract facts from a session transcript; boost utility of mentioned entries."""
        found = re.findall(r"([A-Z]) LIKES ([A-Z])\.", session_text.upper())
        for name, color in found:
            if name in self.facts:
                self.facts[name]["utility"] += 1.0
                if self.facts[name]["color"] != color:      # reconsolidation on conflict
                    self.facts[name]["color"] = color
            else:
                self.facts[name] = dict(color=color, utility=1.5, hits=0,
                                        created=time.time())
        return len(found)

    def sleep(self):
        """Utility-predicted forgetting: decay all, drop those below threshold."""
        self.sleeps += 1
        drop = []
        for name, e in self.facts.items():
            e["utility"] *= self.decay
            e["utility"] += 0.1 * e["hits"]
            e["hits"] = 0
            if e["utility"] < self.forget_threshold:
                drop.append(name)
        for name in drop:
            del self.facts[name]
        return len(drop)

    def __len__(self):
        return len(self.facts)
