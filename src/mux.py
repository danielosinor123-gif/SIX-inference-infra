"""mux.py — MUX: per-request scheduler with a learned cost model.

Keeps EMA of realized latency / accuracy / cost per strategy and picks, for each request's
(SLO, quality floor, budget), the cheapest strategy predicted to satisfy both — the engine
learns its own latency model from realized outcomes (DESIGN.md §6).
"""
import time


class Mux:
    STRATEGIES = ["greedy", "spec4", "spec4+verify"]

    def __init__(self, ema=0.3):
        self.ema = ema
        self.stats = {s: dict(latency=0.05, accuracy=0.5, cost=10.0, n=0) for s in self.STRATEGIES}

    def pick(self, slo_sec, q_floor, budget):
        """Choose the strategy predicted to meet SLO and quality floor with lowest cost."""
        best, best_cost = "greedy", float("inf")
        for s in self.STRATEGIES:
            st = self.stats[s]
            if st["latency"] <= slo_sec and st["accuracy"] >= q_floor and st["cost"] <= budget:
                if st["cost"] < best_cost:
                    best, best_cost = s, st["cost"]
        return best

    def observe(self, strategy, latency, accuracy, cost):
        st = self.stats[strategy]
        a = self.ema
        st["latency"] = (1 - a) * st["latency"] + a * latency
        st["accuracy"] = (1 - a) * st["accuracy"] + a * accuracy
        st["cost"] = (1 - a) * st["cost"] + a * cost
        st["n"] += 1
