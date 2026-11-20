# Research Landscape: LLM Inference Systems (as of Nov 2026)

Compiled from live literature search, Nov 19 2026. Purpose: establish exactly what already
exists so that the ARIADNE system we design occupies genuine white space.

## 1. What already exists (individual mechanisms)

| Mechanism | Representative 2026 work | Status |
|---|---|---|
| Token/block speculative decoding | EAGLE-3, VSD, ConFu, DBLAST, ForeSpark, self-speculative decoding | Mature |
| Distributed / edge-cloud speculative decoding | DiP-SD, StarSD, NebulaSD, X-CoSD, SPADE, PicoSpec, FlexSpec | Mature |
| Speculative **tool calls** | PASTE, SchemaFill, sPTC, ETA, TomasuLLM | Mature |
| Transactional tool side-effects | Atomix, Cordon (semantic transactions) | Exists for tools only |
| Transactional KV rollback on rejection | "Accelerating CPU Speculative Decoding via Unified Graph Execution and Transactional KV Isolation" | Exists for KV pointers only |
| Joint adaptive routing (attention + experts + KV) | **TriRoute** (arXiv 2607.06601), AlterNet, REINFORCED MoE routing | Exists, but **static** (trained offline, no per-request budget, no verification/retrieval in action space) |
| Per-token early exit | CALM, DEEP THINK WITH CONFIDENCE, ConfLayers, DEER, EAT, LYNX, Mixture-of-Depths, PonderLM-3 | Mature (depth axis only) |
| Per-layer/per-token mixed precision | AutoEMatic (entropy-driven layer precision) | Exists (precision axis only) |
| Attention-predictive KV prefetch | InfiniGen, OasisKV, DualDecoder, AttentionPredictor | Mature |
| Semantic KV reuse beyond exact prefix | RcLLM, Probabilistic Language Tries, cross-model KV sharing ("A Universal Context-Reuse Layer") | Exists |
| Latent inter-agent communication (KV/hidden-state handoff) | LatentMAS, StateBridge, AVP-spec, KV-Lingo, Thought Blocks | Mature |
| Sleep / memory consolidation at inference | Google "Sleep" framework, SleepGate (learned KV sleep), "Let Them Sleep" agents, EngramRAG | Exists, but **not coupled to per-token control loop** |
| Token-level hallucination control | Token-Guard, HaluGate, scalable token-level detection | Exists (regenerate, no KV rollback + budget escalation) |
| SLO-aware serving schedulers | Cascade, AdaSpec, VelServe, Sarathi-Serve | Mature (reactive, request-level) |
| Self-improving serving engines | TIDE (online draft adaptation), Autopoiesis (online policy evolution), Reef, Together Compiler | Exist per-subsystem, **not closed-loop across subsystems** |
| Test-time RL / TTT | TTRL, JitRL, TTSR, Policy of Thoughts, ECHO | Exists (task-level, not serving-control-level) |
| Anytime-valid output certificates | CITE, Martingale Majority Certificate, anytime-valid sufficiency | Exists for self-consistency only |
| Verified-solution KV grafting | "Grafting a Verified Solution Cache" (frozen model + verified KV library) | Exists as offline cache |
| External compute markets / auctions | Inference auctions (priority pricing), compute-market economics papers | **Market-level only** — no per-token intra-model compute market exists |
| Metacognition in LLMs | Surveys (Yale LLM-Metacognition), SRCoT closed-loop retrieve-reflect-revise | Pipeline-level, not a serving runtime |

## 2. The gap (what nobody has built)

Every one of the above is an **open-loop, single-axis, per-request** optimization:

1. No system lets a **single scalar budget** be allocated *per token* across a **heterogeneous
   action space** — depth, expert width, retrieval, verification, speculation width, precision,
   tool pre-execution — chosen from the model's **own** uncertainty signals, and **learned online
   from verified outcomes** (TriRoute is static and covers only 3 axes; MoD/CALM only depth;
   FLARE only retrieval).
2. No system makes speculation **transactional across all state simultaneously** — KV cache,
   memory writes, *and* tool calls — with atomic commit/rollback (Atomix/Cordon cover tools only;
   transactional KV covers pointers only).
3. No system does **verification with KV-checkpoint rollback and learned budget escalation**
   (Token-Guard regenerates in place; self-consistency has no token-local rollback).
4. No system **closes the loop across requests and across time**: today's verified outcomes do not
   retrain the allocation policy, the draft model, and the episodic memory *jointly*
   (TIDE/Autopoiesis/SleepGate each touch one subsystem).
5. No runtime exposes an **anytime quality contract integrated with the budget** (CITE certifies
   self-consistency but is not wired into compute allocation; stopping early refunds nothing).

## 4. Mechanisms with **no** prior art (the novelty core, added after the "integration-only" critique)

The v1 pillars were defensible mainly as a closed-loop *integration*. The following three mechanism
**classes** have no counterpart anywhere in the table above (checked against the same 2026 sources);
each is implemented in `src/` and measured in B6–B8 (`experiments/results.md`):

| Mechanism class | Closest prior art | Why prior art does not cover it | Emergent behavior (measured) |
|---|---|---|---|
| **Learned world-model + lookahead planning over strategy *sequences*** (SIMULACRUM, `src/simulacrum.py`) | Cascade/AdaSpec/VelServe/Sarathi-Serve (reactive SLO schedulers), Autopoiesis (per-request policy evolution) | All are **myopic**: they optimize the current request from realized/heuristic stats. None trains a simulator of the engine and enumerates *multi-request plans* inside it before spending real compute | B6: planner pays a consolidation fee on turn A because the world model predicts turn B becomes accurate; myopic/fixed policies never do |
| **Amortized lifetime-cost investment in capability** (CAPITAL COMPUTE, `src/capital.py`) | TIDE/Autopoiesis (online draft adaptation), distillation-at-serving systems | All trigger adaptation from *per-request* signals (loss, uncertainty) or fixed schedules; none does **amortized investment**: "spend 10× now because this pattern will recur 1000×" | B7: acceptance on a recurring template jumps only *after* the recurrence-threshold investment fires, on a draft that never saw the pattern |
| **Proof-carrying outputs / verification inheritance** (engine `cert`) | CITE, Martingale Majority Certificate, anytime-valid sufficiency | Certificates exist but are **one-shot**: every consumer re-verifies from scratch. No system lets a downstream stage *consume* an upstream certificate to skip re-verification (chain of custody of trust) | B8: downstream verify checks → ~0 on certified outputs at identical accuracy |

## 3. ARIADNE's novelty claim

ARIADNE is a **closed-loop cognitive inference runtime**:

```
SENSE (per-token metacognitive signals)
   → LOOM (unified budgeted allocator over ALL compute resources, learned online)
   → SHUTTLE (transactional speculative execution of heterogeneous blocks, atomic rollback)
   → ORACLE (verifier with KV-checkpoint rollback + learned budget escalation)
   → THREAD (episodic memory) + SLEEP (utility-predicted consolidation)
   → outcome feeds back to retrain LOOM policy, draft, and memory  ← the closed loop
```

unified by one internal currency — the **cognitive budget (COGS)** — with dual price learning,
and governed by a per-request **anytime quality contract**. Each pillar exists in isolation in
prior art (see table); the closed-loop, budget-unified, transactional, self-learning
*integration* is the novel whole, and we prototype every pillar end-to-end in `src/`.

On top of the integration, three mechanism **classes** with no prior-art counterpart are added
and measured (see §4): SIMULACRUM (world-model lookahead planning), CAPITAL COMPUTE (amortized
investment), and verification inheritance (proof-carrying outputs).
