# ARIADNE — a closed-loop cognitive inference runtime

**ARIADNE** (Adaptive Reasoning & Inference Architecture with Dynamic Neuro-Execution) is a
novel inference *system*, not another decoding trick. It treats every token generation as a
control problem: the runtime **senses** the model's own uncertainty, **allocates** a single
per-request budget across *all* compute resources (depth, retrieval, verification,
speculation), executes **transactionally** (atomic commit/rollback), **verifies** with rollback
and budget escalation, **remembers** across sessions, and **learns online** from its own
verified outcomes — closing the loop that every existing 2026 system leaves open.

## Why it's novel

Every individual mechanism already exists (see `research/LANDSCAPE.md` for the full 2026 map:
EAGLE-3/DBLAST speculative decoding, TriRoute joint routing, CALM/MoD early exit, SleepGate
consolidation, Atomix/Cordon transactional tools, Token-Guard hallucination control,
Cascade/AdaSpec SLO schedulers, TIDE/Autopoiesis self-improving engines...). What does **not**
exist is the **closed loop**: perceive → allocate → act transactionally → verify with rollback
→ consolidate → learn, unified by one scalar budget (**COGS**) allocated per token across a
heterogeneous action space by a policy trained **online from verified rewards**, with an
internal compute market (dual price learning) and an anytime quality contract.

On top of the closed loop, three mechanism **classes with no prior-art counterpart** (each mapped
against the 2026 literature in `research/LANDSCAPE.md` §4, implemented in `src/`, measured in
B6–B8 — see `docs/DESIGN.md` §7):

1. **SIMULACRUM** (`src/simulacrum.py`) — a *learned world-model of the engine itself*
   `W(request, strategy, context) → (latency, accuracy, cost)` trained online from realized
   outcomes, plus a planner that enumerates **strategy sequences over a whole multi-request
   session inside W before any real compute is spent**. Every existing scheduler
   (Cascade/AdaSpec/VelServe/Autopoiesis) is reactive and myopic; none plans ahead in a
   simulator. Emergent behavior (B6): the planner pays a consolidation fee on turn A *because
   W predicts turn B becomes accurate and 4× cheaper* — myopic and fixed controllers never do.
2. **CAPITAL COMPUTE** (`src/capital.py`) — amortized **lifetime-cost** optimization: recurring
   request patterns are buffered, and past a recurrence threshold the engine *invests* compute in
   capability right now (on-the-spot fine-tune of the draft on the buffered pattern) so all future
   requests run cheaper. All prior systems minimize per-request cost; none makes "spend 10× now,
   save 3× on 1000 future requests" decisions. Emergent behavior (B7): speculative acceptance on a
   recurring REPEAT stream jumps 0.03 → 0.36 only *after* the investment fires (draft capability
   0.00 → 1.00 on the stream).
3. **Verification inheritance** (engine `cert` field) — outputs are **proof-carrying**: each
   `generate()` returns the ORACLE's certificate, and downstream pipeline stages consume it to
   **skip re-verification** (chain of custody of trust; CITE-style certs are one-shot).
   Emergent behavior (B8): stage-2 re-verifies drop 50 → 3 at *identical* accuracy (0.940).

## The pillars

| Pillar | File | What it does |
|---|---|---|
| SENSE | `src/sense.py` | Per-token metacognitive signals (entropy, margin, attention entropy, hidden drift) |
| LOOM | `src/loom.py` | Unified budgeted allocator; REINFORCE + dual price; action = (depth, retrieve, verify, spec-width) |
| SHUTTLE | `src/shuttle.py` + `TinyLM.verify_block/commit_prefix/rollback` | Transactional block speculative decoding, variable-length acceptance, bonus token, atomic KV settle |
| ORACLE | `src/oracle.py` | Verifier + rollback to full-depth checkpoint + budget escalation (greedy, then majority vote) |
| THREAD/SLEEP | `src/thread.py` | Episodic memory, utility-predicted consolidation with forgetting |
| MUX | `src/mux.py` | SLO-aware scheduler with a learned cost model (EMA of realized latency/accuracy) |
| SIMULACRUM | `src/simulacrum.py` | Learned world model of the engine + lookahead planning over strategy *sequences* before executing (B6) |
| CAPITAL | `src/capital.py` | Recurrence-thresholded investment in capability: on-the-spot draft fine-tune (B7) |
| Runtime | `src/engine.py` | `AriadneEngine.generate()` — the closed loop; returns proof-carrying `cert` (B8) |

Full spec with math, pseudocode, complexity and safety analysis: **`docs/DESIGN.md`**.

## Run it

```bash
python experiments/train.py   # train tiny target + draft + cold-draft LMs (~9 min CPU)
python experiments/demo.py    # online-train LOOM, run benchmarks B1-B8, write results.md
python experiments/demo.py new  # fast mode: only B6-B8 (SIMULACRUM / CAPITAL / certs)
```

Requires only `torch` (CPU is fine).

## Measured results (`experiments/results.md`)

On a tiny LM with synthetic ARITH (hard answer token), REPEAT (long generation) and FACT
(cross-session memory) tasks:

- **LOOM** (online-trained): reward 0.03 → 0.85 while cutting spend 18.3 → 2.5 COGS/request;
  at eval, **3.8× less compute than full-depth at ~equal accuracy** (0.920 vs 0.930; 0.989 on
  seen combos — the gap is held-out combos no method can know).
- **SHUTTLE**: **2.66× fewer target layer-tokens per generated token** (acceptance 0.50).
- **ORACLE**: verify+rollback+escalation recovers **all fixable** cheap-decode errors
  (seen combos 0.989 → **1.000**, fixed 2/2; overall matches full-depth 0.930).
- **THREAD**: cross-session fact QA **0.25 → 1.00** with consolidated memory.
- **MUX**: SLO+accuracy hit-rate beats always-greedy and always-spec in every run
  (e.g. 0.98 vs 0.97 vs 0.58; absolute values are wall-clock noisy on CPU, ordering is stable).
- **SIMULACRUM (B6)**: on a 2-step fact session, the planner that looks ahead inside a learned
  world model picks `(consolidate, retrieve)` → turn-B accuracy **1.000 @ 54 COGS**, while the
  myopic policy (same world model, per-request) picks `(full, retrieve-without-memory)` →
  **0.080 @ 82 COGS**. The world model itself learned `retrieve+memory → acc 1.00, cost 10`
  vs `no memory → acc 0.09, cost 39`: the planner wins on *both* axes because it sees turn 2.
- **CAPITAL COMPUTE (B7)**: serving a recurring REPEAT stream with a cold draft (never trained
  on REPEAT), acceptance is flat **~0.03** for the first 30 requests; the investment fires at
  request 30 and acceptance jumps to **~0.36** (draft REPEAT accuracy 0.00 → 1.00 on the stream)
  — a per-request cost minimizer would never pay that investment.
- **Verification inheritance (B8)**: downstream stage consumes the `cert` instead of re-verifying:
  **3 vs 50** re-verifies at identical accuracy (0.940 = 0.940).

## Layout

```
ariadne/
  README.md               <- this file
  docs/DESIGN.md          <- full architecture spec (math, pseudocode, safety)
  research/LANDSCAPE.md   <- 2026 literature map + gap analysis (novelty proof)
  src/                    <- the runtime (tiny_lm, sense, loom, shuttle, oracle, thread, mux,
                             engine, simulacrum, capital)
  experiments/            <- train.py, demo.py, results.md, ckpt.pt
```
