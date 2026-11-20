# ARIADNE — Adaptive Reasoning & Inference Architecture with Dynamic Neuro-Execution

**A closed-loop cognitive inference runtime.**
ARIADNE treats inference not as "run the model" but as a **control problem**: for every token,
decide *how much and what kind* of compute to spend, execute that decision *transactionally*,
*verify* it with the option to roll back, and *learn* from the verified outcome — all under a
single per-request budget, with an episodic memory that consolidates between sessions.

> One-line pitch: *the model is the plant, the runtime is the controller, and the verified
> outcome is the reward signal — inference becomes a closed-loop, self-improving control system.*

---

## 0. Why this is novel (summary; full mapping in `research/LANDSCAPE.md`)

Every individual trick exists in 2026 (speculative decoding, MoD/CALM early exit, TriRoute-style
joint routing, sleep consolidation, transactional tool use, anytime certificates). What does **not**
exist is the **closed loop**: perceive → allocate → act (transactionally) → verify (with rollback)
→ consolidate → learn, unified by **one scalar budget (COGS)** allocated across **all** heterogeneous
resources per token, with the policy **trained online from the system's own verified outcomes**.

On top of the closed loop, three mechanisms were added that have **no prior-art counterpart at
all** (not integration — new mechanism classes), each with measured emergent behavior (§7,
benchmarks B6–B8 in `experiments/results.md`):
**SIMULACRUM** (learned world-model + lookahead planning over strategy *sequences* before any
compute is spent), **CAPITAL COMPUTE** (amortized lifetime-cost investment in capability), and
**proof-carrying outputs with verification inheritance** (downstream stages consume the certificate
instead of re-verifying).

---

## 1. Architecture

```
                        ┌──────────────────────── ARIADNE RUNTIME ────────────────────────┐
  request ──► MUX ──► │  SENSE ──► LOOM ──► SHUTTLE ──► ORACLE ──► emit token           │
  (prompt, budget B,   │     ▲        │         │ (tx)      │ (rollback)   │              │
   SLO, quality floor) │     │        ▼         ▼           ▼              │              │
                       │  outcome ──► policy update ◄── reward (verified) ◄───┘              │
                       │     │                                                            │
                       │     ▼                                                            │
                       │  THREAD ◄──── SLEEP (consolidation, utility-predicted) ◄───────────┘
                       └──────────────────────────────────────────────────────────────────────┘
```

### Pillars

| Pillar | Role | Novel mechanism |
|---|---|---|
| **SENSE** | Per-token metacognitive signals from internal state | Cheap analytic probe (entropy, margin, attention entropy, hidden drift) — no extra model |
| **LOOM** | Allocates one scalar **budget (COGS)** per token across {depth, expert top-k, retrieval, verification, speculation width, precision, tool pre-exec} | **Unified heterogeneous action space + dual price learning + online REINFORCE from verified reward** (TriRoute is static, 3 axes; MoD is 1 axis) |
| **SHUTTLE** | Executes a *block* of heterogeneous speculative actions (tokens + memory writes + tool calls) | **Atomic commit/rollback of KV pointers, memory tx-log, and tool sandbox together**; variable-length acceptance with partial-credit re-draft (Atomix/Cordon: tools only; tx-KV: pointers only) |
| **ORACLE** | Verifier (tool / critic / consistency) with **KV-checkpoint rollback and learned budget escalation** | Token-local rollback + majority-vote escalation policy learned online (Token-Guard regenerates in place; CITE certifies but doesn't roll back) |
| **THREAD** | Episodic memory store shared across sessions | Retrieval timed by LOOM from uncertainty, not fixed |
| **SLEEP** | Between sessions: consolidate trajectories | **Utility-predicted consolidation** (what will actually be reused, judged from realized outcomes) into (a) deduplicated memory, (b) a cognitive prefix, (c) distillation data for draft + LOOM (SleepGate/TIDE each do one of these) |
| **MUX** | Per-request scheduler | Picks strategy profile from a **learned cost model continuously updated by realized outcomes**; enforces the anytime quality contract |
| **SIMULACRUM** | Session-level planner above MUX | **Learned world-model of the engine itself + enumeration of strategy sequences over multi-request sessions inside the simulator, executed only after planning** — no scheduler in the literature plans ahead; all are reactive/myopic (§7.1) |
| **CAPITAL** | Investment layer above LOOM | **Amortized lifetime-cost optimization**: detects recurring request patterns and, past a recurrence threshold, *invests* compute in capability (on-the-spot draft fine-tune, consolidation, policy update) — "spend 10× now, save 3× on 1000 future requests". All prior systems minimize per-request cost (§7.2) |

---

## 2. The cognitive budget (COGS) and the LOOM allocator

### 2.1 Currency
Each request gets a budget `B` of COGS (≈ FLOPs/latency/energy, normalized). Every action
consumes `c(a)`:

```
c(depth=k)      = k · c_layer                # run first k of L layers (early-exit style, CALM/MoD axis)
c(top-k=m)      = (m/k_max) · c_expert        # MoE width axis
c(retrieve)     = c_ret                       # inject top-1 memory entry before prefill
c(verify)       = c_ver                       # run ORACLE on this token
c(speculate=w)  = w · c_draft                 # SHUTTLE block of width w
c(precision=p)  = (p_ref/p) · c_layer         # per-layer precision axis (AutoEMatic-style)
c(tool)         = c_tool                      # speculative tool pre-execution (PASTE-style)
```

### 2.2 State and policy
Per decode step `t`, SENSE produces features:

```
s_t = [ H(p_t)/log V,  margin_t,  H(attn_last_t),  ||h_t − h_{t−1}||,  B_left/B,  t/T_max ]
       └ next-token entropy ┘  top1−top2   mean attention entropy    hidden drift     budget frac   progress
```

LOOM is a small policy `π_θ(a | s_t, B_left)` — a per-token categorical over a **joint** action
`a_t = (k_t, m_t, r_t, v_t, w_t, p_t, tool_t)`. In the prototype we factorize:
`π = π_depth · π_retrieve · π_verify · π_spec` (independent heads; the design allows a joint head).

### 2.3 Learning the allocation online (the closed loop)
Reward for a whole request (credit-assigned to all token actions, REINFORCE with baseline):

```
R = 1[verifier accepts final output]  (+ task reward)
    − λ · (COGS spent)                − μ · (latency)      − ν · (energy)
```

Policy update after each request:

```
adv = R − b̄            # b̄ = EMA baseline
θ ← θ + η · adv · Σ_t ∇_θ log π_θ(a_t | s_t)      (+ small entropy bonus for exploration)
```

**Dual price learning** for the budget: maintain a Lagrange multiplier `λ_B` on budget usage;
update `λ_B ← λ_B + η_λ (spent − B)` each request. `λ_B` enters the features as the *price of
compute*, so the policy learns to spend where marginal accuracy gain per COG is highest — an
internal **compute market** (no prior work has an intra-model per-token compute market; external
inference auctions price *service priority*, not per-token compute).

### 2.4 Anytime quality contract
The runtime maintains a monotone running estimate `q̂` of output quality (certified margin from
ORACLE/self-consistency, CITE-style). The caller may interrupt at any time and keep a prefix
whose certified quality is `≥ q_floor` (the request's quality floor). Interrupting **refunds**
the unspent budget to the system-level pool. This makes every request *anytime-valid* and couples
stopping to the budget for the first time.

---

## 3. SHUTTLE — transactional speculative execution

Standard speculative decoding verifies *tokens* only. SHUTTLE speculates a **block of
heterogeneous actions** and commits them **atomically**:

```
block = [tok_1..tok_w,  mem_write?,  tool_call?]
```

Protocol (per block):
1. **Checkpoint**: snapshot KV pointer lengths `ckpt`, memory tx-log length, tool sandbox state.
2. **Draft**: small draft model proposes `w` tokens; the controller also proposes likely
   memory writes / tool calls (learned n-gram or small predictor over session patterns).
3. **Verify**: target model scores all `w` tokens in **one** forward pass (block-causal mask).
   Accept the longest prefix where `argmax target == draft` (variable-length acceptance);
   the first mismatch position gets the target's own token as a **bonus token**.
4. **Side-effect validation**: tool calls execute in a **sandbox**; memory writes go to a tx-log.
5. **Commit or rollback atomically**:
   - accept → commit KV pointers (`KV ← KV[:ckpt+w']`), commit sandbox effects, commit tx-log;
   - reject beyond prefix → rollback KV to `ckpt`, discard sandboxed effects, drop tx entries.
6. **Partial-credit re-draft**: the rejected suffix is re-drafted with width `w' < w`
   (cost-aware), rather than restarting from `w`.

Correctness invariant: *after any block, committed state is exactly the state the target model
would have produced autoregressively* (speculative decoding's exactness property, extended to
side effects via the sandbox).

---

## 4. ORACLE — verification with rollback and budget escalation

For each token flagged by LOOM (`v_t=1`) or by a risk signal (high entropy at a factual position):

1. Run the verifier `V` (tool result, memory lookup, deterministic checker, or self-consistency
   across `m` samples) on the segment ending at token `t`.
2. If **pass** → commit, continue.
3. If **fail** → **rollback** KV to the last checkpoint (start of the suspect segment), then
   **escalate**: re-decode the segment with a higher budget profile
   `(depth=L, top-k=max, self-consistency m=5, majority vote)`; the escalation policy
   (how much extra budget per failure count) is itself learned from outcomes.
4. Update LOOM: `verify` actions that caught errors get positive advantage; wasted verifies get
   negative — the policy learns *where verification pays*.

---

## 5. THREAD / SLEEP — episodic memory with utility-predicted consolidation

**THREAD store**: entries `e = (key, value, text, utility u_e, hits, created, decay)`. Retrieval
at request start if LOOM's `retrieve` action fires: score `s(e) = sim(query, e) · u_e`, inject
top-k entries as a memory prefix.

**SLEEP** (between sessions / idle):
1. Scan the day's trajectories for (a) fact-like segments, (b) verified solutions, (c) recurring
   tool patterns, (d) LOOM actions that paid off.
2. **Utility prediction**: a small learned predictor `û(e)` estimates P(entry will be retrieved
   and useful in the next horizon), from features (recency, frequency, verifier corrections that
   used it, LOOM retrieve actions that hit it). Consolidate only if `û(e) ≥ τ` — *selective*
   consolidation driven by realized outcomes.
3. Consolidate into: (a) deduplicated memory entries (semantic dedup, usage-weighted decay —
   Ebbinghaus-style forgetting), (b) a compact **cognitive prefix** (memory tokens), (c) a
   distillation buffer that retrains the **draft model and the LOOM policy** — the system gets
   measurably faster and more accurate on recurring workloads over time.

---

## 6. MUX — per-request scheduler with a learned cost model

Each request carries `(budget B, latency SLO, quality floor q)`. MUX keeps per-strategy
`(latency EMA, accuracy EMA, COGS EMA)` for strategies `{GREEDY, SPEC_w, SPEC+ORACLE, DEEP}` and
picks the cheapest strategy whose predicted latency ≤ SLO and predicted accuracy ≥ q, under
budget B. The EMAs are updated from **realized** outcomes, so the engine learns its own latency
model (unlike Cascade/AdaSpec which react to fixed heuristics).

---

## 7. Three mechanisms with no prior-art counterpart

The v1 pillars were novel mainly as *integration*. These three are novel as **mechanisms** — each
has no row in the `research/LANDSCAPE.md` table, and each produces **emergent behavior that none of
its components has on its own** (measured in B6–B8, `experiments/results.md`).

### 7.1 SIMULACRUM — a learned world-model of the engine, used for lookahead planning *before* compute is spent (`src/simulacrum.py`)

Every existing scheduler (Cascade, AdaSpec, VelServe, Sarathi-Serve, Autopoiesis, MUX-EMA) is
**reactive and myopic**: it chooses a strategy for the *current* request from realized or heuristic
statistics. SIMULACRUM instead:

1. Trains a small **world model** `W(request-features, strategy, context) → (latency, accuracy, cost)`
   **online from realized outcomes** of the engine itself;
2. A **Planner** enumerates **strategy sequences over a whole multi-request session inside W**
   (e.g. 2-step lookahead `plan_pair`) and scores them with a reward
   `r = acc − λ·lat − μ·cost`;
3. Only the winning plan is executed on the real engine.

**Emergent property (B6):** the planner pays a consolidation fee on turn A *because W predicts turn
B's accuracy jumps when memory was consolidated* — a myopic controller never does this, since on
turn A alone the consolidating strategy looks strictly worse (same accuracy, extra fee).

### 7.2 CAPITAL COMPUTE — amortized investment decisions (`src/capital.py`)

All existing serving systems minimize **per-request** cost. CAPITAL optimizes **lifetime** cost
across the workload:

1. Every request is logged into a per-template **pattern buffer**;
2. When a template's buffer reaches the recurrence threshold, the predicted
   `recurrence × future-savings` exceeds the one-time cost, so the engine **invests compute now**:
   a few gradient steps fine-tuning the **draft model on the spot** on the buffered examples
   (in the general design: consolidation, policy update, tool-schema caching);
3. All future requests of that template then run cheaper *and* more accurately.

**Emergent property (B7):** on a recurring REPEAT stream served by a *cold* draft (never trained on
REPEAT), speculative acceptance is flat/low for the first window and **jumps after the investment
fires** — a per-request cost minimizer never pays the investment, because per-request it is pure cost.

### 7.3 Proof-carrying outputs / verification inheritance (engine `cert` field)

CITE-style certificates exist but are **one-shot**: every consumer re-verifies. ARIADNE's
`generate()` returns `cert` — the ORACLE's verdict on the emitted output — and downstream pipeline
stages **consume the certificate to skip re-verification** (chain of custody of trust). An output
that failed verification and was escalated to full depth carries `cert=True` only after the fix.

**Emergent property (B8):** downstream verify checks drop to (near) zero on certified outputs with
**identical accuracy** — trust is inherited, not recomputed.

---

## 8. Algorithms (pseudocode)

```
function ARIADNE.generate(request):
    B ← request.budget;  s ← SENSE.init()
    if LOOM decides retrieve at step 0 and B allows:
        mem ← THREAD.query(request.prompt);  prompt ← mem_prefix + prompt
    prefill(prompt)                                   # full depth
    ckpt ← checkpoint()
    while not done:
        logits, signals ← decode_step(last_token, depth=k_t)     # k_t from LOOM
        s ← SENSE.update(logits, signals)
        a ← LOOM.policy(s, B_left)                                # joint action
        if a.spec > 0:  SHUTTLE.block(width=a.spec)               # transactional
        if a.verify:    if not ORACLE.check():  ORACLE.rollback_and_escalate(ckpt)
        emit(token);  B_left −= cost(a);  log(trajectory)
    reward ← ORACLE.final(request, output) − λ·spent
    LOOM.update(trajectory, reward)                               # closed loop
    MUX.update(strategy, realized_latency, reward)
    THREAD.log(request, output, reward)
    return output, certified_quality q̂                            # anytime contract

function SLEEP():
    for e in THREAD:  e.u ← decay(e);  keep if û(e) ≥ τ
    distill(draft, LOOM, day_buffer)                              # self-improvement
```

---

## 9. Complexity & safety

- SENSE: O(V) for entropy + O(L·h·N) attention entropy — negligible vs one layer.
- LOOM policy: tiny MLP, O(F·H) per token — <1% of one layer's FLOPs.
- SHUTTLE verification: one forward over `w` tokens costs the same as `w` decode steps; profit
  iff acceptance rate `α` satisfies `α·w > 1 + w·c_draft/c_target` (measured online; LOOM learns
  the optimal `w` per workload).
- ORACLE rollback: O(segment) recompute, bounded by escalation cap.
- Safety: all side effects (tools, memory writes) are sandboxed until commit → speculative
  execution can never corrupt state. Memory entries carry provenance; THREAD never injects
  unverified content into certified outputs.

---

## 10. Prototype scope (`src/`, `experiments/`)

A runnable PyTorch prototype on a tiny LM proving every pillar end-to-end:
- `tiny_lm.py` — transformer with per-layer KV cache, checkpoint/rollback, block verification,
  adaptive-depth decode.
- `sense.py` / `loom.py` — signals + online-trained budget allocator (REINFORCE + dual price).
- `shuttle.py` — transactional block speculative decoding with variable acceptance + rollback.
- `oracle.py` — deterministic verifier + KV rollback + majority-vote escalation.
- `thread.py` — episodic memory, utility-weighted, consolidation ("sleep").
- `engine.py` / `mux.py` — the closed-loop runtime (emits `cert`, supports `fullc`) + SLO scheduler.
- `simulacrum.py` — SIMULACRUM: WorldModel + lookahead Planner (§7.1).
- `capital.py` — CAPITAL COMPUTE: pattern buffer + threshold-triggered draft investment (§7.2).
- `experiments/train.py` — trains target/draft/cold-draft → `ckpt.pt`.
- `experiments/demo.py` — benchmarks B1–B8 vs baselines with measured numbers → `results.md`.
