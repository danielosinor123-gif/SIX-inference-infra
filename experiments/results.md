# ARIADNE demo results (tiny LM, CPU, torch 2.13.0+cpu)

## LOOM online training (REINFORCE from verified reward, 800 arith requests)
  step 100: mean reward 0.025  avg COGS 18.3
  step 200: mean reward 0.480  avg COGS 8.8
  step 300: mean reward 0.657  avg COGS 4.7
  step 400: mean reward 0.701  avg COGS 4.0
  step 500: mean reward 0.712  avg COGS 3.6
  step 600: mean reward 0.738  avg COGS 3.0
  step 700: mean reward 0.830  avg COGS 2.2
  step 800: mean reward 0.848  avg COGS 2.5
  trained in 31s; final-100 mean reward 0.848

## B1 - ARITH accuracy vs compute (200 test prompts, 10 held-out combos)

| mode | accuracy | layer-tokens/req | COGS/req | rollbacks |
|---|---|---|---|---|
| full-depth (baseline) | 0.930 | 8.0 | 8.0 | 0 |
| entropy early-exit | 0.930 | 3.1 | 3.1 | 0 |
| ARIADNE LOOM | 0.920 | 2.1 | 2.1 | 0 |

ARIADNE on seen combos: acc 0.989; on held-out combos: acc 0.067 (full-depth baseline on held-out: 0.067, rollbacks 0)

## B2 - SHUTTLE speculative decoding on REPEAT (60 prompts, 3-8 char answers)

| mode | wall time (s) | tokens | tokens/s | accuracy | acceptance | target layer-tokens/gen-token |
|---|---|---|---|---|---|---|
| greedy full-depth | 1.61 | 400 | 247.9 | 0.967 | - | 4.00 |
| SHUTTLE spec-4 | 7.26 | 631 | 86.9 | 0.917 | 0.50 | 1.50 |

target-compute reduction: 2.66x fewer target layer-tokens per generated token
(wall-clock on CPU is dominated by per-step Python overhead; on GPU the compute saving
translates directly to latency/energy. Acceptance = fraction of drafted tokens verified.)

## B3 - ORACLE rollback+escalation on all 200 ARITH prompts

| mode | accuracy | layer-tokens/req | rollbacks |
|---|---|---|---|
| full (all 200) | 0.930 | 8.0 | 0 |
| depth1 (all 200) | 0.920 | 2.1 | 0 |
| depth1v (all 200) | 0.930 | 6.1 | 164 |
    ORACLE: checks=1093 fails=164 rollbacks=164 fixed=2
| ariadne (all 200) | 0.920 | 2.1 | 0 |
    ORACLE: checks=200 fails=0 rollbacks=0 fixed=0
| depth1 (seen combos only, 185) | 0.989 | 2.1 | 0 |
| depth1v (seen combos only, 185) | 1.000 | 2.2 | 2 |
    ORACLE: checks=585 fails=2 rollbacks=2 fixed=2

depth1 = always cheapest (1 layer); depth1v = depth1 + ORACLE verify/rollback/escalate;
ARIADNE = LOOM chooses depth per token + ORACLE. ORACLE recovers the fixable depth1 errors.

## B4 - THREAD cross-session memory (8 facts in session 1, asked in session 2)
  sanity in-session QA: 'A LIKES G.' -> answer 'G' (true 'G')
  session 1: 'A LIKES G.\nB LIKES O.\nC LIKES O.\nD LIKES B.\nE LIKES P.\nF LIKES G.\nG LIKES Y.\nH LIKES B.' -> consolidated 8 facts, sleep dropped 0
  session-2 fact accuracy WITHOUT memory: 0.250   WITH THREAD memory: 1.000

## B5 - MUX SLO-aware scheduling (100 mixed requests, random SLOs)

SLO+accuracy hit-rate: MUX (learned cost model) 0.98 | always spec4 0.58 | always greedy 0.97


## B6 - SIMULACRUM: plan a 2-step session in a learned world model BEFORE executing
    session = [state facts (turn A), ask fact (turn B)]; strategies:
    A in {full, full+consolidate}, B in {full, full+retrieve}
    W(turn B) strat=full   ctx=0 -> acc=0.09 cost=38.9 lat=0.079s
    W(turn B) strat=full   ctx=1 -> acc=0.12 cost=39.2 lat=0.078s
    W(turn B) strat=fullr  ctx=0 -> acc=0.18 cost=38.1 lat=0.076s
    W(turn B) strat=fullr  ctx=1 -> acc=1.00 cost=10.3 lat=0.022s

| policy | modal plan (turnA, turnB) | turn-B accuracy | mean total COGS/session |
|---|---|---|---|
| SIMULACRUM planner (2-step lookahead in world model) | ('fullc', 'fullr') | 1.000 | 54.2 |
| myopic (best per-request, same world model) | ('full', 'fullr') | 0.080 | 82.2 |
| fixed (full, full) | ('full', 'full') | 0.080 | 82.2 |

The planner pays the consolidation fee on turn A *because the world model predicts
turn B becomes accurate* - a myopic controller never does this: on turn A alone,
consolidating is strictly worse (same accuracy, extra fee), so it picks (full, full).

## B7 - CAPITAL COMPUTE: recurring REPEAT stream, cold draft gets invested in

cold-draft REPEAT accuracy: 0.00 before investment -> 1.00 after (stream), 0.43 on fresh prompts
draft acceptance per 10-request window (investment triggers at request 30): ['0.03', '0.03', '0.02', '0.38', '0.38', '0.33']
investments made: [('repeat', 30, 800)]
acceptance before investment: 0.03 -> after: 0.36

## B8 - Verification inheritance: downstream consumes the certificate, skips re-verify

accuracy with inheritance 0.940 (without 0.940); stage-2 re-verifies: 3 with inheritance vs 50 without (of 50 requests)

---
All numbers measured on CPU with the tiny prototype LM; see docs/DESIGN.md for the full system.
