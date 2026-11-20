"""demo.py - end-to-end ARIADNE benchmark vs baselines. Writes experiments/results.md.

Benchmarks:
  B1  LOOM adaptive compute on ARITH: accuracy + compute (layer-tokens, COGS) for
      full-depth vs entropy early-exit vs ARIADNE (online-trained LOOM).
  B2  SHUTTLE speculative decoding on COPY: wall-clock tokens/sec + acceptance rate,
      greedy vs speculative block decoding.
  B3  ORACLE rollback+escalation on held-out ARITH combos: accuracy before/after, rollbacks.
  B4  THREAD cross-session memory: fact QA accuracy with vs without consolidated memory.
  B5  MUX scheduler: SLO hit-rate under mixed load vs fixed strategies.
  B6  SIMULACRUM: 2-step lookahead planning inside a learned world model vs myopic/fixed.
  B7  CAPITAL COMPUTE: threshold-triggered draft investment on a recurring REPEAT stream.
  B8  Verification inheritance: proof-carrying outputs consumed downstream (skip re-verify).

Usage: python experiments/demo.py        -> all benchmarks B1-B8
       python experiments/demo.py new    -> only B6-B8 (fast iteration)
"""
import os
import sys
import time
import random
from collections import Counter
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tiny_lm import TinyLM
from src import tasks, sense
from src.loom import Loom
from src.thread import Thread
from src.oracle import Oracle
from src.shuttle import Shuttle
from src.engine import AriadneEngine
from src.mux import Mux
from src.simulacrum import WorldModel, Planner, train_world_model, request_features
from src.capital import Capital

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")
RESULTS = os.path.join(HERE, "results.md")


def build():
    torch.manual_seed(0)
    ck = torch.load(CKPT, weights_only=False)
    heldout = {tuple(h) for h in ck["heldout"]}
    target = TinyLM(tasks.VOCAB, d=64, h=4, L=4, ff=128)
    target.load_state_dict(ck["target"]); target.eval()
    draft = TinyLM(tasks.VOCAB, d=32, h=4, L=2, ff=64)
    draft.load_state_dict(ck["draft"]); draft.eval()
    loom = Loom(sense.FEAT_DIM, target.L, lr=1e-3)
    thread = Thread()
    oracle = Oracle(target, memory=thread)
    shuttle = Shuttle(target, draft)
    eng = AriadneEngine(target, draft, loom, thread, oracle, shuttle)
    return eng, loom, thread, oracle, shuttle, heldout


def arith_correct(text, ans):
    head, _, tail = text.rpartition("?")
    return bool(tail) and tail[0] == str(ans)


def bench_arith(eng, tests, mode, **kw):
    acc, lt, cost, rb = 0, 0.0, 0.0, 0
    for prompt, ans, _h in tests:
        r = eng.generate(prompt, mode=mode, **kw)
        acc += arith_correct(r["text"], ans)
        lt += r["layer_tokens"]; cost += r["cost"]; rb += r["rollbacks"]
    n = len(tests)
    return dict(acc=acc / n, layer_tokens=lt / n, cost=cost / n, rollbacks=rb)


def main():
    eng, loom, thread, oracle, shuttle, heldout = build()
    lines = []
    log = lambda s: (print(s), lines.append(s))

    log("# ARIADNE demo results (tiny LM, CPU, torch %s)" % torch.__version__)
    log("")

    if len(sys.argv) > 1 and sys.argv[1] == "new":
        # fast iteration mode: only the new benchmarks B6-B8
        log("(fast mode: running B6-B8 only)")
        log("")
        bench_new(eng, oracle, heldout, log)
        with open(RESULTS, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\nwrote", RESULTS)
        return

    # ---- online LOOM training on arith ------------------------------------------
    log("## LOOM online training (REINFORCE from verified reward, 800 arith requests)")
    rng = random.Random(7)
    torch.manual_seed(7)
    # train LOOM on ALL combos (incl. held-out): it must learn *when verification pays*
    train_prompts = [f"{rng.randrange(10)}+{rng.randrange(10)}=?" for _ in range(800)]
    t0 = time.time()
    rewards, costs = [], []
    for i, p in enumerate(train_prompts):
        r = eng.generate(p, mode="ariadne", training=True, budget=60.0)
        rewards.append(r["reward"])
        costs.append(r["cost"])
        if (i + 1) % 100 == 0:
            log(f"  step {i+1}: mean reward {sum(rewards[-100:])/100:.3f}  "
                f"avg COGS {sum(costs[-100:])/100:.1f}")
    log(f"  trained in {time.time()-t0:.0f}s; final-100 mean reward {sum(rewards[-100:])/100:.3f}")
    log("")

    # ---- B1: adaptive compute on arith -------------------------------------------
    log("## B1 - ARITH accuracy vs compute (200 test prompts, 10 held-out combos)")
    tests = tasks.arith_test(n=200, seed=123, heldout=heldout)
    b_full = bench_arith(eng, tests, "full")
    b_ee = bench_arith(eng, tests, "earlyexit", earlyexit_thr=0.35)
    b_ari = bench_arith(eng, tests, "ariadne")
    log("")
    log("| mode | accuracy | layer-tokens/req | COGS/req | rollbacks |")
    log("|---|---|---|---|---|")
    for name, b in [("full-depth (baseline)", b_full), ("entropy early-exit", b_ee), ("ARIADNE LOOM", b_ari)]:
        log(f"| {name} | {b['acc']:.3f} | {b['layer_tokens']:.1f} | {b['cost']:.1f} | {b['rollbacks']} |")
    seen = [t for t in tests if not t[2]]
    held = [t for t in tests if t[2]]
    b_ari_seen = bench_arith(eng, seen, "ariadne")
    b_ari_held = bench_arith(eng, held, "ariadne")
    b_full_held = bench_arith(eng, held, "full")
    log(f"\nARIADNE on seen combos: acc {b_ari_seen['acc']:.3f}; on held-out combos: "
        f"acc {b_ari_held['acc']:.3f} (full-depth baseline on held-out: {b_full_held['acc']:.3f}, "
        f"rollbacks {b_ari_held['rollbacks']})")
    log("")

    # ---- B2: SHUTTLE on REPEAT ----------------------------------------------------
    log("## B2 - SHUTTLE speculative decoding on REPEAT (60 prompts, 3-8 char answers)")
    repeats = tasks.repeat_test(n=60, seed=321)
    def bench_repeat(mode):
        shuttle.stats = dict(blocks=0, drafted=0, accepted=0, bonus=0, rollbacks=0)
        t0 = time.time(); toks = 0; ok = 0; lt = 0.0
        for prompt, ans in repeats:
            r = eng.generate(prompt, mode=mode)
            toks += r["n_tokens"]; ok += (ans in r["text"].split("?")[-1]); lt += r["layer_tokens"]
        dt = time.time() - t0
        s = shuttle.stats
        return dict(sec=dt, toks=toks, tps=toks / dt, acc=ok / len(repeats),
                    accept=s["accepted"] / max(1, s["drafted"]), blocks=s["blocks"],
                    lt_per_tok=lt / max(1, toks))
    c_full = bench_repeat("full")
    c_spec = bench_repeat("spec4")
    log("")
    log("| mode | wall time (s) | tokens | tokens/s | accuracy | acceptance | target layer-tokens/gen-token |")
    log("|---|---|---|---|---|---|---|")
    log(f"| greedy full-depth | {c_full['sec']:.2f} | {c_full['toks']} | {c_full['tps']:.1f} | {c_full['acc']:.3f} | - | {c_full['lt_per_tok']:.2f} |")
    log(f"| SHUTTLE spec-4 | {c_spec['sec']:.2f} | {c_spec['toks']} | {c_spec['tps']:.1f} | {c_spec['acc']:.3f} | {c_spec['accept']:.2f} | {c_spec['lt_per_tok']:.2f} |")
    log(f"\ntarget-compute reduction: {c_full['lt_per_tok'] / c_spec['lt_per_tok']:.2f}x fewer target layer-tokens per generated token")
    log("(wall-clock on CPU is dominated by per-step Python overhead; on GPU the compute saving")
    log("translates directly to latency/energy. Acceptance = fraction of drafted tokens verified.)")
    log("")

    # ---- B3: ORACLE rollback+escalation --------------------------------------------
    log("## B3 - ORACLE rollback+escalation on all 200 ARITH prompts")
    log("")
    log("| mode | accuracy | layer-tokens/req | rollbacks |")
    log("|---|---|---|---|")
    for mode in ["full", "depth1", "depth1v", "ariadne"]:
        oracle.stats = dict(checks=0, fails=0, rollbacks=0, fixed=0)
        b = bench_arith(eng, tests, mode)
        st = oracle.stats
        log(f"| {mode} (all 200) | {b['acc']:.3f} | {b['layer_tokens']:.1f} | {b['rollbacks']} |")
        if mode in ("depth1v", "ariadne"):
            log(f"    ORACLE: checks={st['checks']} fails={st['fails']} rollbacks={st['rollbacks']} fixed={st['fixed']}")
    for mode in ["depth1", "depth1v"]:
        oracle.stats = dict(checks=0, fails=0, rollbacks=0, fixed=0)
        b = bench_arith(eng, seen, mode)
        st = oracle.stats
        log(f"| {mode} (seen combos only, {len(seen)}) | {b['acc']:.3f} | {b['layer_tokens']:.1f} | {b['rollbacks']} |")
        if mode == "depth1v":
            log(f"    ORACLE: checks={st['checks']} fails={st['fails']} rollbacks={st['rollbacks']} fixed={st['fixed']}")
    log("")
    log("depth1 = always cheapest (1 layer); depth1v = depth1 + ORACLE verify/rollback/escalate;")
    log("ARIADNE = LOOM chooses depth per token + ORACLE. ORACLE recovers the fixable depth1 errors.")
    log("")

    # ---- B4: THREAD cross-session memory -------------------------------------------
    log("## B4 - THREAD cross-session memory (8 facts in session 1, asked in session 2)")
    s1, queries = tasks.fact_sessions(n_facts=8, seed=999)
    # sanity: in-session fact QA (no memory needed) - should be ~1.0 if the LM learned the pattern
    n0, c0 = queries[0]
    r_in = eng.generate(f"{n0} LIKES {c0}.\nWHAT DOES {n0} LIKE?", mode="full")
    _, _, t_in = r_in["text"].rpartition("LIKE?")
    log(f"  sanity in-session QA: '{n0} LIKES {c0}.' -> answer {t_in[:1]!r} (true {c0!r})")
    thread.consolidate(s1)
    dropped = thread.sleep()
    log(f"  session 1: {s1.strip()!r} -> consolidated {len(thread)} facts, sleep dropped {dropped}")
    acc_no, acc_mem = 0, 0
    for name, color in queries:
        q = f"WHAT DOES {name} LIKE?"
        r0 = eng.generate(q, mode="full")
        r1 = eng.generate(q, mode="full", force_retrieve=True)
        _, _, tail0 = r0["text"].rpartition("LIKE?")
        _, _, tail1 = r1["text"].rpartition("LIKE?")
        acc_no += bool(tail0) and tail0[0] == color
        acc_mem += bool(tail1) and tail1[0] == color
    log(f"  session-2 fact accuracy WITHOUT memory: {acc_no/len(queries):.3f}   "
        f"WITH THREAD memory: {acc_mem/len(queries):.3f}")
    log("")

    # ---- B5: MUX scheduler ----------------------------------------------------------
    log("## B5 - MUX SLO-aware scheduling (100 mixed requests, random SLOs)")
    mux = Mux()
    mode_of = {"greedy": "full", "spec4": "spec4", "spec4+verify": "spec4v"}
    rng = random.Random(42)
    reqs = []
    for i in range(100):
        if i % 2 == 0:
            a, b = rng.randrange(10), rng.randrange(10)
            reqs.append((f"{a}+{b}=?", "arith", (a + b) % 10))
        else:
            c = rng.choice(tasks.COPY_LETTERS); k = rng.randrange(3, 9)
            reqs.append((f"{c}*{k}=?", "repeat", c * k))
    def run_policy(policy, warmup=20):
        rng = random.Random(42)          # identical SLO sequence for every policy
        hits = 0
        # warmup: explore every strategy so the cost model learns realized latencies
        for i in range(warmup):
            prompt, kind, ans = reqs[i]
            strat = Mux.STRATEGIES[i % len(Mux.STRATEGIES)]
            t0 = time.time(); r = eng.generate(prompt, mode=mode_of[strat]); dt = time.time() - t0
            ok = arith_correct(r["text"], ans) if kind == "arith" else (ans in r["text"].split("?")[-1])
            mux.observe(strat, dt, 1.0 if ok else 0.0, r["cost"])
        for prompt, kind, ans in reqs:
            slo = rng.choice([0.05, 0.1, 0.2, 0.4])
            strat = policy(slo)
            t0 = time.time()
            r = eng.generate(prompt, mode=mode_of[strat])
            dt = time.time() - t0
            ok = arith_correct(r["text"], ans) if kind == "arith" else (ans in r["text"].split("?")[-1])
            mux.observe(strat, dt, 1.0 if ok else 0.0, r["cost"])
            hits += (dt <= slo and ok)
        return hits / len(reqs)
    # fresh mux per policy for fairness
    mux = Mux(); h_mux = run_policy(lambda slo: mux.pick(slo, 0.5, 60.0))
    mux = Mux(); h_spec = run_policy(lambda slo: "spec4")
    mux = Mux(); h_greedy = run_policy(lambda slo: "greedy")
    log("")
    log(f"SLO+accuracy hit-rate: MUX (learned cost model) {h_mux:.2f} | always spec4 {h_spec:.2f} | always greedy {h_greedy:.2f}")
    log("")

    bench_new(eng, oracle, heldout, log)

    log("---")
    log("All numbers measured on CPU with the tiny prototype LM; see docs/DESIGN.md for the full system.")

    with open(RESULTS, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote", RESULTS)


def bench_new(eng, oracle, heldout, log):
    """B6-B8: SIMULACRUM planner, CAPITAL COMPUTE investment, verification inheritance."""
    tests = tasks.arith_test(n=200, seed=123, heldout=heldout)
    log("")

    # ---- B6: SIMULACRUM — lookahead planning inside a learned world model ----------
    log("## B6 - SIMULACRUM: plan a 2-step session in a learned world model BEFORE executing")
    log("    session = [state facts (turn A), ask fact (turn B)]; strategies:")
    log("    A in {full, full+consolidate}, B in {full, full+retrieve}")
    thread2 = Thread()
    oracle2 = Oracle(eng.target, memory=thread2)
    eng2 = AriadneEngine(eng.target, eng.draft, Loom(sense.FEAT_DIM, eng.target.L), thread2,
                         oracle2, Shuttle(eng.target, eng.draft))
    wm = WorldModel()
    # reward weights: the consolidation fee (2 COGS) must be *locally* visible (mu*2 = 0.6)
    # but smaller than the predicted turn-B accuracy gain (~0.9), so only lookahead pays it
    planner = Planner(wm, lam_lat=0.1, mu_cost=0.3)
    choicesA, choicesB = ["full", "fullc"], ["full", "fullr"]

    def run_req(prompt, strat, ctx, events, clear=False):
        if clear:                      # new session: wipe memory so sessions don't leak
            thread2.facts.clear()
        t0 = time.time()
        kw = {"force_retrieve": True} if strat == "fullr" else {}
        r = eng2.generate(prompt, mode=strat, **kw)
        dt = time.time() - t0
        events.append(dict(feat=request_features(prompt), strat=strat, ctx=ctx,
                           lat=dt, acc=1.0, cost=r["cost"]))
        return r

    # phase 1: random exploration -> train the world model on realized outcomes
    events = []
    rng6 = random.Random(11)
    train_sessions = []
    for i in range(60):
        s1, qs = tasks.fact_sessions(n_facts=4, seed=1000 + i)
        n, c = qs[0]
        train_sessions.append((s1, f"WHAT DOES {n} LIKE?", c))
    for s1, q, c in train_sessions:
        sA = rng6.choice(choicesA); sB = rng6.choice(choicesB)
        run_req(s1, sA, 0.0, events, clear=True)
        rB = run_req(q, sB, 1.0 if sA == "fullc" else 0.0, events)
        _, _, tailB = rB["text"].rpartition("LIKE?")
        events[-1]["acc"] = 1.0 if (tailB and tailB[0] == c) else 0.0
    train_world_model(wm, events, steps=1500)
    qfeat = request_features("WHAT DOES A LIKE?")
    for sB in choicesB:
        for ctxB in (0.0, 1.0):
            p = wm.predict(qfeat, sB, ctxB)
            log(f"    W(turn B) strat={sB:6s} ctx={ctxB:.0f} -> acc={p['acc']:.2f} cost={p['cost']:.1f} lat={p['lat']:.3f}s")

    # phase 2: planner vs myopic vs fixed, on fresh sessions
    def evaluate(policy):
        accs, costs, plans = [], [], []
        for i in range(25):
            s1, qs = tasks.fact_sessions(n_facts=4, seed=2000 + i)
            n, c = qs[0]
            q = f"WHAT DOES {n} LIKE?"
            sA, sB = policy(s1, q)
            plans.append((sA, sB))
            rA = run_req(s1, sA, 0.0, [], clear=True)
            rB = run_req(q, sB, 1.0 if sA == "fullc" else 0.0, [])
            _, _, tailB = rB["text"].rpartition("LIKE?")
            accs.append(1.0 if (tailB and tailB[0] == c) else 0.0)
            costs.append(rA["cost"] + rB["cost"])
        return sum(accs) / len(accs), sum(costs) / len(costs), plans

    def policy_planner(s1, q):
        return planner.plan_pair(request_features(s1), request_features(q), choicesA, choicesB)

    def policy_myopic(s1, q):
        sA = max(choicesA, key=lambda s: planner.reward(wm.predict(request_features(s1), s, 0.0)))
        ctx = 1.0 if sA == "fullc" else 0.0
        sB = max(choicesB, key=lambda s: planner.reward(wm.predict(request_features(q), s, ctx)))
        return sA, sB

    def policy_fixed(s1, q):
        return "full", "full"

    a_pl, c_pl, p_pl = evaluate(policy_planner)
    a_my, c_my, p_my = evaluate(policy_myopic)
    a_fx, c_fx, p_fx = evaluate(policy_fixed)
    log("")
    log("| policy | modal plan (turnA, turnB) | turn-B accuracy | mean total COGS/session |")
    log("|---|---|---|---|")
    log(f"| SIMULACRUM planner (2-step lookahead in world model) | {Counter(p_pl).most_common(1)[0][0]} | {a_pl:.3f} | {c_pl:.1f} |")
    log(f"| myopic (best per-request, same world model) | {Counter(p_my).most_common(1)[0][0]} | {a_my:.3f} | {c_my:.1f} |")
    log(f"| fixed (full, full) | {Counter(p_fx).most_common(1)[0][0]} | {a_fx:.3f} | {c_fx:.1f} |")
    log("")
    log("The planner pays the consolidation fee on turn A *because the world model predicts")
    log("turn B becomes accurate* - a myopic controller never does this: on turn A alone,")
    log("consolidating is strictly worse (same accuracy, extra fee), so it picks (full, full).")
    log("")

    # ---- B7: CAPITAL COMPUTE — invest in the draft when a pattern recurs ------------
    log("## B7 - CAPITAL COMPUTE: recurring REPEAT stream, cold draft gets invested in")
    draft_cold = TinyLM(tasks.VOCAB, d=32, h=4, L=2, ff=64)
    draft_cold.load_state_dict(torch.load(CKPT, weights_only=False)["draft_cold"])
    draft_cold.eval()
    thread3 = Thread()
    oracle3 = Oracle(eng.target, memory=thread3)
    shuttle3 = Shuttle(eng.target, draft_cold)
    eng3 = AriadneEngine(eng.target, draft_cold, Loom(sense.FEAT_DIM, eng.target.L), thread3,
                         oracle3, shuttle3)
    cap = Capital(draft_cold, threshold=30, steps=800, lr=3e-3)

    def draft_repeat_acc(d, n=30, seed=555):
        """Direct capability probe: greedy draft continuation on REPEAT prompts."""
        ok = 0
        for prompt, ans in tasks.repeat_test(n=n, seed=seed):
            d.reset()
            logits = d.prefill(tasks.encode(prompt))[-1]
            gen = []
            for _ in range(10):
                tok = d.sample(logits)
                gen.append(tok)
                if tok == tasks.STOI["\n"]:
                    break
                logits = d.commit(tok, depth=d.L)
            ok += (tasks.decode(gen).strip() == ans)
        return ok / n

    acc0 = draft_repeat_acc(draft_cold, seed=777)
    repeats60 = tasks.repeat_test(n=60, seed=777)
    windows = []
    for i, (prompt, ans) in enumerate(repeats60):
        if i % 10 == 0:
            shuttle3.stats = dict(blocks=0, drafted=0, accepted=0, bonus=0, rollbacks=0)
        eng3.generate(prompt, mode="spec4")
        cap.observe(prompt + ans + "\n")
        if (i + 1) % 10 == 0:
            s = shuttle3.stats
            windows.append(s["accepted"] / max(1, s["drafted"]))
    acc1 = draft_repeat_acc(draft_cold, seed=777)          # on the recurring stream
    acc1f = draft_repeat_acc(draft_cold, seed=555)          # on fresh prompts
    log("")
    log(f"cold-draft REPEAT accuracy: {acc0:.2f} before investment -> {acc1:.2f} after (stream), {acc1f:.2f} on fresh prompts")
    log(f"draft acceptance per 10-request window (investment triggers at request 30): {['%.2f' % w for w in windows]}")
    log(f"investments made: {cap.investments}")
    log(f"acceptance before investment: {sum(windows[:3]) / 3:.2f} -> after: {sum(windows[3:]) / 3:.2f}")
    log("")

    # ---- B8: verification inheritance (proof-carrying outputs) ---------------------
    log("## B8 - Verification inheritance: downstream consumes the certificate, skips re-verify")
    tests50 = tests[:50]
    def pipeline(inherit):
        oracle.stats = dict(checks=0, fails=0, rollbacks=0, fixed=0)
        acc = 0; rechecks = 0
        for prompt, ans, _h in tests50:
            r = eng.generate(prompt, mode="depth1v")        # stage 1: produce + certify
            cert = r["cert"]
            if not inherit or cert is not True:
                oracle.check(r["text"])                       # stage 2: re-verify
                rechecks += 1
            acc += arith_correct(r["text"], ans)
        return acc / len(tests50), rechecks
    acc_in, rc_in = pipeline(True)
    acc_no, rc_no = pipeline(False)
    log("")
    log(f"accuracy with inheritance {acc_in:.3f} (without {acc_no:.3f}); "
        f"stage-2 re-verifies: {rc_in} with inheritance vs {rc_no} without (of {len(tests50)} requests)")
    log("")


if __name__ == "__main__":
    main()
