"""train.py — train the tiny target LM and draft model on the synthetic corpus, save ckpt."""
import os
import sys
import time
import random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tiny_lm import TinyLM
from src import tasks

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")


def train_model(model, corpus, epochs=4, batch=64, lr=3e-3, seed=0, log=print,
                heldout=None, eval_every=2):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = [tasks.encode(s) for s in corpus]
    tests = tasks.arith_test(n=100, seed=123, heldout=heldout)
    t0 = time.time()
    for ep in range(epochs):
        rng.shuffle(data)
        total, nb = 0.0, 0
        for i in range(0, len(data), batch):
            seqs = data[i:i + batch]
            L = max(len(s) for s in seqs)
            x = torch.zeros(len(seqs), L - 1, dtype=torch.long)
            y = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
            for j, s in enumerate(seqs):
                x[j, : len(s) - 1] = torch.tensor(s[:-1])
                y[j, : len(s) - 1] = torch.tensor(s[1:])
            logits = model.forward_batch(x)                     # (B, T, V)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        msg = f"  epoch {ep + 1}/{epochs}  loss {total / nb:.4f}  ({time.time() - t0:.0f}s)"
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            msg += f"  arith_acc {quick_acc(model, tests):.3f}"
        log(msg)
    return model


@torch.no_grad()
def quick_acc(model, tests):
    ok = 0
    for prompt, ans, _held in tests:
        model.reset()
        logits = model.prefill(tasks.encode(prompt))[-1]
        tok = model.sample(logits)
        ok += (tok == tasks.STOI[str(ans)])
    return ok / len(tests)


def main():
    torch.manual_seed(0)
    print("building corpus...")
    corpus, heldout = tasks.make_corpus(n_arith=4000, n_fact=800, n_repeat=800, seed=0)
    print(f"corpus: {len(corpus)} seqs, heldout arith combos: {len(heldout)}")
    target = TinyLM(tasks.VOCAB, d=64, h=4, L=4, ff=128)
    draft = TinyLM(tasks.VOCAB, d=32, h=4, L=2, ff=64)
    print("training target (d=64, L=4)...")
    train_model(target, corpus, epochs=10, lr=3e-3, heldout=heldout)
    print("training draft (d=32, L=2)...")
    train_model(draft, corpus, epochs=10, lr=4e-3, heldout=heldout)
    # cold draft: never saw REPEAT -> used to demonstrate CAPITAL COMPUTE investment
    draft_cold = TinyLM(tasks.VOCAB, d=32, h=4, L=2, ff=64)
    corpus_no_repeat = [s for s in corpus if "*" not in s]
    print("training cold draft (no REPEAT data)...")
    train_model(draft_cold, corpus_no_repeat, epochs=10, lr=4e-3, heldout=heldout)
    tests = tasks.arith_test(n=200, seed=123, heldout=heldout)
    print(f"target arith acc: {quick_acc(target, tests):.3f}")
    print(f"draft  arith acc: {quick_acc(draft, tests):.3f}")
    torch.save({"target": target.state_dict(), "draft": draft.state_dict(),
                "draft_cold": draft_cold.state_dict(),
                "heldout": sorted(heldout)}, CKPT)
    print("saved", CKPT)


if __name__ == "__main__":
    main()
