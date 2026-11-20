"""tasks.py — char tokenizer + synthetic task generators used to train and evaluate ARIADNE.

Task families (all end with EOS "\\n"):
  ARITH : "3+4=?7\n"   — the answer digit is the *hard* token (needs computation);
                          template tokens are easy. Ideal for adaptive-depth routing (LOOM).
  FACT  : "A LIKES R.\\nWHAT DOES A LIKE?R\n" — in-session copy-from-context. Across sessions
          the assignment is only knowable via THREAD episodic memory (the cross-session test
          plants *random* assignments not seen in training).
  REPEAT: "A*5=?AAAAA\n" — long-generation counting/repetition task; ideal for measuring
          SHUTTLE speculative-decoding acceptance and wall-clock speedup.
"""
import random
import string

ALPHABET = list("0123456789+=?.*\n ") + list(string.ascii_uppercase)
STOI = {c: i for i, c in enumerate(ALPHABET)}
ITOS = {i: c for i, c in enumerate(ALPHABET)}
VOCAB = len(ALPHABET)
EOS = "\n"

NAMES = list("ABCDEFGH")          # 8 names
COLORS = list("RGBYWPKO")         # 7 colors
COPY_LETTERS = list("ABCDEFGHJKLMNPRSTVWY")  # letters used by COPY task


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[i] for i in ids)


def arith_example(a, b):
    return f"{a}+{b}=?{(a + b) % 10}\n"


def fact_example(rng, assignment=None):
    assignment = assignment or {n: rng.choice(COLORS) for n in NAMES}
    n = rng.choice(NAMES)
    return f"{n} LIKES {assignment[n]}.\nWHAT DOES {n} LIKE?{assignment[n]}\n", assignment


def repeat_example(rng):
    c = rng.choice(COPY_LETTERS)
    k = rng.randrange(3, 9)
    return f"{c}*{k}=?{c * k}\n"


def make_corpus(n_arith=4000, n_fact=800, n_repeat=800, seed=0, holdout_frac=0.2):
    """Training corpus. A fraction of (a,b) combos is held out of training so the model is
    genuinely uncertain on them -> LOOM/ORACLE have something to fix."""
    rng = random.Random(seed)
    heldout = {(a, b) for a in range(10) for b in range(10)
               if rng.random() < holdout_frac}
    data = []
    for _ in range(n_arith):
        a, b = rng.randrange(10), rng.randrange(10)
        while (a, b) in heldout:
            a, b = rng.randrange(10), rng.randrange(10)
        data.append(arith_example(a, b))
    for _ in range(n_fact):
        s, _ = fact_example(rng)
        data.append(s)
    for _ in range(n_repeat):
        data.append(repeat_example(rng))
    rng.shuffle(data)
    return data, heldout


def arith_test(n=200, seed=123, heldout=None):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randrange(10), rng.randrange(10)
        out.append((f"{a}+{b}=?", (a + b) % 10, (a, b) in (heldout or set())))
    return out


def repeat_test(n=100, seed=321):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        c = rng.choice(COPY_LETTERS)
        k = rng.randrange(3, 9)
        out.append((f"{c}*{k}=?", c * k))
    return out


def fact_sessions(n_facts=8, seed=999):
    """Cross-session memory test: session 1 states facts with a RANDOM assignment;
    session 2 asks about them. Returns (session1_text, [(name, color)])."""
    rng = random.Random(seed)
    assignment = {n: rng.choice(COLORS) for n in NAMES[:n_facts]}
    s1 = "\n".join(f"{n} LIKES {assignment[n]}." for n in assignment) + "\n"
    queries = [(n, assignment[n]) for n in assignment]
    return s1, queries


def parse_arith(prompt):
    """'3+4=?' -> (3,4) or None."""
    try:
        a, rest = prompt.split("+")
        b, _ = rest.split("=")
        return int(a), int(b)
    except Exception:
        return None


def parse_fact_query(prompt):
    """'WHAT DOES A LIKE?' -> 'A' or None."""
    p = prompt.strip()
    if p.startswith("WHAT DOES ") and p.endswith(" LIKE?"):
        return p[len("WHAT DOES "):-len(" LIKE?")].strip()
    return None
