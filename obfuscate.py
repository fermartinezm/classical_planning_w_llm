#!/usr/bin/env python3
"""
Build semantically obfuscated copies of the IPC 2023 Learning
Track domains at increasing levels.

Levels
------
L1  renamed:   domain name, types, constants, predicates, actions and variables become
               random strings. Object names in tasks are kept. Order is untouched.
               (Closest to the paper's Blocksworld obfuscation test, minus objects.)
L2  renamed + shuffled: L1 + object names randomised + order permuted
               (predicate declarations, actions, action parameters, precondition /
               effect conjuncts, task objects, initial facts, goal conjuncts).
L3  misleading: every symbol (incl. objects) is consistently renamed with meaningful
               words from a *different* domain (e.g. Blocksworld uses Transport's
               vocabulary). Order is untouched, so L3 vs L2/L1 isolates wrong cues.

For every level the same symbol mapping is applied to the domain, to every training
and testing task, and to example-state.out / example-static.out (used in the prompt).
All comments are stripped (Sokoban tasks contain an ASCII map of the level, Rovers tasks
contain commented-out facts, domain headers describe the domain in English).

Different --seeds give different random renamings (professor's suggestion: several
copies of the same domain with different renamings). Seed 1 is written to
<out>/<LEVEL>/, seed k>1 to <out>/<LEVEL>-s<k>/.

The output is deterministic: the same level/seed always produces the same files, so
benchmarks/obfuscated/ is not committed (it is ~700 MB) and is rebuilt with this script.

Usage (from the repository root)
-----
    python obfuscate.py                       # L1 L2 L3, seed 1, all 8 domains
    python obfuscate.py --levels L1 --seeds 2 3
    python obfuscate.py --domains blocksworld --levels L2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import string
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

SExpr = Union[str, List["SExpr"]]

DOMAINS = ["blocksworld", "childsnack", "floortile", "miconic",
           "rovers", "sokoban", "spanner", "transport"]

# Donor domain for L3 (a derangement: no domain keeps its own vocabulary).
L3_DONOR = {
    "blocksworld": "transport", "transport": "blocksworld",
    "childsnack": "rovers", "rovers": "childsnack",
    "miconic": "sokoban", "sokoban": "miconic",
    "floortile": "spanner", "spanner": "floortile",
}

RESERVED = {"define", "domain", "problem", "and", "or", "not", "imply", "exists",
            "forall", "when", "either", "object", "=", "-"}

IDENT_RE = re.compile(r"\??[A-Za-z][A-Za-z0-9_\-]*")


# --------------------------------------------------------------------------- parsing
def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for line in text.splitlines():
        line = line.split(";", 1)[0]          # strip comments (full-line and inline)
        line = line.replace("(", " ( ").replace(")", " ) ")
        tokens.extend(t.lower() for t in line.split())
    return tokens


def parse(text: str) -> SExpr:
    tokens = tokenize(text)
    pos = 0

    def rec() -> SExpr:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            lst: List[SExpr] = []
            while tokens[pos] != ")":
                lst.append(rec())
            pos += 1
            return lst
        if tok == ")":
            raise ValueError("Unexpected ')'")
        return tok

    tree = rec()
    return tree


def section(tree: List[SExpr], key: str) -> Optional[List[SExpr]]:
    for item in tree:
        if isinstance(item, list) and item and item[0] == key:
            return item
    return None


def typed_list(items: List[SExpr]) -> List[Tuple[str, Optional[str]]]:
    """'a b - t c' -> [(a,t),(b,t),(c,None)]"""
    out: List[Tuple[str, Optional[str]]] = []
    pending: List[str] = []
    i = 0
    while i < len(items):
        it = items[i]
        if it == "-":
            typ = items[i + 1]
            if isinstance(typ, list):      # (either ...) — not used in our domains
                typ = " ".join(map(str, typ))
            out.extend((p, typ) for p in pending)
            pending = []
            i += 2
            continue
        pending.append(it)  # type: ignore[arg-type]
        i += 1
    out.extend((p, None) for p in pending)
    return out


def emit_typed(pairs: List[Tuple[str, Optional[str]]]) -> List[SExpr]:
    out: List[SExpr] = []
    for name, typ in pairs:
        out.append(name)
        if typ is not None:
            out.extend(["-", typ])
    return out


# --------------------------------------------------------------------------- printing
def fmt(e: SExpr) -> str:
    if isinstance(e, str):
        return e
    return "(" + " ".join(fmt(x) for x in e) + ")"


def fmt_block(e: SExpr, indent: str) -> str:
    """Print an (and ...) one conjunct per line, anything else on one line."""
    if isinstance(e, list) and e and e[0] == "and" and len(e) > 2:
        inner = ("\n" + indent + "     ").join(fmt(x) for x in e[1:])
        return "(and " + inner + ")"
    return fmt(e)


def print_domain(tree: List[SExpr]) -> str:
    lines = [f"(define {fmt(tree[1])}"]
    for item in tree[2:]:
        assert isinstance(item, list)
        head = item[0]
        if head == ":predicates":
            lines.append("(:predicates")
            lines.extend("    " + fmt(p) for p in item[1:])
            lines.append(")")
        elif head == ":action":
            lines.append(f"(:action {item[1]}")
            j = 2
            while j < len(item):
                key, val = item[j], item[j + 1]
                lines.append(f"  {key} {fmt_block(val, '  ' + ' ' * (len(str(key)) + 1))}")
                j += 2
            lines.append(")")
        else:
            lines.append(fmt(item))
        lines.append("")
    lines.append(")")
    return "\n".join(lines) + "\n"


def print_problem(tree: List[SExpr]) -> str:
    lines = [f"(define {fmt(tree[1])}"]
    for item in tree[2:]:
        assert isinstance(item, list)
        head = item[0]
        if head == ":objects":
            lines.append("(:objects")
            pairs = typed_list(item[1:])
            lines.extend("    " + " ".join(map(str, emit_typed([pr]))) for pr in pairs)
            lines.append(")")
        elif head == ":init":
            lines.append("(:init")
            lines.extend("    " + fmt(f) for f in item[1:])
            lines.append(")")
        elif head == ":goal":
            lines.append("(:goal " + fmt_block(item[1], "      ") + ")")
        else:
            lines.append(fmt(item))
    lines.append(")")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- symbols
class Symbols:
    """Collect every renameable symbol of a domain (+ all its tasks) by category."""

    def __init__(self) -> None:
        self.cat: Dict[str, str] = {}             # symbol -> category (first/priority)
        self.obj_type: Dict[str, Optional[str]] = {}
        self.order: List[str] = []
        self.types: List[str] = []
        self.type_parent: Dict[str, Optional[str]] = {}
        self.domain_name: Optional[str] = None

    PRIORITY = ["predicate", "action", "type", "constant", "object",
                "domain", "problem", "variable"]

    def add(self, sym: str, cat: str) -> None:
        if sym in RESERVED or sym.startswith(":"):
            return
        old = self.cat.get(sym)
        if old is None:
            self.order.append(sym)
            self.cat[sym] = cat
        elif self.PRIORITY.index(cat) < self.PRIORITY.index(old):
            self.cat[sym] = cat

    def collect_vars(self, e: SExpr) -> None:
        if isinstance(e, str):
            if e.startswith("?"):
                self.add(e, "variable")
        else:
            for x in e:
                self.collect_vars(x)

    def from_domain(self, tree: List[SExpr]) -> None:
        self.domain_name = tree[1][1]  # type: ignore[index]
        self.add(tree[1][1], "domain")  # type: ignore[index]
        types = section(tree, ":types")
        if types:
            for name, parent in typed_list(types[1:]):
                self.add(name, "type")
                self.types.append(name)
                self.type_parent[name] = parent
                if parent and parent != "object":
                    self.add(parent, "type")
        consts = section(tree, ":constants")
        if consts:
            for name, typ in typed_list(consts[1:]):
                self.add(name, "constant")
                self.obj_type.setdefault(name, typ)
        preds = section(tree, ":predicates")
        if preds:
            for p in preds[1:]:
                self.add(p[0], "predicate")  # type: ignore[index]
        for item in tree[2:]:
            if isinstance(item, list) and item and item[0] == ":action":
                self.add(item[1], "action")  # type: ignore[arg-type]
        self.collect_vars(tree)

    def from_problem(self, tree: List[SExpr]) -> None:
        self.add(tree[1][1], "problem")  # type: ignore[index]
        objs = section(tree, ":objects")
        if objs:
            for name, typ in typed_list(objs[1:]):
                self.add(name, "object")
                self.obj_type.setdefault(name, typ)
        # any symbol in init/goal not yet known is treated as an object
        for key in (":init", ":goal"):
            sec = section(tree, key)
            if sec:
                self._atoms_objects(sec[1:])

    def _atoms_objects(self, es: List[SExpr]) -> None:
        for e in es:
            if isinstance(e, list) and e:
                if e[0] in ("and", "not", "or"):
                    self._atoms_objects(e[1:])
                else:
                    for arg in e[1:]:
                        if isinstance(arg, str) and arg not in self.cat:
                            self.add(arg, "object")


# --------------------------------------------------------------------------- mappings
def rand_name(rng: random.Random, used: Set[str], prefix: str = "", length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    while True:
        name = prefix + rng.choice(string.ascii_lowercase) + "".join(
            rng.choice(alphabet) for _ in range(length - 1))
        if name not in used and name not in RESERVED:
            used.add(name)
            return name


def build_random_map(sym: Symbols, rng: random.Random, rename_objects: bool,
                     keep: Set[str]) -> Dict[str, str]:
    used: Set[str] = set(keep)
    mapping: Dict[str, str] = {}
    var_counter = 0
    for s in sym.order:
        c = sym.cat[s]
        if c == "object" and not rename_objects:
            continue
        if c == "variable":
            var_counter += 1
            mapping[s] = "?" + rand_name(rng, used, length=5)
        else:
            mapping[s] = rand_name(rng, used)
    return mapping


def build_misleading_map(sym: Symbols, donor: Symbols, rng: random.Random) -> Dict[str, str]:
    used: Set[str] = set()
    mapping: Dict[str, str] = {}

    def unique(base: str) -> str:
        name, k = base, 2
        while name in used or name in RESERVED:
            name, k = f"{base}-{k}", k + 1
        used.add(name)
        return name

    def pool(cat: str) -> List[str]:
        p = [s for s in donor.order if donor.cat[s] == cat and not s.startswith("?")]
        rng.shuffle(p)
        return p

    pools = {c: pool(c) for c in ("predicate", "action", "type")}
    donor_types = pools["type"] or ["item"]
    idx = {c: 0 for c in pools}

    def take(cat: str) -> str:
        p = pools[cat] if pools[cat] else donor_types
        name = p[idx[cat] % len(p)]
        idx[cat] += 1
        return unique(name)

    # schema symbols first
    for s in sym.order:
        c = sym.cat[s]
        if c in ("predicate", "action", "type"):
            mapping[s] = take(c)
    donor_name = str(donor.domain_name)
    dom = sym.domain_name
    if dom is not None and dom not in mapping:
        mapping[dom] = unique(donor_name)
    # constants / objects: named after the (misleading) type they belong to
    counters: Dict[str, int] = {}
    for s in sym.order:
        c = sym.cat[s]
        if c in ("constant", "object"):
            t = sym.obj_type.get(s)
            base = mapping.get(t, donor_types[0]) if t else donor_types[0]
            base = re.sub(r"-\d+$", "", base)
            counters[base] = counters.get(base, 0) + 1
            mapping[s] = unique(f"{base}{counters[base]}")
        elif c == "problem":
            mapping[s] = unique(f"{donor_name}-task")
    var_counter = 0
    for s in sym.order:
        if sym.cat[s] == "variable":
            var_counter += 1
            mapping[s] = unique(f"?v{var_counter}")
    return mapping


# --------------------------------------------------------------------------- transforms
def rename(e: SExpr, m: Dict[str, str]) -> SExpr:
    if isinstance(e, str):
        return m.get(e, e)
    return [rename(x, m) for x in e]


def shuffle_and(e: SExpr, rng: random.Random) -> SExpr:
    if isinstance(e, list) and e and e[0] == "and":
        body = list(e[1:])
        rng.shuffle(body)
        return ["and"] + body
    return e


def shuffle_domain(tree: List[SExpr], rng: random.Random) -> List[SExpr]:
    head = tree[:2]
    rest = tree[2:]
    actions = [x for x in rest if isinstance(x, list) and x and x[0] == ":action"]
    others = [x for x in rest if not (isinstance(x, list) and x and x[0] == ":action")]
    new_others = []
    for sec in others:
        if isinstance(sec, list) and sec and sec[0] == ":predicates":
            body = list(sec[1:])
            rng.shuffle(body)
            sec = [":predicates"] + body
        new_others.append(sec)
    new_actions = []
    for act in actions:
        act = list(act)
        j = 2
        while j < len(act):
            key = act[j]
            if key == ":parameters":
                pairs = typed_list(act[j + 1])  # type: ignore[arg-type]
                rng.shuffle(pairs)
                act[j + 1] = emit_typed(pairs)
            elif key in (":precondition", ":effect"):
                act[j + 1] = shuffle_and(act[j + 1], rng)
            j += 2
        new_actions.append(act)
    rng.shuffle(new_actions)
    return head + new_others + new_actions


def shuffle_problem(tree: List[SExpr], rng: random.Random) -> List[SExpr]:
    out = list(tree[:2])
    for sec in tree[2:]:
        if isinstance(sec, list) and sec:
            if sec[0] == ":objects":
                pairs = typed_list(sec[1:])
                rng.shuffle(pairs)
                sec = [":objects"] + emit_typed(pairs)
            elif sec[0] == ":init":
                body = list(sec[1:])
                rng.shuffle(body)
                sec = [":init"] + body
            elif sec[0] == ":goal":
                sec = [":goal", shuffle_and(sec[1], rng)]
        out.append(sec)
    return out


def rename_text(text: str, m: Dict[str, str]) -> str:
    """Rename identifiers inside example-state.out / example-static.out."""
    return IDENT_RE.sub(lambda mt: m.get(mt.group(0).lower(), mt.group(0)), text)


# --------------------------------------------------------------------------- verify
def verify_level(src: Path, out_root: Path, level_dir: str, domain: str) -> List[str]:
    """Leak check: no original schema name (predicate, action, type, constant, domain name)
    may survive in the obfuscated files the LLM sees (domain, p01, p99, example-*.out).
    Only meaningful for L1/L2 (L3 may legitimately reuse a word shared by two domains)."""
    sym, _ = load_symbols(src, domain)
    schema = {s for s in sym.order
              if sym.cat[s] in ("predicate", "action", "type", "constant", "domain")
              and len(s) > 2 and not s.startswith("?")}
    tdir = out_root / level_dir / "training" / domain
    leaks = []
    for name in ("domain.pddl", "p01.pddl", "p99.pddl", "example-state.out", "example-static.out"):
        f = tdir / name
        if not f.is_file():
            continue
        words = {w.lower() for w in IDENT_RE.findall(f.read_text(encoding="utf-8"))}
        for w in sorted(words & schema):
            leaks.append(f"{name}: '{w}'")
    return leaks


# --------------------------------------------------------------------------- driver
def load_symbols(src: Path, domain: str) -> Tuple[Symbols, Dict[Path, List[SExpr]]]:
    sym = Symbols()
    trees: Dict[Path, List[SExpr]] = {}
    for split in ("training", "testing"):
        ddir = src / split / domain
        dtree = parse((ddir / "domain.pddl").read_text(encoding="utf-8", errors="replace"))
        trees[ddir / "domain.pddl"] = dtree  # type: ignore[assignment]
        sym.from_domain(dtree)  # type: ignore[arg-type]
    for split in ("training", "testing"):
        ddir = src / split / domain
        for p in sorted(ddir.glob("*.pddl")):
            if p.name == "domain.pddl":
                continue
            t = parse(p.read_text(encoding="utf-8", errors="replace"))
            trees[p] = t  # type: ignore[assignment]
            sym.from_problem(t)  # type: ignore[arg-type]
    return sym, trees


def obfuscate_domain(src: Path, out_root: Path, domain: str, level: str, seed: int,
                     cache: Dict[str, Symbols]) -> Dict[str, str]:
    sym, trees = load_symbols(src, domain)
    cache[domain] = sym
    rng = random.Random(f"{level}-{seed}-{domain}")

    if level == "L1":
        keep = {s for s in sym.order if sym.cat[s] == "object"}
        mapping = build_random_map(sym, rng, rename_objects=False, keep=keep)
    elif level == "L2":
        mapping = build_random_map(sym, rng, rename_objects=True, keep=set())
    elif level == "L3":
        donor = L3_DONOR[domain]
        if donor not in cache:
            cache[donor], _ = load_symbols(src, donor)
        mapping = build_misleading_map(sym, cache[donor], rng)
    else:
        raise ValueError(level)

    assert len(set(mapping.values())) == len(mapping), "mapping is not injective"

    level_dir = level if seed == 1 else f"{level}-s{seed}"
    for path, tree in trees.items():
        split = path.parent.parent.name
        dst = out_root / level_dir / split / domain / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        new = rename(tree, mapping)
        is_domain = path.name == "domain.pddl"
        if level == "L2":
            srng = random.Random(f"shuffle-{seed}-{domain}-{split}-{path.name}")
            if is_domain:
                # same shuffled domain for training and testing
                srng = random.Random(f"shuffle-{seed}-{domain}-domain")
                new = shuffle_domain(new, srng)  # type: ignore[arg-type]
            else:
                new = shuffle_problem(new, srng)  # type: ignore[arg-type]
        text = print_domain(new) if is_domain else print_problem(new)  # type: ignore[arg-type]
        dst.write_text(text, encoding="utf-8")

    for extra in ("example-state.out", "example-static.out"):
        srcf = src / "training" / domain / extra
        if srcf.is_file():
            dst = out_root / level_dir / "training" / domain / extra
            dst.write_text(rename_text(srcf.read_text(encoding="utf-8"), mapping), encoding="utf-8")

    mdir = out_root / level_dir / "mappings"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{domain}.json").write_text(json.dumps(
        {"level": level, "seed": seed, "domain": domain,
         "donor": L3_DONOR.get(domain) if level == "L3" else None,
         "mapping": mapping}, indent=2), encoding="utf-8")
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="benchmarks/ipc2023-learning")
    ap.add_argument("--out", default="benchmarks/obfuscated")
    ap.add_argument("--levels", nargs="+", default=["L1", "L2", "L3"], choices=["L1", "L2", "L3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    for seed in args.seeds:
        for level in args.levels:
            level_dir = out / (level if seed == 1 else f"{level}-s{seed}")
            # remove stale files only for the domains being rebuilt
            for d in args.domains:
                for split in ("training", "testing"):
                    shutil.rmtree(level_dir / split / d, ignore_errors=True)
            cache: Dict[str, Symbols] = {}
            for d in args.domains:
                m = obfuscate_domain(src, out, d, level, seed, cache)
                msg = f"[{level} seed={seed}] {d:12s} renamed {len(m):4d} symbols -> {level_dir}"
                if level in ("L1", "L2"):
                    leaks = verify_level(src, out, level_dir.name, d)
                    msg += "  leak check: OK" if not leaks else f"  LEAKS: {leaks[:5]}"
                print(msg)


if __name__ == "__main__":
    main()
