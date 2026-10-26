"""Produce obfuscated copies of the IPC 2023 Learning Track domains (Axis 1).

Levels
  L0  original files, copied unchanged
  L1  renamed: domain name, types, constants, predicates, actions and variables become
      random strings; object names and the order of everything stay as they are
  L2  L1 + object names randomised + order permuted (types, constants, predicates,
      actions, action parameters, precondition/effect conjuncts, objects, initial
      facts, goal conjuncts)
  L3  misleading: the same symbols as L1, but renamed with meaningful words taken
      from a different domain (see vocab.DONOR), matched by arity where possible

Input:  benchmarks/ipc2023-learning/{training,testing}/<domain>/
Output: benchmarks/obfuscated/<level>/{training,testing}/<domain>/  (same layout, so
        llm-heuristics.py and Pyperplan work on it unchanged) and
        benchmarks/obfuscated/<level>/mappings/<domain>/{mapping,task_mapping}.json

The renaming is one consistent map per domain and level, applied to the domain file,
every task file and the example-state.out / example-static.out files that go into the
prompt. Comments are removed at L1-L3 because they leak names. validate_plan.py uses
the mapping files to translate plans back to the original names.

Run from the repository root:
  python -m axis1.obfuscate                       # all levels, all 8 domains
  python -m axis1.obfuscate --levels L1 --domains blocksworld
"""

import argparse
import json
import random
import re
import shutil
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1 import sexp as pddl
from axis1.config import DOMAINS, OUT, SRC, test_tasks, training_tasks
from axis1.vocab import DEFAULT_TYPE, DONOR, EXTRA

KEYWORDS = {
    "define", "domain", "problem", "and", "or", "not", "imply", "exists", "forall",
    "when", "=", "either", "object", "-", "increase", "decrease", "assign",
    "total-cost", "minimize", "maximize",
}

LEVELS = {
    "L0": dict(rename=False, rename_objects=False, shuffle=False, style=None),
    "L1": dict(rename=True, rename_objects=False, shuffle=False, style="random"),
    "L2": dict(rename=True, rename_objects=True, shuffle=True, style="random"),
    "L3": dict(rename=True, rename_objects=False, shuffle=False, style="misleading"),
}


def is_reserved(tok):
    return tok in KEYWORDS or tok.startswith(":") or _is_number(tok)


def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- analysis

def analyse_domain(tree):
    info = {"name": pddl.section(tree, "domain")[1], "types": [], "constants": [],
            "predicates": [], "actions": []}
    types_sec = pddl.section(tree, ":types")
    if types_sec:
        for name, parent in pddl.parse_typed_list(types_sec[1:]):
            for t in [name] + ([parent] if isinstance(parent, str) else []):
                if t != "object" and t not in info["types"]:
                    info["types"].append(t)
    const_sec = pddl.section(tree, ":constants")
    if const_sec:
        info["constants"] = pddl.parse_typed_list(const_sec[1:])
    pred_sec = pddl.section(tree, ":predicates")
    for decl in pred_sec[1:] if pred_sec else []:
        info["predicates"].append((decl[0], pddl.parse_typed_list(decl[1:])))
    for e in tree:
        if isinstance(e, list) and e and e[0] == ":action":
            params = action_field(e, ":parameters") or []
            info["actions"].append((e[1], pddl.parse_typed_list(params)))
    return info


def action_field(action, key):
    for i, tok in enumerate(action):
        if tok == key:
            return action[i + 1]
    return None


def set_action_field(action, key, value):
    for i, tok in enumerate(action):
        if tok == key:
            action[i + 1] = value
            return


def task_objects(tree):
    sec = pddl.section(tree, ":objects")
    return pddl.parse_typed_list(sec[1:]) if sec else []


# ---------------------------------------------------------------- shuffling (L2)

def shuffle_domain(tree, rng):
    """Permute order in place; return {action: [param names in new order]} (original names)."""
    param_order = {}
    for e in tree:
        if not isinstance(e, list) or not e:
            continue
        if e[0] == ":predicates":
            body = e[1:]
            rng.shuffle(body)
            e[1:] = body
        elif e[0] in (":types", ":constants"):
            pairs = pddl.parse_typed_list(e[1:])
            rng.shuffle(pairs)
            e[1:] = pddl.unparse_typed_list(pairs)
        elif e[0] == ":action":
            params = pddl.parse_typed_list(action_field(e, ":parameters") or [])
            rng.shuffle(params)
            set_action_field(e, ":parameters", pddl.unparse_typed_list(params))
            param_order[e[1]] = [p for p, _ in params]
            for key in (":precondition", ":effect"):
                f = action_field(e, key)
                if isinstance(f, list) and f and f[0] == "and":
                    body = f[1:]
                    rng.shuffle(body)
                    f[1:] = body
    idx = [i for i, e in enumerate(tree) if isinstance(e, list) and e and e[0] == ":action"]
    blocks = [tree[i] for i in idx]
    rng.shuffle(blocks)
    for i, b in zip(idx, blocks):
        tree[i] = b
    return param_order


def shuffle_task(tree, rng):
    for e in tree:
        if not isinstance(e, list) or not e:
            continue
        if e[0] == ":objects":
            pairs = pddl.parse_typed_list(e[1:])
            rng.shuffle(pairs)
            e[1:] = pddl.unparse_typed_list(pairs)
        elif e[0] == ":init":
            body = e[1:]
            rng.shuffle(body)
            e[1:] = body
        elif e[0] == ":goal" and isinstance(e[1], list) and e[1] and e[1][0] == "and":
            body = e[1][1:]
            rng.shuffle(body)
            e[1][1:] = body


# ---------------------------------------------------------------- name generation

class Namer:
    def __init__(self, rng, forbidden):
        self.rng = rng
        self.used = set(forbidden) | KEYWORDS

    def take(self, name):
        if name in self.used:
            raise ValueError(f"name clash: {name}")
        self.used.add(name)
        return name

    def random(self, length=6):
        while True:
            s = "".join(self.rng.choice(string.ascii_lowercase) for _ in range(length))
            if s not in self.used:
                return self.take(s)

    def first_free(self, candidates):
        for c in candidates:
            if c not in self.used:
                return self.take(c)
        return None

    def suffixed(self, base):
        k = 2
        while f"{base}-{k}" in self.used:
            k += 1
        return self.take(f"{base}-{k}")


def donor_vocab(donor, src):
    info = analyse_domain(pddl.parse((src / "training" / donor / "domain.pddl").read_text()))
    extra = EXTRA[donor]
    return {
        "domain": donor,
        "predicate": [(n, len(p)) for n, p in info["predicates"]],
        "action": [(n, len(p)) for n, p in info["actions"]],
        "type": info["types"] + [t for t in extra["type"] if t not in info["types"]],
        "extra_predicate": extra["predicate"],
        "extra_action": extra["action"],
    }


def misleading_names(items, donor_items, extras, namer, rng):
    """items/donor_items: [(name, arity)]. Same arity first, then any donor name, then extras."""
    pool = list(donor_items)
    rng.shuffle(pool)
    out = {}
    for name, arity in items:
        same = [d for d, a in pool if a == arity]
        new = (namer.first_free(same) or namer.first_free([d for d, _ in pool])
               or namer.first_free(extras))
        out[name] = new or namer.suffixed(pool[0][0] if pool else "thing")
    return out


# ---------------------------------------------------------------- symbol maps

def build_domain_map(info, opts, rng, src, folder):
    """Return (symbols, categories, donor) for domain-level symbols."""
    originals = ({info["name"]} | set(info["types"]) | {c for c, _ in info["constants"]}
                 | {p for p, _ in info["predicates"]} | {a for a, _ in info["actions"]})
    namer = Namer(rng, originals)
    symbols, cats = {}, {}

    def claim(orig, new, cat):
        if orig in symbols or is_reserved(orig):
            return
        symbols[orig] = new
        cats[orig] = cat

    donor = None
    if opts["style"] == "random":
        claim(info["name"], namer.random(8), "domain")
        for t in info["types"]:
            claim(t, namer.random(), "type")
        for p, _ in info["predicates"]:
            claim(p, namer.random(), "predicate")
        for a, _ in info["actions"]:
            claim(a, namer.random(), "action")
        for c, _ in info["constants"]:
            claim(c, namer.random(), "constant")
    else:
        donor = DONOR[folder]
        dv = donor_vocab(donor, src)
        claim(info["name"], namer.first_free([donor]) or namer.suffixed(donor), "domain")
        for t in info["types"]:
            claim(t, namer.first_free(dv["type"]) or namer.suffixed(dv["type"][0]), "type")
        preds = misleading_names([(p, len(a)) for p, a in info["predicates"]],
                                 dv["predicate"], dv["extra_predicate"], namer, rng)
        acts = misleading_names([(a, len(p)) for a, p in info["actions"]],
                                dv["action"], dv["extra_action"], namer, rng)
        for p, new in preds.items():
            claim(p, new, "predicate")
        for a, new in acts.items():
            claim(a, new, "action")
        default_type = DEFAULT_TYPE[donor]
        counters = {}
        for c, typ in info["constants"]:
            base = symbols.get(typ, default_type) if isinstance(typ, str) else default_type
            counters[base] = counters.get(base, 0) + 1
            claim(c, namer.first_free([f"{base}{counters[base]}"]) or namer.suffixed(base), "constant")
    return symbols, cats, donor, namer


def rename_variables(scope, symbols, style, rng, default_type):
    """Rename all ?variables inside one predicate declaration or action."""
    params = {}
    if scope and scope[0] == ":action":
        params = dict(pddl.parse_typed_list(action_field(scope, ":parameters") or []))
    elif scope:
        params = dict(pddl.parse_typed_list(scope[1:]))
    order = []
    for a in pddl.atoms(scope):
        if a.startswith("?") and a not in order:
            order.append(a)
    var_map, used, counters = {}, set(), {}
    for v in order:
        if style == "random":
            while True:
                new = "?" + "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
                if new not in used:
                    break
        else:
            typ = params.get(v)
            base = symbols.get(typ, default_type) if isinstance(typ, str) else default_type
            counters[base] = counters.get(base, 0) + 1
            new = f"?{base}{counters[base]}"
        used.add(new)
        var_map[v] = new
    return var_map


def apply_map(expr, mapping):
    return pddl.map_atoms(expr, lambda a: mapping.get(a, a))


# ---------------------------------------------------------------- per-domain driver

def write_domain(tree, symbols, opts, rng, default_type, where):
    """Rename symbols and (per scope) variables in a parsed domain; return the new tree."""
    new_tree = []
    for e in tree:
        if isinstance(e, list) and e and e[0] == ":predicates":
            decls = [apply_map(d, {**symbols, **rename_variables(d, symbols, opts["style"], rng, default_type)})
                     for d in e[1:]]
            new_tree.append([":predicates"] + decls)
        elif isinstance(e, list) and e and e[0] == ":action":
            vm = rename_variables(e, symbols, opts["style"], rng, default_type)
            new_tree.append(apply_map(e, {**symbols, **vm}))
        else:
            new_tree.append(apply_map(e, symbols))
    check_no_leak(new_tree, set(symbols), where)
    return new_tree


def obfuscate_domain(domain, level, opts, seed, src, out_root, task_files):
    """task_files: paths relative to src, e.g. training/blocksworld/p01.pddl."""
    out = out_root / level
    domain_dirs = [f"training/{domain}", f"testing/{domain}"]

    if not opts["rename"]:  # L0: verbatim copy
        for d in domain_dirs:
            (out / d).mkdir(parents=True, exist_ok=True)
            shutil.copy(src / d / "domain.pddl", out / d / "domain.pddl")
        for name in EXAMPLE_FILES:
            shutil.copy(src / "training" / domain / name, out / "training" / domain / name)
        for rel in task_files:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / rel, out / rel)
        return

    rng = random.Random(f"{seed}:{domain}:{level}")
    tree = pddl.parse((src / "training" / domain / "domain.pddl").read_text())
    info = analyse_domain(tree)
    original_params = {a: [p for p, _ in ps] for a, ps in info["actions"]}
    param_order = shuffle_domain(tree, rng) if opts["shuffle"] else dict(original_params)

    symbols, cats, donor, _ = build_domain_map(info, opts, rng, src, domain)
    default_type = DEFAULT_TYPE[donor] if opts["style"] == "misleading" else None

    # The training and testing copies of a domain differ only cosmetically (Blocksworld's
    # testing copy also declares :typing), so both get the obfuscated training domain.
    text = pddl.to_text(write_domain(tree, symbols, opts, rng, default_type,
                                     f"{level}/{domain}/domain.pddl")) + "\n"
    for d in domain_dirs:
        (out / d).mkdir(parents=True, exist_ok=True)
        (out / d / "domain.pddl").write_text(text)

    const_names = {symbols.get(c, c) for c, _ in info["constants"]}
    task_mapping = {}
    for rel in task_files:
        task_mapping[rel] = obfuscate_task(src / rel, out / rel, rel, symbols, opts,
                                           seed, domain, level, default_type, const_names)
    task_mapping["examples"] = obfuscate_examples(src / "training" / domain, out / "training" / domain,
                                                  symbols, opts, seed, domain, level)

    mapping = {
        "domain": domain, "level": level, "seed": seed, "options": opts, "donor": donor,
        "obfuscated_domain_name": symbols[info["name"]],
        "symbols": symbols, "categories": cats,
        "action_params": {a: {"original": original_params[a], "obfuscated_order": param_order[a]}
                          for a in original_params},
    }
    mdir = out / "mappings" / domain
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "mapping.json").write_text(json.dumps(mapping, indent=1, sort_keys=True))
    (mdir / "task_mapping.json").write_text(json.dumps(task_mapping, indent=1, sort_keys=True))


EXAMPLE_FILES = ["example-state.out", "example-static.out"]
_FACT = re.compile(r"'(\([^']*\))'")


def obfuscate_examples(src_dir, out_dir, symbols, opts, seed, domain, level):
    """Rewrite the example state/static files shown in the prompt: frozenset({'(p a b)', ...})."""
    rng = random.Random(f"{seed}:{domain}:{level}:examples")
    files = {name: [f[1:-1].split() for f in _FACT.findall((src_dir / name).read_text().lower())]
             for name in EXAMPLE_FILES}
    objects = {}
    if opts["rename_objects"]:
        taken = set(symbols.values()) | set(symbols) | KEYWORDS
        for facts in files.values():
            for fact in facts:
                for obj in fact[1:]:
                    if obj in symbols or obj in objects:
                        continue
                    while True:
                        new = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
                        if new not in taken:
                            break
                    taken.add(new)
                    objects[obj] = new
    mapping = {**symbols, **objects}
    for name, facts in files.items():
        new = ["(" + " ".join(mapping.get(t, t) for t in fact) + ")" for fact in facts]
        if opts["shuffle"]:
            rng.shuffle(new)
        check_no_leak(new_facts_atoms(new), set(symbols) | set(objects), f"{level}/{domain}/{name}")
        body = "{" + ", ".join(repr(f) for f in new) + "}" if new else ""
        (out_dir / name).write_text(f"frozenset({body})")
    return {"objects": objects}


def new_facts_atoms(facts):
    return [f.strip("()").split() for f in facts]


def obfuscate_task(path, out_path, rel, symbols, opts, seed, domain, level, default_type, const_names):
    rng = random.Random(f"{seed}:{domain}:{level}:{rel}")
    tree = pddl.parse(path.read_text())
    if opts["shuffle"]:
        shuffle_task(tree, rng)

    prob = pddl.section(tree, "problem")
    taken = set(symbols.values()) | set(symbols) | KEYWORDS
    mapping = dict(symbols)
    if opts["style"] == "random":
        new_prob = "".join(rng.choice(string.ascii_lowercase) for _ in range(8))
    else:
        suffix = prob[1].rsplit("-", 1)[-1]
        new_prob = f"{symbols[pddl.section(tree, ':domain')[1]]}-{suffix}"
    mapping[prob[1]] = new_prob

    objects = {}
    if opts["rename_objects"]:
        counters = {}
        for obj, typ in task_objects(tree):
            if obj in symbols:   # also a domain symbol: already mapped consistently
                continue
            if opts["style"] == "random":
                while True:
                    new = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
                    if new not in taken:
                        break
            else:
                base = symbols.get(typ, default_type) if isinstance(typ, str) else default_type
                counters[base] = counters.get(base, 0) + 1
                new = f"{base}{counters[base]}"
                while new in taken:
                    counters[base] += 1
                    new = f"{base}{counters[base]}"
            taken.add(new)
            objects[obj] = new
        mapping.update(objects)

    new_tree = apply_map(tree, mapping)
    check_no_leak(new_tree, set(symbols) | set(objects), f"{level}/{domain}/{rel}")
    new_objs = [o for o, _ in task_objects(new_tree)]
    if len(new_objs) != len(set(new_objs)) or set(new_objs) & const_names:
        raise RuntimeError(f"{level}/{domain}/{rel}: object names clash after renaming")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(pddl.to_text(new_tree) + "\n")
    return {"problem_name": new_prob, "objects": objects}


def check_no_leak(tree, renamed, where):
    """Every renamed original symbol must be gone (unless a new name happens to equal it)."""
    left = {a for a in pddl.atoms(tree) if a in renamed}
    if left:
        raise RuntimeError(f"{where}: original names still present: {sorted(left)[:10]}")


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"], choices=list(LEVELS))
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--splits", nargs="+", default=["training", "testing"], choices=["training", "testing"])
    ap.add_argument("--all-test-tasks", action="store_true",
                    help="obfuscate all 90 test tasks instead of the 30-task subset (every third)")
    ap.add_argument("--rename-objects", choices=["level", "yes", "no"], default="level",
                    help="override whether object names are renamed (default: as the level says)")
    args = ap.parse_args(argv)

    for level in args.levels:
        opts = dict(LEVELS[level])
        if args.rename_objects != "level" and opts["rename"]:
            opts["rename_objects"] = args.rename_objects == "yes"
        for domain in args.domains:
            tasks = []
            if "training" in args.splits:
                tasks += training_tasks(domain)
            if "testing" in args.splits:
                tasks += test_tasks(domain, subset=not args.all_test_tasks)
            obfuscate_domain(domain, level, opts, args.seed, args.src, args.out, tasks)
            print(f"{level} {domain}: domain + {len(tasks)} tasks -> {args.out / level}")


if __name__ == "__main__":
    main()
