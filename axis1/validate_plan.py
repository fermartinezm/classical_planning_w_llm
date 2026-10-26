"""Translate a plan found on an obfuscated task back to original names and check it there.

The check simulates the plan on the ORIGINAL grounded task (the authors' Pyperplan in
src/pyperplan):
every action must be applicable in turn and the goal must hold at the end. A plan that
passes is a correct plan for the unobfuscated task, so this catches obfuscator bugs that
change the semantics of a domain.

  python -m axis1.validate_plan --level L2 --domain blocksworld \
      --task training/blocksworld/p05.pddl --plan path/to/p05.pddl.soln
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1.config import OUT, PYPERPLAN, SRC


def load_mappings(out_root, level, domain):
    mdir = out_root / level / "mappings" / domain
    mapping = json.loads((mdir / "mapping.json").read_text())
    tasks = json.loads((mdir / "task_mapping.json").read_text())
    return mapping, tasks


def translate_plan(lines, mapping, task_entry):
    """Map '(act a b c)' lines from obfuscated names back to the original ones."""
    inverse = {v: k for k, v in mapping["symbols"].items()}
    inverse.update({v: k for k, v in task_entry.get("objects", {}).items()})
    out = []
    for line in lines:
        line = line.strip().lower()
        if not line or line.startswith(";"):
            continue
        name, *args = line.strip("()").split()
        orig_action = inverse.get(name, name)
        args = [inverse.get(a, a) for a in args]
        params = mapping["action_params"].get(orig_action)
        if params and params["obfuscated_order"] != params["original"]:
            by_param = dict(zip(params["obfuscated_order"], args))
            args = [by_param[p] for p in params["original"]]
        out.append("(" + " ".join([orig_action] + args) + ")")
    return out


def check_plan(domain_file, task_file, plan):
    """Return (ok, message). Simulates `plan` (original names) on the grounded task."""
    if str(PYPERPLAN) not in sys.path:
        sys.path.insert(0, str(PYPERPLAN))
    from grounding import ground
    from pddl.parser import Parser

    parser = Parser(str(domain_file), str(task_file))
    dom = parser.parse_domain()
    task = ground(parser.parse_problem(dom))
    ops = {op.name: op for op in task.operators}
    state = task.initial_state
    for i, step in enumerate(plan):
        op = ops.get(step)
        if op is None:
            return False, f"step {i}: unknown or pruned operator {step}"
        if not op.applicable(state):
            return False, f"step {i}: {step} not applicable"
        state = op.apply(state)
    if not task.goal_reached(state):
        return False, "goal not reached at the end of the plan"
    return True, f"valid ({len(plan)} steps)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--task", required=True, help="e.g. training/blocksworld/p05.pddl")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--obf", type=Path, default=OUT)
    args = ap.parse_args(argv)

    lines = args.plan.read_text().splitlines()
    if args.level != "L0":
        mapping, tasks = load_mappings(args.obf, args.level, args.domain)
        lines = translate_plan(lines, mapping, tasks[args.task])
    task = args.src / args.task
    ok, msg = check_plan(task.parent / "domain.pddl", task, lines)
    print(msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
