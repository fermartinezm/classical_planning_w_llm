from heuristics.heuristic_base import Heuristic
from typing import List, Set, Dict
import math


def get_parts(fact: str) -> List[str]:
    """Return the components of a PDDL fact string, e.g. "(p a b)" -> ["p","a","b"]."""
    return fact[1:-1].split()


def make_fact(pred: str, args: List[str]) -> str:
    """Construct a fact string from predicate and argument list."""
    if args:
        return "({} {})".format(pred, " ".join(args))
    else:
        return "({})".format(pred)


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic builds a small relaxed planning graph (h_add style) using the
    grounded operators of the secret_domain. It estimates the number of actions
    required to satisfy all goal facts by computing the minimum additive cost
    to derive each goal fact from the current state's facts assuming actions
    only add facts (i.e., deletes are ignored). The returned heuristic is the
    sum over the estimated costs of the goal facts.

    Assumptions
    -----------
    - Facts are represented as strings like "(predicate arg1 arg2)".
    - The only predicates used in the domain are:
        yfm3frbn ?x
        a4ha3ujl ?x
        ug241uf7
        jjtnwk3y ?x
        s0899d5u ?x ?y
      and the operator schemas are exactly those present in the domain file.
    - All relevant objects can be extracted from the initial state, goals and
      static facts supplied by the task. Grounding over that object set is
      adequate for estimating costs.
    - We use a relaxed model (no delete effects) to compute costs, which makes
      the heuristic inadmissible but efficient and informative for greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Extract goal facts from task.goals and static facts from task.static.
    - Collect the set of objects by scanning task.initial_state, task.goals, and task.static.
      Objects are all tokens after the predicate name in the fact strings.
    - Ground all operator schemas of the domain over the collected object set to
      form a list of concrete operators (preconditions and add effects).
      The operator schemas used (grounded here) are:
        * cb5rcmgi(x):
            pre:  yfm3frbn(x), a4ha3ujl(x), ug241uf7
            add:  jjtnwk3y(x)
        * dezigdo8(x):
            pre:  jjtnwk3y(x)
            add:  yfm3frbn(x), ug241uf7, a4ha3ujl(x)
        * ig1fewq3(s,q):
            pre:  yfm3frbn(q), jjtnwk3y(s)
            add:  ug241uf7, yfm3frbn(s), s0899d5u(s,q)
        * i57yke4t(s,q):
            pre:  s0899d5u(s,q), yfm3frbn(s), ug241uf7
            add:  jjtnwk3y(s), yfm3frbn(q)
    - Note: In the relaxed graph we ignore delete effects deliberately.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. From the current state (node.state) create a cost map for facts:
         - Facts true in the state get cost 0.
         - All other facts that appear either in goals or as possible adds from
           grounded operators are initialized with cost = infinity.
    2. Repeatedly consider each grounded operator:
         - If all its preconditions have finite costs, compute the action cost
           as 1 + sum(cost(precondition) for all preconditions).
         - For each fact in the operator's add effects, if the action cost is
           smaller than the current stored cost for that fact, update the fact cost.
    3. Iterate until a full pass produces no cost improvements (fixed point).
    4. The heuristic value is the sum of the computed costs for each goal fact.
         - If any goal fact remains at infinite cost (unreachable in the relaxed
           model), the heuristic returns a large finite value (here: 10**6) so
           the planner treats the state as very far from the goal but still
           keeps numbers finite.
    5. Return the sum rounded to an integer.
    """

    def __init__(self, task):
        # Goals as provided by task (set of fact strings)
        self.goals: Set[str] = set(task.goals)

        # Static facts (unused for this domain but stored)
        self.static: Set[str] = set(task.static)

        # Collect objects by parsing facts in initial_state, goals and static facts.
        objs: Set[str] = set()
        sources = []
        if hasattr(task, "initial_state") and task.initial_state is not None:
            sources.append(task.initial_state)
        sources.append(self.goals)
        sources.append(self.static)
        for factset in sources:
            for fact in factset:
                parts = get_parts(fact)
                # parts[0] is predicate, rest are objects (if any)
                for token in parts[1:]:
                    objs.add(token)
        self.objects = sorted(objs)

        # Ground operator list: each operator as dict with 'pre' and 'add' sets of fact strings.
        self.operators = []  # type: List[Dict[str, Set[str]]]

        # Helper to append unary operator grounded over all objects
        def add_unary_op(name, pred_pre_list, pred_add_list):
            for x in self.objects:
                pre = set()
                add = set()
                for p in pred_pre_list:
                    if p == "ug241uf7":
                        pre.add(make_fact("ug241uf7", []))
                    elif "%x" in p:
                        pred = p.replace("%x", "").strip()
                        pre.add(make_fact(pred, [x]))
                    else:
                        # direct predicate name (no args)
                        pre.add(make_fact(p, []))
                for a in pred_add_list:
                    if a == "ug241uf7":
                        add.add(make_fact("ug241uf7", []))
                    elif "%x" in a:
                        pred = a.replace("%x", "").strip()
                        add.add(make_fact(pred, [x]))
                    else:
                        add.add(make_fact(a, []))
                self.operators.append({"name": name.format(x=x), "pre": pre, "add": add})

        # Helper to append binary operator grounded over all object pairs
        def add_binary_op(name_template, pred_pre_list, pred_add_list):
            for s in self.objects:
                for q in self.objects:
                    pre = set()
                    add = set()
                    for p in pred_pre_list:
                        if p == "ug241uf7":
                            pre.add(make_fact("ug241uf7", []))
                        else:
                            # p contains %s and/or %q placeholders
                            pred_str = p.replace("%s", s).replace("%q", q)
                            parts = pred_str.split()
                            pred = parts[0]
                            args = parts[1:]
                            add_fact = make_fact(pred, args)
                            pre.add(add_fact)
                    for a in pred_add_list:
                        if a == "ug241uf7":
                            add.add(make_fact("ug241uf7", []))
                        else:
                            pred_str = a.replace("%s", s).replace("%q", q)
                            parts = pred_str.split()
                            pred = parts[0]
                            args = parts[1:]
                            add_fact = make_fact(pred, args)
                            add.add(add_fact)
                    self.operators.append(
                        {"name": name_template.format(s=s, q=q), "pre": pre, "add": add}
                    )

        # Define cb5rcmgi(x)
        # pre: yfm3frbn(x), a4ha3ujl(x), ug241uf7
        # add: jjtnwk3y(x)
        add_unary_op(
            "cb5rcmgi({x})",
            ["yfm3frbn %x", "a4ha3ujl %x", "ug241uf7"],
            ["jjtnwk3y %x"],
        )

        # Define dezigdo8(x)
        # pre: jjtnwk3y(x)
        # add: yfm3frbn(x), ug241uf7, a4ha3ujl(x)
        add_unary_op(
            "dezigdo8({x})",
            ["jjtnwk3y %x"],
            ["yfm3frbn %x", "ug241uf7", "a4ha3ujl %x"],
        )

        # Define ig1fewq3(s,q)
        # pre: yfm3frbn(q), jjtnwk3y(s)
        # add: ug241uf7, yfm3frbn(s), s0899d5u(s,q)
        add_binary_op(
            "ig1fewq3({s},{q})",
            ["yfm3frbn %q", "jjtnwk3y %s"],
            ["ug241uf7", "yfm3frbn %s", "s0899d5u %s %q"],
        )

        # Define i57yke4t(s,q)
        # pre: s0899d5u(s,q), yfm3frbn(s), ug241uf7
        # add: jjtnwk3y(s), yfm3frbn(q)
        add_binary_op(
            "i57yke4t({s},{q})",
            ["s0899d5u %s %q", "yfm3frbn %s", "ug241uf7"],
            ["jjtnwk3y %s", "yfm3frbn %q"],
        )

        # Precompute all facts that could be relevant: facts present in initial/goal/static
        # and all add effects from the grounded operators.
        self.all_possible_facts: Set[str] = set()
        if hasattr(task, "initial_state") and task.initial_state is not None:
            self.all_possible_facts |= set(task.initial_state)
        self.all_possible_facts |= set(self.goals)
        self.all_possible_facts |= set(self.static)
        for op in self.operators:
            self.all_possible_facts |= op["add"]

    def __call__(self, node):
        state: Set[str] = set(node.state)

        # If goal already reached, heuristic is zero (requirement 2)
        if self.goals <= state:
            return 0

        INF = float("inf")

        # Initialize cost for facts: 0 for facts in current state, inf otherwise.
        cost: Dict[str, float] = {fact: INF for fact in self.all_possible_facts}
        for f in state:
            if f in cost:
                cost[f] = 0.0

        # Iteratively relax until fixed point or until no improvement.
        changed = True
        # Upper bound on iterations: number of facts + 5 to ensure termination
        max_iters = len(self.all_possible_facts) + 5
        iters = 0
        while changed and iters < max_iters:
            changed = False
            iters += 1
            for op in self.operators:
                # Check if all preconditions are in our cost dict; if not, treat missing as INF
                prec_costs = []
                all_prec_known = True
                for p in op["pre"]:
                    if p not in cost:
                        # If a precondition is not in cost set, it could still be in the
                        # initial state (rare) or unreachable; treat as INF.
                        all_prec_known = False
                        prec_costs = None
                        break
                    else:
                        prec_costs.append(cost[p])
                        # If any precondition currently INF, action not yet applicable in relaxed model
                if not all_prec_known or any(c == INF for c in prec_costs):
                    continue
                action_cost = 1.0 + sum(prec_costs)
                # Update add effects
                for a in op["add"]:
                    if a not in cost:
                        # If an add effect was not in our fact universe, add it now.
                        cost[a] = INF
                    if action_cost < cost[a]:
                        cost[a] = action_cost
                        changed = True

        # Sum the costs of goal facts
        total = 0.0
        for g in self.goals:
            c = cost.get(g, INF)
            if c == INF:
                # Unreachable in the relaxed model: return a large finite value
                return 10 ** 6
            total += c

        # Return integer heuristic
        return int(math.ceil(total))