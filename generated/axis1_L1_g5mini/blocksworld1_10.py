from itertools import product
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string, without surrounding parentheses.
    Example: "(s0899d5u b1 b2)" -> ["s0899d5u", "b1", "b2"]
    """
    return fact[1:-1].split()


def make_fact(predicate, *args):
    """Construct a fact string from predicate and arguments."""
    if args:
        return "({} {})".format(predicate, " ".join(args))
    else:
        return "({})".format(predicate)


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A relaxed planning (h_add-style) domain-dependent heuristic tailored to the
    secret_domain (z1spq3ra). It computes the estimated number of actions
    needed to reach the goal by constructing a relaxed reachability graph:
    - delete effects are ignored (monotonic relaxation),
    - action cost is 1 plus the sum of costs of its preconditions (h_add).
    The heuristic returns the sum of the relaxed costs of the individual goal
    facts. This is not admissible but is informative and efficiently computable.

    Assumptions
    -----------
    - Facts are PDDL strings like "(predicate arg1 arg2)". Zero-argument
      predicates appear as "(predicate)".
    - The task passed to the constructor contains initial/static/goal facts and
      these facts mention all relevant objects (we collect objects from them).
    - Operators in the domain are exactly the four schemas from the provided
      domain file (cb5rcmgi, dezigdo8, ig1fewq3, i57yke4t). The heuristic
      instantiates these schemas over the objects found in the task to build
      grounded (relaxed) operators.
    - Because deletes are ignored, the heuristic may underestimate interactions
      but gives a useful guidance for greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Store the goal facts from the given task (self.goals).
    - Extract static facts (task.static) for possible use (not needed here but
      recorded).
    - Collect the set of objects by scanning task.initial_state, task.goals and
      task.static for argument tokens of facts.
    - Ground the four domain operator schemas over the collected objects to
      produce a list of grounded operator descriptions. Each grounded operator
      is represented as a tuple (preconds_set, add_effects_set).
      (Delete effects are intentionally ignored in the relaxation.)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state we perform the following steps:

    1. Quick goal check:
       - If all goal facts are already true in the node.state, return 0.

    2. Initialization of relaxed costs:
       - Set cost 0 for every fact that is currently true in the node.state.
       - All other facts start as "unknown" (no cost assigned).

    3. Iterative relaxed reachability (h_add):
       - Repeatedly scan all grounded operators. For an operator whose every
         precondition has an assigned cost, compute the operator cost as:
            op_cost = 1 + sum(cost(pre) for pre in preconds)
         (This is the h_add action cost -- an action contributes the sum of
         its precondition costs plus one for itself.)
       - For every fact that the operator adds, if op_cost is strictly less
         than the current recorded cost for that fact (or if the fact had no
         cost yet), set the fact's cost to op_cost and mark progress.
       - Continue until no new costs are discovered or all goal facts have
         assigned costs.

    4. Goal aggregation:
       - The heuristic value is the sum of the computed costs for each goal
         fact that is not already true.
       - If a goal fact never receives a cost (unreachable in the relaxation),
         return a conservative large finite cost derived from the number of
         objects (so the heuristic remains finite). This is unlikely for
         solvable instances since the domain's operators are constructive in
         the relaxation.

    Rationale / Effectiveness
    -------------------------
    - By using an h_add style cost (sum over precondition costs), the heuristic
      captures resource accumulation required to enable actions (e.g., many
      operators require the global predicate (ug241uf7) or particular local
      facts). Ignoring deletes makes the computation efficient and monotonic,
      while still informative enough to guide greedy best-first search in this
      domain where many actions shuffle binary relations and single-object
      flags.
    """

    def __init__(self, task):
        # Goals (set of fact strings) that must hold in a goal state.
        self.goals = set(task.goals)

        # Keep static facts for completeness (not used in the current heuristic,
        # but recorded for possible extension).
        self.static = set(task.static)

        # Collect objects by scanning facts from initial, goal and static sets.
        objects = set()
        sources = []
        # Task may expose initial_state as attribute; include it if present.
        if hasattr(task, "initial_state"):
            sources.append(task.initial_state)
        sources.append(self.goals)
        sources.append(self.static)

        for source in sources:
            for f in source:
                parts = get_parts(f)
                # parts[0] is predicate name; remaining are arguments.
                for arg in parts[1:]:
                    objects.add(arg)
        # Frozen set of objects for later grounding.
        self.objects = sorted(objects)

        # Precompute all grounded (relaxed) operators by instantiating the four schemas.
        # Each grounded operator is represented as a tuple (preconds_set, add_effects_set).
        self.grounded_ops = []
        # Helper for zero-arg predicate string
        ug_fact = make_fact("ug241uf7")

        # cb5rcmgi ?x
        for o in self.objects:
            pre = {make_fact("yfm3frbn", o), make_fact("a4ha3ujl", o), ug_fact}
            add = {make_fact("jjtnwk3y", o)}
            self.grounded_ops.append((frozenset(pre), frozenset(add)))

        # dezigdo8 ?x
        for o in self.objects:
            pre = {make_fact("jjtnwk3y", o)}
            add = {
                make_fact("yfm3frbn", o),
                ug_fact,
                make_fact("a4ha3ujl", o),
            }
            self.grounded_ops.append((frozenset(pre), frozenset(add)))

        # ig1fewq3 ?sei ?q9qne
        for sei, q in product(self.objects, repeat=2):
            pre = {make_fact("yfm3frbn", q), make_fact("jjtnwk3y", sei)}
            add = {
                ug_fact,
                make_fact("yfm3frbn", sei),
                make_fact("s0899d5u", sei, q),
            }
            self.grounded_ops.append((frozenset(pre), frozenset(add)))

        # i57yke4t ?sei ?q9qne
        for sei, q in product(self.objects, repeat=2):
            pre = {
                make_fact("s0899d5u", sei, q),
                make_fact("yfm3frbn", sei),
                ug_fact,
            }
            add = {make_fact("jjtnwk3y", sei), make_fact("yfm3frbn", q)}
            self.grounded_ops.append((frozenset(pre), frozenset(add)))

    def __call__(self, node):
        """
        Compute the h_add-style relaxed heuristic value for the given node.

        Returns an integer estimate (sum of individual goal fact costs under
        relaxed planning). Guaranteed to be 0 on goal states and finite for
        solvable instances (returns a conservative large finite value if some
        goal is unreachable in the relaxation).
        """
        state = set(node.state)

        # 1) If node already satisfies all goals, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Initialize costs: facts true in state have cost 0.
        fact_cost = {f: 0 for f in state}

        # 3) Iteratively propagate costs using grounded (relaxed) operators.
        changed = True
        # We will loop until a fixed point. For safety, cap number of iterations
        # to avoid pathological infinite loops (should not occur).
        max_iterations = 10000
        iters = 0

        while changed and iters < max_iterations:
            iters += 1
            changed = False
            # Try each grounded operator
            for preconds, adds in self.grounded_ops:
                # Check if all preconditions have assigned costs (i.e., reachable).
                pre_costs = []
                all_known = True
                for p in preconds:
                    if p in fact_cost:
                        pre_costs.append(fact_cost[p])
                    else:
                        all_known = False
                        break
                if not all_known:
                    continue
                # h_add action cost: 1 + sum of precondition costs
                action_cost = 1 + sum(pre_costs)
                # For each add effect, update its cost if improved (or not set).
                for a in adds:
                    old_c = fact_cost.get(a)
                    if old_c is None or action_cost < old_c:
                        fact_cost[a] = action_cost
                        changed = True

            # Early exit if we've assigned costs to all goal facts
            if all(g in fact_cost for g in self.goals):
                break

        # 4) Aggregate: sum costs for goal facts that are not yet true in the state.
        total = 0
        unreachable = False
        for g in self.goals:
            if g in state:
                continue
            c = fact_cost.get(g)
            if c is None:
                unreachable = True
            else:
                total += c

        if not unreachable:
            return int(total)

        # If at least one goal is unreachable in the relaxation, return a
        # conservative finite estimate. Use number of objects times a constant
        # as a bound to keep the value finite but large enough to deprioritize
        # clearly infeasible states in greedy search.
        return max(1, 10 * max(1, len(self.objects)))