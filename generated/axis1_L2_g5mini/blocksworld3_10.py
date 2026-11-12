from heuristics.heuristic_base import Heuristic
from collections import defaultdict
from math import inf

def get_parts(fact):
    """Remove surrounding parentheses and split a fact into components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (dnbnd6kg). It computes an
    estimate of the number of actions required to reach the goal by performing
    a fast delete-relaxation (h_add-style) over grounded operators that are
    relevant to the current state and goals. The heuristic ignores delete
    effects (i.e., treats add-effects only) and propagates additive costs from
    facts true in the current state to the goal facts.

    Assumptions
    -----------
    - The representation of facts is the planner's internal string format,
      e.g., "(predicate arg1 arg2)".
    - All domain operators and predicates are as in the supplied domain file.
    - The cost of every grounded operator is 1 (each operator application
      contributes cost 1 in the relaxed model).
    - Delete effects are ignored (delete-relaxation), producing an optimistic
      but informative estimation useful to guide greedy best-first search.
    - Objects relevant for grounding are taken from the current state's facts
      and the goal facts. This keeps the grounding compact and efficient.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - The list of goal facts (self.goals).
    - A small description of all operator schemas in the domain (manually
      encoded based on the domain file). Each schema contains:
        * parameter arity (0, 1, or 2)
        * lists of precondition templates and add-effect templates
        * (delete effects are not used for the relaxation, so they are ignored)
    The operator schemas are encoded as lists of templates where a template is
    a tuple (predicate_string, positions), and positions indicate substitution
    of parameters (0 for first param, 1 for second, or None for the nullary zzar).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Parse the current state's facts into a set for fast membership tests.
       Extract the set of concrete objects that appear either in the state's
       facts or in the goal facts. These objects are the only ones used to
       ground operator schemas (keeps grounding small).

    2. Ground operator schemas for all combinations of relevant objects:
       - Nullary operators: instantiated once (zzar pre/effect).
       - Unary operators: instantiated for each object.
       - Binary operators: instantiated for each ordered pair of objects.

       For grounding we produce concrete precondition fact strings and
       concrete add-effect fact strings in the same string format as the
       planner's state representation.

    3. Initialize a cost dictionary for facts:
       - Facts present in the current state get cost 0.
       - All other facts start with cost = +infinity.

    4. Iteratively relax and propagate costs (h_add style):
       - For each grounded operator, compute its relaxed cost as:
           operator_cost = 1 + sum(cost(pre) for pre in operator_preconditions)
         (if some pre has cost infinity, operator_cost is infinity).
       - For each add-effect of the operator, set
           cost(effect) = min(cost(effect), operator_cost)
       - Repeat until no cost decreases in an iteration.

    5. Compute the heuristic value:
       - Sum the relaxed costs of each goal fact.
       - If any goal fact remains unreachable (cost = infinity), return a
         large finite penalty: 1000 + number_of_unreached_goals. This keeps the
         heuristic finite for solvable states while penalizing unreachable
         goals in the relaxed model.

    6. Return the computed sum as the heuristic estimate. The heuristic is
       0 exactly when all goals are present in the current state (cost 0).
    """

    def __init__(self, task):
        # Store goals as a list of goal fact strings
        self.goals = list(task.goals)

        # Manually encoded operator schemas based on the provided domain.
        # Each operator schema is a dict:
        #  - "arity": number of parameters (0,1,2)
        #  - "pre": list of precondition templates
        #  - "add": list of add-effect templates
        # Templates use integers 0 and 1 to indicate substitution positions
        # from the parameters. Use None for nullary predicate zzar.
        #
        # The templates correspond to fact strings like "(predicate {0})"
        # where {0} and {1} will be replaced by actual object names.
        self.operator_schemas = [
            # inz39ca6 (?ff153 ?p3nj1)
            {
                "name": "inz39ca6",
                "arity": 2,
                "pre_templates": [
                    "(bclhkf4r {0})",
                    "(zzar8qlk)",
                    "(gaijkwef {0} {1})",
                ],
                "add_templates": [
                    "(mdawkki4 {0})",
                    # (not zzar8qlk) is a delete in the real domain -> ignored
                    "(bclhkf4r {1})",
                    # delete effects not included in relaxation
                ],
            },
            # fqixw4x9 (?ff153)
            {
                "name": "fqixw4x9",
                "arity": 1,
                "pre_templates": [
                    "(zzar8qlk)",
                    "(bclhkf4r {0})",
                    "(chxwa5j4 {0})",
                ],
                "add_templates": [
                    "(mdawkki4 {0})",
                ],
            },
            # mri4ci1h (?ff153)
            {
                "name": "mri4ci1h",
                "arity": 1,
                "pre_templates": [
                    "(mdawkki4 {0})",
                ],
                "add_templates": [
                    "(zzar8qlk)",
                    "(bclhkf4r {0})",
                    "(chxwa5j4 {0})",
                ],
            },
            # j2i3gtal (?ff153 ?p3nj1)
            {
                "name": "j2i3gtal",
                "arity": 2,
                "pre_templates": [
                    "(bclhkf4r {1})",
                    "(mdawkki4 {0})",
                ],
                "add_templates": [
                    "(zzar8qlk)",
                    "(bclhkf4r {0})",
                    "(gaijkwef {0} {1})",
                ],
            },
        ]

    def __call__(self, node):
        state = set(node.state)  # frozenset -> set for possible membership ops

        # If all goals already true, heuristic is 0.
        if all(goal in state for goal in self.goals):
            return 0

        # Extract relevant objects from current state and goals
        objects = set()
        for fact in list(state) + self.goals:
            parts = get_parts(fact)
            # skip the predicate name (parts[0]) and collect the rest tokens
            for token in parts[1:]:
                objects.add(token)

        # Build grounded operators (relaxed: only add-effects are considered)
        grounded_ops = []
        objs_list = list(objects)

        # Helper to instantiate templates
        def instantiate(template, binding):
            # binding is a tuple with length equal to arity (0,1,2)
            if "{0}" in template or "{1}" in template:
                return template.format(*binding)
            else:
                return template  # e.g., "(zzar8qlk)"

        for schema in self.operator_schemas:
            arity = schema["arity"]
            if arity == 0:
                bindings = [tuple()]
            elif arity == 1:
                bindings = [(o,) for o in objs_list]
            else:  # arity == 2
                # ordered pairs including same object allowed
                bindings = [(o1, o2) for o1 in objs_list for o2 in objs_list]

            for binding in bindings:
                preconds = {instantiate(t, binding) for t in schema["pre_templates"]}
                adds = {instantiate(t, binding) for t in schema["add_templates"]}
                grounded_ops.append((preconds, adds))

        # Initialize fact costs: facts in state -> 0, others -> inf
        cost = defaultdict(lambda: inf)
        # Seed costs with facts present in current state (0)
        for fact in state:
            cost[fact] = 0

        # Important: also ensure that the nullary zzar predicate (if not in state)
        # is present in cost dict as inf, so operators mentioning it see a cost.
        # (defaultdict already returns inf)

        # Propagate costs in the relaxed model (h_add): iterate until fixpoint
        changed = True
        # To keep the loop bounded in degenerate cases, set a generous iteration cap
        iteration_cap = 5000
        iterations = 0
        while changed and iterations < iteration_cap:
            iterations += 1
            changed = False
            for preconds, adds in grounded_ops:
                # relaxed operator cost is 1 + sum(costs of preconditions)
                sum_pre = 0
                unreachable = False
                for p in preconds:
                    c = cost[p]
                    if c == inf:
                        unreachable = True
                        break
                    sum_pre += c
                if unreachable:
                    continue
                op_cost = 1 + sum_pre
                # update add-effects costs
                for a in adds:
                    if op_cost < cost[a]:
                        cost[a] = op_cost
                        changed = True

        # Sum costs of goal facts
        total = 0
        unreachable_goals = 0
        for g in self.goals:
            c = cost[g]
            if c == inf:
                unreachable_goals += 1
            else:
                total += c

        if unreachable_goals > 0:
            # Return a large finite penalty if some goals cannot be reached in the
            # delete-relaxed model. The penalty depends on how many goals are
            # unreachable to still provide some ordering signal.
            return 1000 + 50 * unreachable_goals

        # The h_add value is the sum of individual goal costs. Return as integer.
        return int(total)