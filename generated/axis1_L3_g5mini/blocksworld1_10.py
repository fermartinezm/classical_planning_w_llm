from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into its parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether `fact` matches a pattern given by `args` (supports '*' wildcards).
    Example: match("(at a b)", "at", "*", "b") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "transport" (secret_domain) domain.
    The heuristic estimates how many actions are needed to reach the goal by
    analyzing which goal facts are missing and giving each missing fact a small
    cost estimate that reflects the typical number of operators required to
    establish that fact in this domain.

    Assumptions
    -----------
    - Facts are represented as strings like "(at obj1 obj2)".
    - Goals are conjunctions of facts (provided via task.goals).
    - The domain's important predicates are:
        - road ?x
        - capacity-predecessor ?x
        - capacity    (0-ary)
        - in ?x
        - at ?x ?y
    - The heuristic is not required to be admissible. It should be efficiently
      computable and reasonably informative to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    - The constructor stores the set of goal facts (self.goals).
    - Static facts from the task are kept if present (self.static) for future
      extensions; in the provided problems static is often empty.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    The heuristic works as follows for a given state (node.state):

    1. If all goal facts are already true in the state, return 0.

    2. Otherwise, compute which goal facts are missing:
       missing_goals = goal_facts - state_facts

    3. For each missing goal fact, apply a small domain-informed cost model:
       - (at A B): To create an "at" fact the domain provides the "drop" operator
         which directly adds (at A B) but needs (in A) and (road B) as preconditions.
         - If both preconditions are currently present -> cost = 1 (only drop required).
         - If exactly one precondition is present -> cost = 2 (one action to obtain
           the missing precondition, plus drop).
         - If neither precondition is present -> cost = 3 (create an "in A"
           (via drive/drive-2) and create/relocate a "road B" (via pick-up/drop),
           then drop).
       - (road X): Road facts are frequently created by pick-up or drop.
         - If (in X) holds now, pick-up would immediately produce (road X) -> cost = 1.
         - Otherwise -> cost = 2 (assume one intermediate action + pick-up/drop).
       - (capacity-predecessor X): This fact is produced by pick-up on X.
         - If (in X) currently holds -> cost = 1.
         - Otherwise -> cost = 2 (one action to get X into the vehicle, then pick-up).
       - (capacity): 0-ary predicate. If missing -> cost = 1 (a single action such
         as pick-up or drop often establishes capacity).
       - Any other missing predicate -> default cost = 2.

    4. Sum the costs for all missing goal facts to produce the heuristic value.

    Rationale
    ---------
    - The numeric costs are small integers (1–3) chosen to be cheap to compute
      while incorporating basic knowledge of operator preconditions and effects.
    - This heuristic is purposely non-admissible but informative: it favors
      states where many goal facts are already true or where missing goals can
      be achieved with a single applicable operator.
    - The heuristic is always finite for solvable states and returns 0 exactly
      for goal states.
    """

    def __init__(self, task):
        """
        Store goal facts and static facts for potential use.

        Parameters
        ----------
        task : Task
            The planning task (provides .goals and .static).
        """
        # Goals are a set/frozenset of fact strings that must hold in the goal.
        self.goals = task.goals
        # Static facts (may be empty for this domain) are recorded for completeness.
        self.static = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for `node`.

        Parameters
        ----------
        node : search.Node
            The search node whose .state is a frozenset of fact strings.

        Returns
        -------
        int
            Estimated number of actions to reach the goal (0 iff node.state meets goals).
        """
        state = node.state

        # If all goals are satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        missing = self.goals - state
        total_cost = 0

        # Helpers to check presence of simple facts in the state.
        def has_fact_string(fact_str):
            return fact_str in state

        for fact in missing:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                in_obj = has_fact_string(f"(in {obj})")
                road_loc = has_fact_string(f"(road {loc})")

                if in_obj and road_loc:
                    cost = 1
                elif in_obj or road_loc:
                    cost = 2
                else:
                    cost = 3

                total_cost += cost

            elif predicate == "road" and len(parts) == 2:
                x = parts[1]
                # If x is currently inside, pick-up will create (road x) immediately.
                if has_fact_string(f"(in {x})"):
                    total_cost += 1
                else:
                    total_cost += 2

            elif predicate == "capacity-predecessor" and len(parts) == 2:
                x = parts[1]
                # This is typically produced by pick-up which needs (in x).
                if has_fact_string(f"(in {x})"):
                    total_cost += 1
                else:
                    total_cost += 2

            elif predicate == "capacity" and len(parts) == 1:
                # capacity is 0-ary; many operators produce it (pick-up, drop).
                if has_fact_string("(capacity)"):
                    # Though missing, defensive: if present, shouldn't be here; skip.
                    continue
                total_cost += 1

            else:
                # Generic fallback for unexpected predicates.
                total_cost += 2

        # Ensure heuristic is integer and finite.
        return int(total_cost)