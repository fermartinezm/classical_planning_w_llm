from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def make_fact(predicate, *args):
    """Construct a fact string from predicate and arguments in the same format used by the planner."""
    return f"({predicate}{(' ' + ' '.join(args)) if args else ''})"


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'transport' (secret_domain) domain.

    The heuristic estimates the number of actions required to satisfy the goal
    conjunction by counting unsatisfied goal facts and estimating how many
    operators are minimally needed to achieve each. It is not admissible, but
    it aims to provide a tight and fast-to-compute estimate to guide greedy
    best-first search.

    Assumptions
    -----------
    - Facts are strings in the format '(predicate arg1 arg2 ...)'.
    - Relevant predicates in this domain are:
        - road X
        - capacity-predecessor X
        - capacity
        - in X
        - at X Y
    - We assume that creating an 'at' fact requires (at least) bringing the
      first argument into an 'in' state and performing a 'drop' that requires
      the second argument to have 'road'. We count these prerequisites as
      separate actions when missing.
    - We accept overestimates for simplicity and speed (the heuristic is not
      required to be admissible).

    Heuristic Initialization
    ------------------------
    - The constructor stores:
      - the list of goal facts to check (task.goals),
      - the static facts (task.static) for potential use (kept as a set).
    - No heavy preprocessing is needed because the domain has no complex static
      structure to exploit.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state (node.state), the heuristic follows these rules:

    1. If all goal facts are satisfied in the state, return 0.

    2. For each goal fact that is not currently true, estimate the minimal
       number of actions required to make it true (under optimistic but
       realistic assumptions about shared effects). The estimates per predicate:

       - at A B:
         * A direct 'drop' action is needed to create (at A B): cost += 1
         * 'drop' requires A to be 'in' (precondition). If '(in A)' is missing,
           we estimate one extra action to make A 'in' (e.g., a 'drive' or
           'drive-2'): cost += 1
         * 'drop' also requires '(road B)'. If '(road B)' is missing, add 1.
         Overall, unsatisfied '(at A B)' contributes between 1 and 3 actions.

       - road X:
         * If '(in X)' holds, a single 'pick-up' can create '(road X)': cost = 1
         * Else, if there exists some fact '(at Y X)' and both '(road Y)' and
           '(capacity)' hold in the state, a single 'drive-2' may add '(road X)': cost = 1
         * Otherwise, we estimate it will take 2 actions to create '(road X)'
           (e.g., move to make 'in X' and then 'pick-up' or a short sequence that
           yields road): cost = 2

       - capacity-predecessor X:
         * This fact is added by 'pick-up' and requires '(in X)'. If '(in X)'
           is true we count 1 (the pick-up). Otherwise we estimate 2 actions
           (one to get X into 'in' and one to pick it up): cost = 2

       - capacity:
         * If missing, at least one action that provides capacity (pick-up or drop)
           is required: cost = 1

       - Any other predicate (unexpected):
         * Conservative default cost = 1

    3. Sum the per-goal estimates. This sum is the heuristic value returned.
       The estimator may double-count some effects that one action could
       achieve for multiple goals, but this is acceptable for a fast,
       informative heuristic guiding greedy search.

    Notes on properties
    -------------------
    - The heuristic is 0 exactly for goal states (when all goal facts are present).
    - It always returns a finite integer for states reachable in the domain (it
      uses small constant upper bounds for missing prerequisites).
    - Static facts (if present) are stored and can be used; in this domain they
      are typically empty.
    """

    def __init__(self, task):
        """Store goals and static facts for use by the heuristic."""
        # Goals are stored as the original fact strings (e.g., "(at a b)").
        self.goals = task.goals
        # Store static facts for potential use (often empty in this domain).
        self.static = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Parameters
        ----------
        node : search node
            The planner node whose .state attribute is a frozenset of fact strings.

        Returns
        -------
        int
            Estimated number of actions to reach a state that satisfies all goals.
        """
        state = node.state  # frozenset of facts

        # Quick check: if all goals satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper to test presence of a fact in the current state.
        def has_fact(predicate, *args):
            return make_fact(predicate, *args) in state

        # Helper to find facts of the form '(at X Y)' in the current state.
        # Returns list of (X, Y) pairs.
        at_facts = []
        for f in state:
            parts = get_parts(f)
            if parts and parts[0] == "at" and len(parts) >= 3:
                at_facts.append((parts[1], parts[2]))

        total_cost = 0
        for goal in self.goals:
            if goal in state:
                continue  # goal already satisfied

            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]

            if pred == "at" and len(args) == 2:
                a, b = args
                # 1 for the drop itself
                cost = 1
                # ensure 'in a' to be able to drop a into b
                if not has_fact("in", a):
                    cost += 1
                # ensure 'road b' as required by drop's precondition
                if not has_fact("road", b):
                    cost += 1
                total_cost += cost

            elif pred == "road" and len(args) == 1:
                x = args[0]
                # If in x, a single pick-up can create road x
                if has_fact("in", x):
                    total_cost += 1
                else:
                    # If there exists at Y X and Y currently has road and capacity,
                    # drive-2 could set road X in one action.
                    fast_possible = False
                    for (y, y_target) in at_facts:
                        if y_target == x and has_fact("road", y) and has_fact("capacity"):
                            fast_possible = True
                            break
                    total_cost += 1 if fast_possible else 2

            elif pred == "capacity-predecessor" and len(args) == 1:
                x = args[0]
                # pick-up creates capacity-predecessor but requires 'in x'
                if has_fact("in", x):
                    total_cost += 1
                else:
                    total_cost += 2

            elif pred == "capacity" and len(args) == 0:
                # Any pick-up or drop can create capacity; count 1 if missing.
                total_cost += 1

            else:
                # Fallback conservative estimate for any unexpected goal predicate.
                total_cost += 1

        # Ensure heuristic is integer and finite.
        return int(total_cost)