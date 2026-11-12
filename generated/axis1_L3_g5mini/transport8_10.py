from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Remove surrounding parentheses from a PDDL fact string and split into parts.
    Example: "(arm-empty item2 item17)" -> ["arm-empty", "item2", "item17"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (a variation of Blocksworld).
    The primary goal facts in the provided instances are of the form (arm-empty <arm> <obj>).
    The heuristic estimates the number of actions required to make all goal arm-empty
    facts true by counting how many goal arm-empty facts are currently false and
    estimating a small per-goal repair cost based on the current state of each arm.

    Assumptions
    -----------
    - Goals are (arm-empty ARM OBJ) facts (as in the example instances).
    - Actions that change arm-empty facts move the second argument of an arm-empty
      predicate from one object to another (so an arm that is empty with a different
      object can be re-targeted in a small number of actions).
    - Static facts (such as 'holding' and 'on' relations) are not directly modified
      by this heuristic, but are extracted in case a future refinement is necessary.
    - The heuristic is non-admissible but aims to be informative and fast to compute.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - The set of goal facts (self.goals).
    - Static facts parsed into dictionaries/sets for quick lookup:
      - self.holding_pairs: set of tuples (hand, obj) for facts "(holding hand obj)".
      - self.on_pairs: set of tuples (top, bottom) for facts "(on top bottom)".
      - self.on_table_pairs: set of tuples (x, y) for facts "(on-table x y)".
      - self.clear_pairs: set of tuples (container, item) for facts "(clear container item)".
    These static structures are inexpensive to build and allow lightweight, domain-aware
    reasoning if needed by the heuristic.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (with node.state being a frozenset of PDDL fact strings),
    the heuristic proceeds as follows:

    1. If the goal is already reached (all goal facts are in the state), return 0.
       This guarantees the heuristic is 0 exactly for goal states.

    2. For each goal fact that is not satisfied in the current state:
       a. Parse the goal: it should be of the form (arm-empty ARM OBJ).
       b. Check whether there exists an arm-empty fact in the current state for the
          same ARM but with a different object (i.e., "(arm-empty ARM X)" present).
          - If such a fact exists, we assume the arm can be shifted to the desired OBJ
            with one action (cost = 1). This models simple operations like pickup/stack
            that move the arm-empty relation's second argument.
          - If no arm-empty fact exists for that ARM in the current state, we assume
            at least two actions are needed to create/move the arm-empty relation to
            the desired argument (cost = 2). This captures the idea that the arm must
            first be put into a usable configuration and then positioned at OBJ.
       c. Optionally, if domain static information indicates particularly difficult
          dependencies in the future, the per-goal cost could be increased; currently
          we keep it small to stay optimistic yet informative.

    3. The heuristic value is the sum of the per-goal estimated costs. This value is
       always finite (small integers), positive for non-goal states, and zero for
       goal states only.

    Rationale
    ---------
    - The heuristic is lightweight and focuses on the dominant goal predicate in the
      provided instances ((arm-empty ...)). It aims to guide greedy best-first search
      towards states where arms have the required arm-empty configuration.
    - By distinguishing between "same arm currently has some arm-empty" (cheap repair)
      and "arm has no arm-empty fact" (slightly more expensive), the heuristic encodes
      a small amount of domain knowledge while remaining fast to compute.
    """

    def __init__(self, task):
        # Store the goal facts as provided by the Task object.
        self.goals = task.goals

        # Parse static facts into convenient structures.
        static_facts = task.static if hasattr(task, "static") else frozenset()

        self.holding_pairs = set()
        self.on_pairs = set()
        self.on_table_pairs = set()
        self.clear_pairs = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "holding" and len(parts) >= 3:
                self.holding_pairs.add((parts[1], parts[2]))
            elif pred == "on" and len(parts) >= 3:
                self.on_pairs.add((parts[1], parts[2]))
            elif pred == "on-table" and len(parts) >= 3:
                self.on_table_pairs.add((parts[1], parts[2]))
            elif pred == "clear" and len(parts) >= 3:
                self.clear_pairs.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # If all goals are reached, heuristic is 0 (only for goal states).
        if self.goals <= state:
            return 0

        # Helper: find any arm-empty fact for a specific ARM in the current state.
        def arm_has_any_arm_empty(curr_state, arm):
            for fact in curr_state:
                parts = get_parts(fact)
                if parts and parts[0] == "arm-empty" and parts[1] == arm:
                    return True
            return False

        total_cost = 0
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            if pred != "arm-empty":
                # For goal predicates other than arm-empty, fall back to cost 1 per unmet goal.
                if goal not in state:
                    total_cost += 1
                continue

            # Expect format: (arm-empty ARM OBJ)
            if len(parts) < 3:
                # malformed goal fact: be conservative
                if goal not in state:
                    total_cost += 2
                continue

            arm = parts[1]
            obj = parts[2]

            if goal in state:
                # already satisfied, no cost
                continue

            # If the same arm currently has an arm-empty fact with a different object,
            # assume a single action can move it to the goal object.
            if arm_has_any_arm_empty(state, arm):
                total_cost += 1
            else:
                # Otherwise assume we'll need at least two actions to put the arm into
                # the required configuration.
                total_cost += 2

        # Ensure heuristic is positive for non-goal states (we returned 0 earlier for goals).
        return int(max(1, total_cost)) if total_cost > 0 else 1