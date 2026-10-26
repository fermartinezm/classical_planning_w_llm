from heuristics.heuristic_base import Heuristic

def get_predicate(fact_string):
    """Extracts the predicate name from a PDDL fact string."""
    # Example: '(on b1 b2)' -> 'on'
    # Example: '(arm-empty)' -> 'arm-empty'
    parts = fact_string[1:-1].split()
    return parts[0] if parts else ""

class blocksworldHeuristic(Heuristic):
    """
    blocksworldHeuristic

    Summary:
    Estimates the cost to reach the goal state in Blocksworld by counting discrepancies
    between the current state and the goal state. It penalizes current facts related
    to block positions ('on', 'on-table') and arm status ('holding') that are not
    part of the goal, and also penalizes goal facts related to block positions
    ('on', 'on-table') and clearness ('clear') that are not true in the current state.

    Assumptions:
    - The heuristic is designed for the standard Blocksworld domain with predicates
      'clear', 'on-table', 'arm-empty', 'holding', and 'on'.
    - Goal states consist of conjunctions of 'on', 'on-table', and 'clear' predicates.
    - The heuristic is non-admissible and intended for greedy best-first search.
    - Static facts are not used by this heuristic (as the Blocksworld domain typically has no static facts).

    Heuristic Initialization:
    The constructor `__init__` stores the set of goal facts from the planning task
    for efficient lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic value for a given state is computed as follows:
    1. Initialize the heuristic value `h` to 0.
    2. Iterate through each fact currently true in the state:
       - Extract the predicate name using `get_predicate`.
       - If the predicate is 'on', 'on-table', or 'holding':
         - Check if this exact fact string is present in the set of goal facts.
         - If the fact is *not* a goal fact, increment `h` by 1. This penalizes
           incorrect block placements, blocks on the table that shouldn't be,
           and holding a block (as the arm should be empty in the goal).
    3. Iterate through each fact defined in the goal:
       - Extract the predicate name using `get_predicate`.
       - If the predicate is 'on', 'on-table', or 'clear':
         - Check if this exact goal fact string is currently true in the state.
         - If the goal fact is *not* true in the current state, increment `h` by 1.
           This penalizes missing required block placements, missing required
           blocks on the table, and missing required clear blocks.
    4. The final value of `h` is the heuristic estimate.

    This heuristic counts the total number of "wrong" structural facts in the current
    state and "missing" required structural facts from the goal state. While simple,
    it provides a measure of how far the current state is from the goal in terms
    of the desired block configuration.
    """
    def __init__(self, task):
        # Store the goal facts for quick lookup
        self.goals = task.goals # frozenset of goal fact strings
        # Blocksworld typically has no static facts, so task.static is not used here.

    def __call__(self, node):
        # Get the current state from the node
        state = node.state # frozenset of current fact strings

        # If the current state is the goal state, the heuristic is 0
        if state == self.goals:
            return 0

        h = 0

        # Penalty for current facts that are not goal facts
        # We focus on predicates that define the structure or arm status
        for fact in state:
            predicate = get_predicate(fact)
            # Consider 'on', 'on-table', and 'holding' facts in the current state
            if predicate in ['on', 'on-table', 'holding']:
                # If this fact is not required in the goal state, penalize it
                if fact not in self.goals:
                    h += 1

        # Penalty for goal facts that are not currently true
        # We focus on predicates that are typically part of the goal definition
        for goal_fact in self.goals:
            predicate = get_predicate(goal_fact)
            # Consider 'on', 'on-table', and 'clear' goal facts
            if predicate in ['on', 'on-table', 'clear']:
                 # If this goal fact is not true in the current state, penalize it
                 if goal_fact not in state:
                    h += 1

        return h
