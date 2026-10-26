from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robby rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of children who are currently waiting and not yet served,
    and multiplies this count by a constant factor representing the estimated actions
    needed per child.

    # Assumptions:
    - For each unserved child, we need to perform a sequence of actions: make a sandwich,
      put it on a tray, and serve it.
    - Tray movement is implicitly considered as part of the serving process.
    - We assume sufficient bread, content, sandwiches, and trays are available to serve all children.
    - The heuristic is based on counting the number of unserved children and assuming a fixed cost per child.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base Heuristic class.
      Static facts are not explicitly used in this simplified heuristic but could be incorporated
      for more refined estimations (e.g., considering gluten allergies to differentiate action types).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal predicates: `(served ?c - child)` for all children.
    2. Iterate through the current state and count the number of children who are `waiting` but NOT `served`.
       - To do this, identify all children who are `waiting` at some place.
       - For each waiting child, check if they are also `served` in the current state.
       - Count the children who are `waiting` but not `served`.
    3. Estimate the number of actions per unserved child. A simplified estimate is 3 actions:
       - `make_sandwich` (or `make_sandwich_no_gluten`)
       - `put_on_tray`
       - `serve_sandwich` (or `serve_sandwich_no_gluten`)
    4. Multiply the count of unserved children by the estimated number of actions per child (e.g., 3 or a slightly higher value to account for tray movement if needed).
    5. Return the resulting value as the heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the childsnack heuristic."""
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """Estimate the number of actions to serve all waiting children."""
        state = node.state

        served_children = set()
        waiting_children = set()

        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
            elif match(fact, "waiting", "*", "*"):
                waiting_children.add(get_parts(fact)[1])

        unserved_waiting_children_count = 0
        for child in waiting_children:
            if child not in served_children:
                unserved_waiting_children_count += 1

        # Estimate actions per unserved child (make, put_on_tray, serve)
        actions_per_child = 3
        heuristic_value = unserved_waiting_children_count * actions_per_child

        # Heuristic is 0 if all goals are achieved
        if node.is_goal:
            return 0

        return heuristic_value
