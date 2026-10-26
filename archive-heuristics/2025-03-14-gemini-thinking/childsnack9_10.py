from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children with snacks.
    It counts the number of children who are currently waiting and not yet served, and estimates
    a fixed cost for serving each of them. This cost includes making a sandwich, putting it on a tray,
    and serving it at the child's location. Tray movement is not explicitly considered, assuming
    trays can be moved efficiently when needed.

    # Assumptions:
    - For each unserved child, we need to perform a sequence of actions: make sandwich, put on tray, and serve sandwich.
    - We simplify the problem by not explicitly accounting for tray movements. We assume trays can be moved to the required locations with minimal cost when needed.
    - Sufficient bread and content are available in the kitchen to make sandwiches for all children.
    - Sufficient sandwiches and trays are available.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the children who are currently waiting and have not yet been served.
    2. For each unserved child, estimate a fixed number of actions required to serve them.
       This estimate includes:
         - Making a sandwich (1 action).
         - Putting the sandwich on a tray (1 action).
         - Serving the sandwich to the child (1 action).
    3. Sum up the estimated actions for all unserved children.
    4. Return the total sum as the heuristic value. This value represents the estimated number of actions
       needed to reach a goal state from the current state.
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
        waiting_children_places = {}

        # Identify served children
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        # Identify waiting children and their places
        for fact in self.static_facts: # Use static facts as waiting is usually static in instances
            if match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                waiting_children_places[child] = place

        unserved_children_count = 0
        for child in waiting_children_places:
            if child not in served_children:
                unserved_children_count += 1

        # Estimate 3 actions per unserved child: make, put_on_tray, serve
        heuristic_value = unserved_children_count * 3

        # If all children are served, the heuristic value is 0 (goal state)
        all_goals_met = True
        for goal in self.goals:
            if goal not in state:
                all_goals_met = False
                break
        if all_goals_met:
            return 0

        return heuristic_value
