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
    This heuristic estimates the number of actions required to serve all waiting children with snacks.
    It counts the number of children who are currently waiting and not yet served, and multiplies this count by a constant factor (currently 3) to estimate the total actions needed.
    This factor represents a rough estimate of the actions needed per child (make sandwich, put on tray, serve).

    # Assumptions:
    - For each unserved child, at least a 'make sandwich', 'put on tray', and 'serve sandwich' action will be needed.
    - Tray movements are implicitly considered in the constant factor.
    - Resource availability (bread, content, sandwiches, trays) is not explicitly checked in the heuristic, assuming they are generally available if needed.

    # Heuristic Initialization
    - The heuristic does not require any specific initialization beyond the standard heuristic base class.
    - It implicitly uses the goal conditions to identify which children need to be served.
    - Static facts are not explicitly used in this simplified heuristic, but could be used for more refined versions (e.g., to consider gluten allergies).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal predicates: These are the `(served ?c - child)` predicates that define when a child is served.
    2. Identify the children who are in the goal but are not yet served in the current state.
    3. Count the number of these unserved children.
    4. Multiply the count of unserved children by a constant factor (e.g., 3) to estimate the total number of actions required.
       This factor is a rough estimate of the minimum actions needed per child to make, prepare, and serve a snack.
    """

    def __init__(self, task):
        """Initialize the heuristic. No specific initialization needed for this heuristic."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state from the current state."""
        state = node.state
        unserved_children_count = 0

        goal_children = set()
        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                goal_children.add(get_parts(goal_fact)[1])

        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        waiting_children_places = {}
        for fact in self.static: # Use static facts to find waiting children and their places
            if match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                waiting_children_places[child] = place

        for child in goal_children:
            if child not in served_children and child in waiting_children_places: # Consider only waiting children in goal
                unserved_children_count += 1

        return unserved_children_count * 3 # Multiply by 3 as a rough estimate of actions per child
