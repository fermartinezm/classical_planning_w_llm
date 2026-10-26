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


class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of children who are currently waiting and not yet served,
    and multiplies this count by a constant factor representing the estimated actions
    needed per child. This heuristic is based on the assumption that serving each child
    requires a roughly constant number of actions (making sandwich, putting on tray, serving).

    # Assumptions:
    - Serving each child requires a fixed number of actions.
    - The heuristic does not explicitly consider resource availability (bread, content, trays)
      or tray movements, simplifying the estimation for efficiency.
    - It assumes that for each waiting and unserved child, a sequence of actions will be needed.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base heuristic class.
    - The heuristic relies on the goal conditions (served children) and the current state
      to estimate the remaining work.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the children who are currently waiting at some place.
    2. Identify the children who are already served.
    3. Count the number of children who are waiting but not yet served.
    4. Estimate a fixed number of actions required to serve each of these children (e.g., 3 actions: make sandwich, put on tray, serve).
    5. Multiply the count of unserved waiting children by the estimated number of actions per child to get the heuristic value.
    6. If all children are served (goal state), the heuristic value is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state

        served_children = set()
        waiting_children = set()
        waiting_places = {}

        for fact in state:
            if match(fact, "served", "*"):
                child = get_parts(fact)[1]
                served_children.add(child)
            if match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                waiting_children.add(child)
                waiting_places[child] = place

        goal_served_children = set()
        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                goal_served_children.add(get_parts(goal_fact)[1])

        unserved_waiting_children_in_goal = 0
        for child in goal_served_children:
            if child not in served_children:
                unserved_waiting_children_in_goal += 1

        if unserved_waiting_children_in_goal == 0:
            return 0

        # Estimate actions per unserved child. Let's assume 3 actions: make, put_on_tray, serve.
        actions_per_child = 3
        heuristic_value = unserved_waiting_children_in_goal * actions_per_child

        return heuristic_value
