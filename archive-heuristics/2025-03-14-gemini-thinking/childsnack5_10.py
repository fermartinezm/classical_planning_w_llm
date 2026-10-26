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
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of children who are not yet served and multiplies it by a constant factor,
    and adds the number of distinct locations where unserved children are waiting to account for tray movements.

    # Assumptions
    - For each unserved child, we minimally need to make a sandwich, put it on a tray, move the tray to the child's location, and serve the sandwich.
    - We assume that bread and content are always available at the kitchen.
    - We consider that moving a tray to a location can potentially serve multiple children at the same location.

    # Heuristic Initialization
    No specific initialization is needed as the heuristic is computed based on the current state and goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are currently waiting to be served. This is done by checking the goal predicates `(served ?c)` against the current state.
    2. Identify the distinct locations where these unserved children are waiting. This accounts for the `move_tray` actions, assuming we need to move the tray to each distinct location at least once.
    3. Estimate the cost for each unserved child as 3 actions (making sandwich, putting on tray, serving sandwich).
    4. The heuristic value is calculated as (Number of unserved children * 3) + (Number of distinct waiting locations of unserved children).
    """

    def __init__(self, task):
        """Initialize the heuristic. No specific initialization needed for childsnacks domain."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        unserved_children = 0
        waiting_children_locations = set()

        goal_served_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                goal_served_children.add(get_parts(goal)[1])

        served_children_in_state = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children_in_state.add(get_parts(fact)[1])

        all_children_in_goal = set(goal_served_children)

        unserved_children_names = []
        for child_name in all_children_in_goal:
            if child_name not in served_children_in_state:
                unserved_children += 1
                unserved_children_names.append(child_name)

        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child_name = get_parts(fact)[1]
                location = get_parts(fact)[2]
                if child_name in unserved_children_names:
                    waiting_children_locations.add(location)

        return (unserved_children * 3) + len(waiting_children_locations)
