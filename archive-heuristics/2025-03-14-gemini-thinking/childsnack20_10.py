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
    and multiplies this count by a constant factor (estimated actions per child).

    # Assumptions:
    - For each unserved child, a fixed number of actions are needed to make, put on tray, move tray (if needed), and serve a sandwich.
    - Bread and content are always available at the kitchen if they are initially available.
    - Trays are available and can be moved to the required places.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - It also extracts static information about children's allergies and waiting places to be used during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal children: Extract all children that need to be served from the goal description.
    2. Identify served children in the current state: Check the current state for 'served' predicates to find out which children are already served.
    3. Count unserved goal children: Calculate the number of goal children who are not yet served in the current state.
    4. Estimate actions per unserved child: Assume a constant number of actions (e.g., 4) are needed to serve each unserved child (make sandwich, put on tray, move tray, serve).
    5. Calculate total heuristic value: Multiply the count of unserved goal children by the estimated actions per child.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                self.goal_children.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to serve all children."""
        state = node.state
        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        unserved_goal_children_count = 0
        for child in self.goal_children:
            if child not in served_children:
                unserved_goal_children_count += 1

        # Estimate 4 actions per unserved child: make, put_on_tray, move_tray, serve
        estimated_actions_per_child = 4
        heuristic_value = unserved_goal_children_count * estimated_actions_per_child

        return heuristic_value
