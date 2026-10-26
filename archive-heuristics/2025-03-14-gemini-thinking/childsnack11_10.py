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
    It counts the number of children who are not yet served and multiplies it by a constant factor (4),
    representing a rough estimate of the actions needed per child (make sandwich, put on tray, move tray, serve).

    # Assumptions
    - For each unserved child, we minimally need to perform a sequence of actions: making a sandwich,
      putting it on a tray, moving the tray to the child's location, and serving the sandwich.
    - We assume that resources (bread, content, trays) are always available to make sandwiches for all children.
    - We do not explicitly track the availability of resources or the current locations of trays and sandwiches
      in detail within the heuristic calculation for simplicity and efficiency.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal predicates, specifically the `served` predicates for each child.
    2. Iterate through the goal predicates to find all children that need to be served.
    3. For each child, check if the `served` predicate is present in the current state.
    4. Count the number of children for whom the `served` predicate is NOT present in the current state.
    5. Multiply this count by a constant factor (4, as a rough estimate of actions per child) to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        super().__init__(task)
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        unserved_children_count = 0
        goal_children = set()

        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                goal_children.add(get_parts(goal_fact)[1])

        for child in goal_children:
            served_predicate = f'(served {child})'
            if served_predicate not in state:
                unserved_children_count += 1

        # Simple heuristic: assume 4 actions needed per unserved child
        return unserved_children_count * 4
