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
    It counts the number of children who are currently waiting and not yet served,
    and multiplies this count by a constant factor, representing the estimated actions
    needed per child. This heuristic assumes a fixed cost for serving each child,
    regardless of the current state of sandwich preparation or tray location.

    # Assumptions:
    - For each unserved child, a fixed number of actions are needed to make a sandwich,
      put it on a tray, and serve it.
    - Tray movements are implicitly considered in the fixed cost per child.
    - The heuristic does not differentiate between gluten-free and regular sandwiches
      or the availability of ingredients. It provides a simplified estimate.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task.
      Specifically, it identifies the 'served' predicates in the goal to determine
      which children need to be served.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal children: Extract all children that need to be 'served'
       from the goal description.
    2. Count unserved goal children: Iterate through the goal children and check
       for each child if the '(served child)' fact is present in the current state.
       Count the number of goal children for whom the '(served child)' fact is NOT
       in the current state. These are the unserved children.
    3. Estimate cost: Multiply the count of unserved children by a constant factor
       (e.g., 3). This factor represents a rough estimate of the number of actions
       required to serve each child (make sandwich, put on tray, serve).
    4. Return the total estimated cost. This value represents the heuristic estimate
       for the given state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        Specifically, identify the children that need to be served from the goal.
        """
        self.goals = task.goals
        self.goal_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                self.goal_children.add(get_parts(goal)[1])

    def __call__(self, node):
        """
        Estimate the number of actions needed to serve all children in the goal.
        This heuristic counts the number of children who are in the goal but not yet served
        in the current state and multiplies it by a constant factor (here, implicitly 3).
        """
        state = node.state
        unserved_children_count = 0
        for child in self.goal_children:
            if not f'(served {child})' in state:
                unserved_children_count += 1

        # A simple estimate: 3 actions per unserved child (make, put on tray, serve).
        # This is a simplification and can be adjusted based on domain analysis.
        return unserved_children_count * 3
