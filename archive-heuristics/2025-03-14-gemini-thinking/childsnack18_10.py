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
    This heuristic estimates the number of children that are yet to be served.
    It counts the number of children for whom the 'served' predicate is not true in the current state.
    This heuristic is admissible in relaxed planning where we only consider serve actions.

    # Assumptions:
    - Serving each child requires at least one action.
    - The goal is to have all specified children served.

    # Heuristic Initialization
    - The heuristic initializes by identifying the goal children from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that need to be served based on the goal conditions.
    2. For each child, check if the 'served' predicate is true in the current state.
    3. Count the number of children for whom 'served' is not true.
    4. The heuristic value is the count of unserved children. This represents a lower bound on the number of 'serve' actions required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to serve all children."""
        state = node.state
        unserved_children_count = 0
        for child_name in self.goal_children:
            served_predicate = f'(served {child_name})'
            if served_predicate not in state:
                unserved_children_count += 1
        return unserved_children_count
