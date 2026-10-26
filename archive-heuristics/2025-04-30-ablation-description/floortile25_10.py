from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch # Although not strictly needed for the simplest version, useful for parsing

# Helper functions from example heuristics
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number
    of goal conditions that are not yet satisfied in the current state.
    In the Floortile domain, goal conditions are typically that specific tiles
    must be painted with specific colors.

    # Assumptions
    - The goal consists solely of (painted ?tile ?color) facts.
    - Action costs are uniform (implicitly assumed by counting unsatisfied goals).
    - Tiles painted with the wrong color cannot be repainted or cleared (based on domain analysis).
      The heuristic simply counts such tiles as unsatisfied goals.

    # Heuristic Initialization
    - The heuristic stores the set of goal facts provided by the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the set of goal facts from the task definition (stored during initialization).
    2. Get the current state, which is a set of facts that are true.
    3. Initialize a counter for unsatisfied goal conditions to 0.
    4. Iterate through each fact in the set of goal facts.
    5. For each goal fact, check if it is present in the current state.
    6. If a goal fact is *not* found in the current state, increment the counter.
    7. The heuristic value is the final count of goal facts not present in the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal conditions.

        @param task: The planning task object containing goals and static facts.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are available in task.static but not needed for this simple heuristic.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated cost (number of unsatisfied goal facts) to reach the goal.
        """
        state = node.state  # Current world state as a frozenset of strings.

        # The heuristic is the number of goal facts that are not present in the current state.
        unsatisfied_goals_count = 0
        for goal_fact in self.goals:
            if goal_fact not in state:
                unsatisfied_goals_count += 1

        return unsatisfied_goals_count

