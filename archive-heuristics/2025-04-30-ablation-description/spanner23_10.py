from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

# Helper function from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function from examples (included for completeness, though not used by this specific heuristic)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't compare more args than parts or vice versa
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the remaining effort by counting the number of
    goal conditions that are not yet satisfied in the current state.
    In the Spanner domain, the goals are typically of the form (tightened ?n).
    Therefore, this heuristic counts the number of nuts that are required
    to be tightened according to the goal, but are not yet tightened in
    the current state.

    # Assumptions
    - The goal is always a conjunction of (tightened ?n) predicates.
    - Each (tightened ?n) goal can only be achieved by the tighten_nut action.
    - The cost of achieving each unsatisfied goal is at least 1 (the tighten_nut action itself).

    # Heuristic Initialization
    - The heuristic stores the set of goal conditions from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state from the input node.
    2. Get the set of goal conditions stored during initialization.
    3. Calculate the set difference between the goal conditions and the current state. This gives the set of unsatisfied goal conditions.
    4. The heuristic value is the number of elements in the set of unsatisfied goal conditions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal conditions.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        # static_facts = task.static

    def __call__(self, node):
        """
        Estimate the minimum cost to reach a goal state from the current state.
        This is calculated as the number of goal conditions not yet satisfied.
        """
        state = node.state

        # The heuristic value is the number of goal facts that are not in the current state.
        # This is equivalent to |goals - state|.
        unsatisfied_goals = self.goals - state

        # The heuristic value is the count of unsatisfied goal facts.
        # Each unsatisfied goal (tightened ?n) requires at least one tighten_nut action.
        # This heuristic counts the number of such required actions.
        h_value = len(unsatisfied_goals)

        return h_value
