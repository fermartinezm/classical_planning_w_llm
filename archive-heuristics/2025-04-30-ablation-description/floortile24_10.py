from fnmatch import fnmatch
# Assuming Heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or raise an error
        # For robustness, return empty list or handle gracefully
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of goal conditions related to painted tiles
    that are not yet satisfied in the current state. It counts how many tiles
    still need to be painted with their target color according to the goal state.

    # Assumptions
    - The primary objective of the task is to paint specific tiles with specific colors.
    - The goal state is defined primarily by `(painted tile color)` facts.
    - Each tile that needs to be painted correctly contributes 1 to the heuristic value.
    - This heuristic does not consider the cost of movement, color changes, or clearing tiles,
      making it potentially non-admissible but simple and fast to compute.

    # Heuristic Initialization
    - The constructor extracts all `(painted ?tile ?color)` facts from the task's goal conditions.
    - These required painted states are stored in a set for efficient lookup during heuristic computation.
    - Static facts are available but not used in this specific heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Access the current state of the world, which is a set of facts.
    2. Initialize a counter for unsatisfied painted goals to 0.
    3. Iterate through the set of required `(painted tile color)` facts stored during initialization.
    4. For each required painted fact (e.g., `'(painted tile_1_1 white)'`):
       - Check if this fact is present in the current state.
       - If the fact is NOT found in the current state, it means this goal is not yet achieved for this tile.
       - Increment the counter for unsatisfied painted goals by 1.
    5. The final value of the counter represents the heuristic estimate for the current state.
    6. Return the counter value. The heuristic is 0 if and only if all required painted goals are present in the state (assuming the full goal is only painted facts, as in the examples).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.

        Args:
            task: The planning task object containing initial state, goals, operators, etc.
        """
        # Extract and store the set of goal facts that represent painted tiles.
        # We only care about (painted tile color) goals for this heuristic.
        # The task.goals is already a frozenset of strings.
        self.painted_goals = {
            goal for goal in task.goals if match(goal, "painted", "*", "*")
        }

        # Note: Static facts (task.static) are available but not used in this heuristic.
        # Example: self.static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal state.
        """
        state = node.state  # Current world state as a frozenset of strings.

        # The heuristic is the number of painted goals that are not in the current state.
        # We can compute this by finding the difference between the set of required
        # painted goals and the set of facts in the current state.
        unsatisfied_painted_goals = len(self.painted_goals - state)

        # This heuristic is 0 if and only if all facts in self.painted_goals are
        # present in the state. Based on the provided example instances where the
        # goal only consists of (painted tile color) facts, this means the heuristic
        # is 0 iff the state is the goal state.

        return unsatisfied_painted_goals
