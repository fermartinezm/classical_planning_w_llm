from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the number of loose nuts and whether the man is currently carrying a usable spanner.

    # Assumptions:
    - For each loose nut, we need to perform at least one 'tighten_nut' action.
    - If the man is not carrying a usable spanner, we assume a 'pickup_spanner' action is needed.
    - We simplify the cost of reaching the nut's location and assume it's covered within the cost of tightening.

    # Heuristic Initialization
    - The heuristic initializes by identifying the goal conditions (tightened nuts).
    - Static facts (links) are not directly used in this simple heuristic but could be used for more sophisticated path cost estimation in future versions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of nuts that are currently 'loose'. These are the nuts that need to be tightened to achieve the goal.
    2. Check if the man is currently 'carrying' a 'usable' spanner.
    3. Initialize the heuristic value to the count of loose nuts (assuming one action per nut to tighten, which implicitly includes walking to the nut).
    4. If the man is NOT carrying a usable spanner, increment the heuristic value by 1. This accounts for the action of picking up a spanner before tightening any nuts.
    5. Return the calculated heuristic value. This value represents a lower bound on the number of actions needed, under the simplifying assumptions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state

        def match(fact, *args):
            """
            Utility function to check if a PDDL fact matches a given pattern.
            - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
            - `args`: The pattern to match (e.g., "at", "*", "rooma").
            - Returns `True` if the fact matches the pattern, `False` otherwise.
            """
            parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        loose_nuts_count = 0
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts_count += 1

        carrying_usable_spanner = False
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                spanner_name = fact.split()[2] # Extract spanner name
                if f'(usable {spanner_name})' in state:
                    carrying_usable_spanner = True
                    break

        heuristic_value = loose_nuts_count
        if not carrying_usable_spanner and loose_nuts_count > 0: # Only add pickup cost if there are nuts to tighten
            heuristic_value += 1

        return heuristic_value
