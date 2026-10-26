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

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts that are specified as goals to be tightened.
    It considers the need to move to the nut's location, pick up a usable spanner, and then tighten the nut.

    # Assumptions:
    - For each nut that needs to be tightened, we assume we need to perform a 'tighten_nut' action.
    - Before tightening, the man might need to walk to the nut's location and pick up a usable spanner.
    - We simplify the cost of getting a usable spanner to a fixed value if the man is not already carrying one.

    # Heuristic Initialization
    - The heuristic initializes by identifying the goal nuts (nuts that need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that is currently 'loose' and is in the goal to be 'tightened':
    1. Check if the man is at the same location as the nut. If not, add 1 to the cost (for a 'walk' action).
    2. Check if the man is carrying a usable spanner. If not, add 2 to the cost (to account for actions like 'walk to spanner' and 'pickup_spanner').
    3. Add 1 to the cost for the 'tighten_nut' action itself.
    4. Sum up these costs for all nuts that are 'loose' in the current state and need to be 'tightened' according to the goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts_to_tighten.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        # Get man's location
        man_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_location = get_parts(fact)[2]
                break

        # Get currently carried usable spanner
        carrying_usable_spanner = False
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                carried_spanner = get_parts(fact)[2]
                for fact2 in state:
                    if match(fact2, "usable", carried_spanner):
                        carrying_usable_spanner = True
                        break
                if carrying_usable_spanner:
                    break

        loose_nuts_to_tighten = []
        for nut_name in self.goal_nuts_to_tighten:
            is_loose = False
            is_tightened_in_state = False
            for fact in state:
                if match(fact, "loose", nut_name):
                    is_loose = True
                    break
                if match(fact, "tightened", nut_name):
                    is_tightened_in_state = True
                    break
            if is_loose and not is_tightened_in_state:
                loose_nuts_to_tighten.append(nut_name)

        for nut_name in loose_nuts_to_tighten:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut_name, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            cost_for_nut = 0
            if man_location != nut_location:
                cost_for_nut += 1
            if not carrying_usable_spanner:
                cost_for_nut += 2
            cost_for_nut += 1
            heuristic_value += cost_for_nut

        return heuristic_value
