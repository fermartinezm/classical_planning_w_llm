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
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the actions needed for each loose nut individually and sums them up.
    The heuristic accounts for the need to walk to the nut's location, pick up a usable spanner,
    and finally tighten the nut.

    # Assumptions:
    - There is always at least one usable spanner available in the domain.
    - The locations are connected in such a way that all locations are reachable from each other if needed.
    - We only consider the actions needed to tighten each nut independently, without optimizing for shared actions.

    # Heuristic Initialization
    - The heuristic initializes by identifying all nuts that are initially loose from the initial state.
    - It does not require any static information for its calculation, as it operates directly on the current state and goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the heuristic value to 0.
    2. Identify all nuts that are currently 'loose' in the state.
    3. For each loose nut:
        a. Increment the heuristic value by 1 (for the 'tighten_nut' action itself).
        b. Determine the location of the nut.
        c. Check if the man is at the nut's location. If not, increment the heuristic value by 1 (for a 'walk' action to the nut's location).
        d. Check if the man is carrying a usable spanner. If not, increment the heuristic value by 1 (for a 'pickup_spanner' action).
    4. Return the total accumulated heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic. No specific initialization needed for this heuristic."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        heuristic_value = 0

        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.append(get_parts(fact)[1])

        if not loose_nuts: # if no loose nuts, goal is reached or no loose nuts in problem definition
            goal_achieved = True
            for goal_fact in self.goals:
                if goal_fact not in state:
                    goal_achieved = False
                    break
            if goal_achieved:
                return 0 # Goal state reached

        for nut in loose_nuts:
            nut_location = None
            man_location = None
            carrying_usable_spanner = False

            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                if match(fact, "at", "bob", "*"): # Assuming man's name is always 'bob' as per examples
                    man_location = get_parts(fact)[2]
                if match(fact, "carrying", "bob", "*"): # Assuming man's name is always 'bob'
                    spanner_name = get_parts(fact)[2]
                    if match(fact, "usable", spanner_name):
                        carrying_usable_spanner = True

            nut_cost = 1 # for tighten_nut action
            if man_location != nut_location:
                nut_cost += 1 # for walk action
            if not carrying_usable_spanner:
                nut_cost += 1 # for pickup_spanner action

            heuristic_value += nut_cost

        return heuristic_value
