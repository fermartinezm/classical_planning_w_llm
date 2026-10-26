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
    This heuristic estimates the number of actions required to tighten all loose nuts specified in the goal.
    It considers the actions: walk, pickup_spanner, and tighten_nut.

    # Assumptions:
    - For each loose nut that needs to be tightened, at least one 'tighten_nut' action is required.
    - If the man is not at the nut's location, a 'walk' action is needed.
    - If the man is not carrying a usable spanner, a 'pickup_spanner' action is needed.
    - We assume there is always a usable spanner available at some location.

    # Heuristic Initialization
    - Extract the goal conditions to identify which nuts need to be tightened.
    - No static facts are used in this heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of nuts that are required to be tightened in the goal state.
    2. Initialize the heuristic value to 0.
    3. Check if the man is currently carrying a usable spanner. If not, increment the heuristic value by 1 (for 'pickup_spanner').
    4. For each nut that is currently 'loose' and is in the set of nuts that need to be tightened in the goal:
        a. Check if the man is at the same location as the nut. If not, increment the heuristic value by 1 (for 'walk').
        b. Increment the heuristic value by 1 (for 'tighten_nut').
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        # Check if carrying a usable spanner
        carrying_usable_spanner = False
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                spanner = get_parts(fact)[2]
                for usable_fact in state:
                    if match(usable_fact, "usable", spanner):
                        carrying_usable_spanner = True
                        break
                if carrying_usable_spanner:
                    break
        if not carrying_usable_spanner:
            heuristic_value += 1  # pickup_spanner

        # Get man's location
        man_location = None
        man = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                if match(fact, "at", "?m - man", "*"):
                    man = get_parts(fact)[1]
                    man_location = get_parts(fact)[2]
                    break

        loose_goal_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in self.goal_nuts:
                    loose_goal_nuts.append(nut)

        for nut in loose_goal_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break
            if man_location != nut_location:
                heuristic_value += 1  # walk
            heuristic_value += 1      # tighten_nut

        return heuristic_value
