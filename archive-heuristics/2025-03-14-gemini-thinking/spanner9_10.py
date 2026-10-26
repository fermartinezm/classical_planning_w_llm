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
    It considers the actions needed to move the man to the nut's location, pick up a usable spanner,
    and finally tighten the nut.

    # Assumptions:
    - There is always a usable spanner available in the problem.
    - Moving between any two linked locations costs 1 action.
    - Picking up a spanner costs 1 action.
    - Tightening a nut costs 1 action.

    # Heuristic Initialization
    - The heuristic initializes by identifying the goal nuts that need to be tightened.
    - It also extracts static information about linked locations to potentially use for path cost estimation (though currently simplified to 1 if locations are different and linked).

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that is required to be tightened in the goal state and is not tightened in the current state:
    1. Initialize the estimated cost for tightening this nut to 1 (for the 'tighten_nut' action itself).
    2. Determine the location of the nut from the current state.
    3. Check if the man is at the same location as the nut. If not, increment the cost by 1 (for a 'walk' action to the nut's location).
    4. Check if the man is carrying a usable spanner. If not:
        a. Increment the cost by 1 (for a 'pickup_spanner' action).
        b. Find the location of a usable spanner.
        c. Check if the man is at the same location as a usable spanner. If not, increment the cost by 1 (for a 'walk' action to the usable spanner's location).
    5. Sum up the costs calculated for each nut that needs to be tightened.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        # Get man's location
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*") and get_parts(fact)[1] == "bob": # Assuming man's name is bob
                man_location = get_parts(fact)[2]
                break

        # Get usable spanner location (assuming there's at least one usable spanner)
        usable_spanner_location = None
        usable_spanner = None
        for fact in state:
            if match(fact, "at", "*", "*") and match(fact, "usable", get_parts(fact)[1]):
                usable_spanner_location = get_parts(fact)[2]
                usable_spanner = get_parts(fact)[1]
                break

        # Check if man is carrying a usable spanner
        carrying_usable_spanner = False
        if usable_spanner:
            for fact in state:
                if match(fact, "carrying", "*", "*") and get_parts(fact)[1] == "bob" and get_parts(fact)[2] == usable_spanner:
                    carrying_usable_spanner = True
                    break

        for nut in self.goal_nuts:
            if not f'(tightened {nut})' in state:
                nut_location = None
                for fact in state:
                    if match(fact, "at", "*", "*") and get_parts(fact)[1] == nut:
                        nut_location = get_parts(fact)[2]
                        break

                nut_cost = 1 # for tighten_nut action

                if man_location != nut_location:
                    nut_cost += 1 # for walk to nut location

                if not carrying_usable_spanner:
                    nut_cost += 1 # for pickup_spanner action
                    if man_location != usable_spanner_location:
                        nut_cost += 1 # for walk to spanner location

                heuristic_value += nut_cost

        return heuristic_value
