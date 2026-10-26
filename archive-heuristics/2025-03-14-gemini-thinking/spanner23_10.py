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
    It considers the actions needed for each nut that is not yet tightened in the current state:
    tightening the nut, walking to the nut's location if necessary, and picking up a spanner if not carrying one.

    # Assumptions:
    - There is always a usable spanner available at some location.
    - The graph of locations connected by 'link' predicates is connected.
    - We simplify path planning and assume a fixed cost for walking to a needed location.

    # Heuristic Initialization
    - Extracts the goal conditions (tightened nuts).
    - Extracts static facts, specifically 'link' predicates (although not directly used in this simplified version).
    - Identifies all nuts from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that needs to be tightened (based on the goal conditions):
    1. Check if the nut is already tightened in the current state. If yes, no cost for this nut.
    2. If the nut is not tightened:
        a. Initialize the cost for this nut to 1 (for the 'tighten_nut' action).
        b. Determine the location of the nut. We can assume the nut's location is static throughout the problem and can be derived from the initial state or goal.  For simplicity, we assume we can find the nut's location in the current state.
        c. Check if the man is at the nut's location in the current state. If not, increment the cost by 1 (for a 'walk' action to the nut's location).
        d. Check if the man is carrying any spanner in the current state. If not, increment the cost by 1 (for a 'pickup_spanner' action).
    3. Sum up the costs calculated for each nut that is not yet tightened to get the total heuristic value.
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
        """Estimate the number of actions needed to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        for nut_name in self.goal_nuts:
            if f'(tightened {nut_name})' not in state:
                nut_location = None
                man_location = None
                carrying_spanner = False

                # Find nut location (assuming it's the same as in initial state or current state)
                for fact in state:
                    if match(fact, "at", nut_name, "*"):
                        nut_location = get_parts(fact)[2]
                        break
                if nut_location is None: # Fallback to initial state if not found in current state (less robust)
                    for fact in node.init_state:
                        if match(fact, "at", nut_name, "*"):
                            nut_location = get_parts(fact)[2]
                            break
                if nut_location is None: # If still not found, skip this nut for heuristic calculation
                    continue

                # Find man location
                for fact in state:
                    if match(fact, "at", "*", nut_location):
                        if get_parts(fact)[1] == 'bob': # Assuming man's name is bob
                            man_location = nut_location
                            break
                    if match(fact, "at", "*", "*"):
                        if get_parts(fact)[1] == 'bob': # Assuming man's name is bob
                            current_man_location = get_parts(fact)[2]
                            if man_location is None:
                                man_location = current_man_location


                # Check if carrying a spanner
                for fact in state:
                    if match(fact, "carrying", "bob", "*"): # Assuming man's name is bob
                        carrying_spanner = True
                        break

                nut_cost = 1  # For tighten_nut action

                if man_location != nut_location:
                    nut_cost += 1  # For walk action

                if not carrying_spanner:
                    nut_cost += 1  # For pickup_spanner action

                heuristic_value += nut_cost

        return heuristic_value
