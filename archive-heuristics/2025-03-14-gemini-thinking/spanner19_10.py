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
    This heuristic estimates the number of actions required to tighten all loose nuts to reach the goal state.
    It considers the actions needed to pick up a usable spanner, walk to the nut's location, and finally tighten the nut.

    # Assumptions:
    - The goal is always to tighten a set of nuts.
    - There is always at least one usable spanner available in the initial state.
    - Walking between any two linked locations costs 1 action.
    - Picking up a spanner costs 1 action.
    - Tightening a nut costs 1 action.

    # Heuristic Initialization
    - Extracts the goal nuts that need to be tightened.
    - Extracts static information about linked locations.
    - Identifies usable spanners from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that is loose in the current state and needs to be tightened according to the goal:
    1. Initialize the estimated cost for this nut to 0.
    2. Check if the man is currently carrying a usable spanner. If not, increment the cost by 1 (for pickup_spanner action).
    3. Determine the location of the nut and the man. If they are not at the same location, increment the cost by 1 (for walk action).
    4. Increment the cost by 1 for the tighten_nut action itself.
    5. Sum up the costs for all nuts that need to be tightened to get the total heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.usable_spanners = set()
        initial_state = task.initial_state

        for fact in initial_state:
            if match(fact, "usable", "*"):
                self.usable_spanners.add(get_parts(fact)[1])
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])


    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        goal_nut_not_tightened = []
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut_name = get_parts(goal)[1]
                if f'(tightened {nut_name})' not in state:
                    goal_nut_not_tightened.append(nut_name)

        for nut_name in goal_nut_not_tightened:
            if f'(loose {nut_name})' in state: # Only consider loose nuts for tightening
                nut_location = None
                man_location = None
                carrying_usable = False

                for fact in state:
                    if match(fact, "at", nut_name, "*"):
                        nut_location = get_parts(fact)[2]
                    if match(fact, "at", "bob", "*"): # Assuming man's name is always bob
                        man_location = get_parts(fact)[2]
                    if match(fact, "carrying", "bob", "*"): # Assuming man's name is always bob
                        carried_spanner = get_parts(fact)[2]
                        if carried_spanner in self.usable_spanners:
                            carrying_usable = True
                            break

                nut_cost = 0
                if not carrying_usable:
                    nut_cost += 1 # pickup_spanner
                if man_location != nut_location:
                    nut_cost += 1 # walk
                nut_cost += 1 # tighten_nut
                heuristic_value += nut_cost

        return heuristic_value
