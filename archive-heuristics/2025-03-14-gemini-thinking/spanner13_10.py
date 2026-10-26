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
    This heuristic estimates the minimum number of actions required to tighten all loose nuts.
    It considers the actions needed to move the man to the nut's location, pick up a usable spanner if necessary,
    and finally tighten the nut. It sums up the estimated costs for each nut that needs to be tightened.

    # Assumptions:
    - For each loose nut, we assume we need to perform at most one 'walk' action to reach the nut's location if the man is not already there.
    - For each nut, we assume we need to perform at most one 'pickup_spanner' action if the man is not already carrying a usable spanner.
    - We assume there is always at least one usable spanner available in the domain.
    - We do not explicitly calculate shortest paths between locations, but rather assume a single 'walk' action is sufficient if locations differ.

    # Heuristic Initialization
    - The heuristic initializes by pre-processing the goal conditions to identify all nuts that need to be tightened.
    - It also extracts static information about the initial locations of nuts and usable spanners from the initial state of the task.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition (tightened ?nut) that is not satisfied in the current state:
    1. Initialize the estimated cost for this nut to 0.
    2. Check if the man is at the same location as the nut.
       - Determine the nut's location from the initial state.
       - Determine the man's current location from the current state.
       - If the man is not at the nut's location, increment the cost by 1 (for a 'walk' action).
    3. Check if the man is carrying a usable spanner.
       - Iterate through all spanners.
       - Check if there is a fact '(carrying man spanner)' and '(usable spanner)' in the current state.
       - If the man is not carrying any usable spanner, increment the cost by 1 (for a 'pickup_spanner' action).
    4. Increment the cost by 1 (for the 'tighten_nut' action itself).
    5. The heuristic value for the state is the sum of the costs calculated for each nut that is not yet tightened in the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Specifically, identify the nuts that need to be tightened and their initial locations.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        self.nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.nuts_to_tighten.add(get_parts(goal)[1])

        self.nut_locations = {}
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[0] in self.nuts_to_tighten:
                    self.nut_locations[parts[0]] = parts[2]

    def __call__(self, node):
        """
        Estimate the number of actions needed to tighten all loose nuts from the current state.
        """
        state = node.state
        heuristic_value = 0

        for nut in self.nuts_to_tighten:
            if f'(tightened {nut})' not in state:
                nut_cost = 0

                # Get man and nut location
                man_location = None
                nut_location = self.nut_locations.get(nut)
                man = None
                for fact in state:
                    if match(fact, "at", "*", "*"):
                        parts = get_parts(fact)
                        if parts[0] == 'bob': # Assuming 'bob' is the man
                            man_location = parts[2]
                            man = parts[0]
                            break
                if man_location is None:
                    return float('inf') # Should not happen in valid problems

                if man_location != nut_location:
                    nut_cost += 1 # walk action

                carrying_usable_spanner = False
                for fact in state:
                    if match(fact, "carrying", man, "*"):
                        spanner = get_parts(fact)[2]
                        if f'(usable {spanner})' in state:
                            carrying_usable_spanner = True
                            break
                if not carrying_usable_spanner:
                    nut_cost += 1 # pickup_spanner action

                nut_cost += 1 # tighten_nut action
                heuristic_value += nut_cost

        return heuristic_value
