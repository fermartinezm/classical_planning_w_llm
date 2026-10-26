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
    - For each loose nut, we assume we need to perform the following actions in sequence:
        1. Walk to the nut's location (if not already there).
        2. Pickup a usable spanner (if not already carrying one).
        3. Tighten the nut.
    - We assume there is always a usable spanner available and locations are reachable if needed.
    - The heuristic is not admissible but aims to be informative and efficient for greedy search.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal conditions (tightened nuts) and static facts (links between locations).
    - It identifies the nuts that need to be tightened to achieve the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the nuts that are required to be tightened based on the goal state.
    2. For each nut that is required to be tightened:
        a. Check if the nut is already tightened in the current state. If yes, no cost for this nut.
        b. If the nut is not tightened, estimate the cost to tighten it:
            i.  Assume 1 action for tightening the nut ('tighten_nut').
            ii. Check if the man is at the nut's location. If not, assume 1 action to walk to the nut's location ('walk').
            iii.Check if the man is carrying a usable spanner. If not, assume 1 action to pickup a usable spanner ('pickup_spanner').
        c. Sum up the estimated costs for all nuts that are not yet tightened.
    3. The total sum represents the heuristic estimate for the current state.
    """

    def __init__(self, task):
        """
        Initialize the spanner heuristic. Extracts goal conditions and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        for nut_name in self.goal_nuts:
            if f'(tightened {nut_name})' not in state:
                nut_location = None
                man_location = None
                carrying_usable_spanner = False

                # Find nut location
                for fact in state:
                    if match(fact, "at", nut_name, "*"):
                        nut_location = get_parts(fact)[2]
                        break
                if nut_location is None: # Nut location not found in state, which should not happen in valid problem instances.
                    continue # Skip this nut and assume it's not reachable.

                # Find man location
                for fact in state:
                    if match(fact, "at", "*", nut_location) and get_parts(fact)[1] not in ['spanner1', 'spanner2', 'spanner3', 'spanner4', 'spanner5', 'spanner6', 'spanner7', 'spanner8', 'spanner9', 'spanner10', 'spanner11', 'spanner12', 'spanner13', 'spanner14', 'spanner15', 'spanner16', 'spanner17', 'spanner18', 'spanner19', 'spanner20']: # Assuming man is not named 'spannerX'
                        man_location = nut_location
                        break
                if man_location is None:
                    for fact in state:
                        if match(fact, "at", "*", "*") and get_parts(fact)[1] not in ['spanner1', 'spanner2', 'spanner3', 'spanner4', 'spanner5', 'spanner6', 'spanner7', 'spanner8', 'spanner9', 'spanner10', 'spanner11', 'spanner12', 'spanner13', 'spanner14', 'spanner15', 'spanner16', 'spanner17', 'spanner18', 'spanner19', 'spanner20']:
                            man_location_other = get_parts(fact)[2]
                            if man_location_other != nut_location:
                                heuristic_value += 1 # Walk action needed
                            break
                    if man_location is None: # Man location not found at all, which should not happen in valid problem instances.
                        continue # Skip this nut and assume man is not reachable.


                # Check if carrying usable spanner
                for fact in state:
                    if match(fact, "carrying", "*", "*"):
                        spanner_name = get_parts(fact)[2]
                        if f'(usable {spanner_name})' in state:
                            carrying_usable_spanner = True
                            break
                if not carrying_usable_spanner:
                    heuristic_value += 1 # Pickup spanner action needed

                heuristic_value += 1 # Tighten nut action needed

        return heuristic_value
