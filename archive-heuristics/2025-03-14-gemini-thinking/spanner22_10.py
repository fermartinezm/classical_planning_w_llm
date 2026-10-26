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
    It considers the actions needed to move the man to the nut's location, pick up a usable spanner, and finally tighten the nut.

    # Assumptions:
    - There is always a usable spanner available.
    - Locations are reachable if needed.
    - We simplify the cost of walking and picking up a spanner to a fixed cost if needed.

    # Heuristic Initialization
    - Extracts the goal nuts that need to be tightened.
    - Extracts static information about links between locations.
    - Identifies usable spanners from the initial state.
    - Identifies the man from the initial state.
    - Stores the initial location of each nut.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic cost to 0.
    2. Identify all nuts that are in the goal and are currently 'loose' in the given state.
    3. For each such loose nut:
       a. Increment the cost by 1 (for the 'tighten_nut' action).
       b. Determine the location of the nut from the initial state.
       c. Determine the current location of the man from the current state.
       d. If the man's current location is not the nut's location, increment the cost by 1 (for 'walk' action, simplified).
       e. Check if the man is carrying a usable spanner in the current state.
       f. If the man is not carrying a usable spanner, increment the cost by 1 (for 'pickup_spanner' action, simplified).
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

        self.usable_spanners = set()
        for fact in self.initial_state:
            if match(fact, "usable", "*"):
                self.usable_spanners.add(get_parts(fact)[1])

        self.man = None
        for fact in self.initial_state:
            if match(fact, "at", "*", "*") and get_parts(fact)[0] == 'at' and get_parts(fact)[1] not in self.usable_spanners: # Assuming man is not a spanner
                try:
                    if task.type_dict[get_parts(fact)[1]] == 'man':
                        self.man = get_parts(fact)[1]
                        break
                except KeyError: # In case types are not defined, assume first object that is 'at' and not a spanner is the man.
                    if self.man is None:
                        self.man = get_parts(fact)[1]
                        break


        self.nut_locations = {}
        for fact in self.initial_state:
            if match(fact, "at", "*", "*") and get_parts(fact)[0] == 'at':
                obj = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                if obj not in self.usable_spanners and obj != self.man: # Assuming nuts are not spanners or man
                    try:
                        if task.type_dict[obj] == 'nut':
                            self.nut_locations[obj] = loc
                    except KeyError: # If types are not defined, assume all objects that are 'at' and not spanners or man are nuts.
                        if obj not in self.usable_spanners and obj != self.man:
                             self.nut_locations[obj] = loc


    def __call__(self, node):
        """Estimate the minimum cost to tighten all goal nuts."""
        state = node.state
        cost = 0

        for nut in self.goal_nuts:
            if f'(tightened {nut})' not in state:
                cost += 1 # tighten_nut action

                man_location = None
                for fact in state:
                    if match(fact, "at", self.man, "*"):
                        man_location = get_parts(fact)[2]
                        break

                nut_location = self.nut_locations.get(nut)

                if man_location != nut_location:
                    cost += 1 # walk action

                carrying_usable_spanner = False
                for fact in state:
                    if match(fact, "carrying", self.man, "*"):
                        carried_spanner = get_parts(fact)[2]
                        if carried_spanner in self.usable_spanners:
                            carrying_usable_spanner = True
                            break
                if not carrying_usable_spanner:
                    cost += 1 # pickup_spanner action

        return cost
