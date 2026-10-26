from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the actions needed for each loose nut individually and sums them up.
    For each loose nut, it estimates the cost of tightening it, considering if the man is already carrying a usable spanner and if the man is at the nut's location.

    # Assumptions:
    - For each loose nut, we minimally need one 'tighten_nut' action.
    - If the man is not carrying a usable spanner, we need one 'pickup_spanner' action.
    - If the man is not at the location of the nut, we need one 'walk' action.
    - The heuristic is a sum of these minimal costs for each loose nut.
    - It assumes that there is always a usable spanner available to pick up if needed.

    # Heuristic Initialization
    - The heuristic does not require any specific initialization beyond the task itself.
    - It will extract the goal conditions (tightened nuts) and the initial state to calculate the heuristic value for any given state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the heuristic value to 0.
    2. Identify all nuts that are currently 'loose' in the state.
    3. For each loose nut:
        a. Add 1 to the heuristic value (for the 'tighten_nut' action).
        b. Check if the man is carrying a usable spanner in the current state.
           If not, add 1 to the heuristic value (for the 'pickup_spanner' action).
        c. Determine the location of the nut and the location of the man in the current state.
           If they are not at the same location, add 1 to the heuristic value (for the 'walk' action).
    4. Return the total heuristic value.

    This heuristic is admissible under the assumption that each of these actions (walk, pickup_spanner, tighten_nut) are necessary and independent for each loose nut. However, for greedy best-first search, admissibility is not required, and this heuristic aims to provide a reasonable estimate of the remaining actions.
    """

    def __init__(self, task):
        """Initialize the heuristic. No specific initialization needed for this heuristic."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0
        loose_nuts = set()
        tightened_nuts = set()
        man_location = None
        carrying_usable_spanner = False

        def get_objects_from_fact(fact_str):
            return fact_str[1:-1].split()[1:]

        def get_predicate_name(fact_str):
            return fact_str[1:-1].split()[0]

        for fact in state:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'loose':
                loose_nuts.add(objects[0])
            elif predicate == 'tightened':
                tightened_nuts.add(objects[0])
            elif predicate == 'at':
                if objects[0] == 'bob': # Assuming 'bob' is the man's name
                    man_location = objects[1]
            elif predicate == 'carrying':
                spanner_carried = objects[1]
                if f'(usable {spanner_carried})' in state:
                    carrying_usable_spanner = True

        untightened_loose_nuts = loose_nuts - tightened_nuts

        for nut in untightened_loose_nuts:
            nut_location = None
            for fact in state:
                predicate = get_predicate_name(fact)
                objects = get_objects_from_fact(fact)
                if predicate == 'at' and objects[0] == nut:
                    nut_location = objects[1]
                    break

            heuristic_value += 1 # for tighten_nut action

            if not carrying_usable_spanner:
                heuristic_value += 1 # for pickup_spanner action

            if man_location != nut_location:
                heuristic_value += 1 # for walk action

        return heuristic_value
