from fnmatch import fnmatch
# Assuming heuristic_base.py is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided elsewhere
# In a real planning system, this would be imported.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise an error
        # For robustness, return empty list or handle gracefully
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It calculates the cost for each unserved passenger independently
    and sums these costs. The cost for a single passenger includes the necessary
    lift movements to their origin (if not boarded) and destination floors,
    plus the board and depart actions.

    # Assumptions
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic calculates the minimum actions needed for each passenger
      in isolation, ignoring potential synergies (e.g., picking up/dropping
      off multiple passengers on a single trip or at the same floor). This
      relaxation makes the heuristic non-admissible but potentially more
      informative for greedy search.
    - The floor structure is a linear sequence defined by `(above f_upper f_lower)`
      facts, where `f_upper` is immediately above `f_lower`.

    # Heuristic Initialization
    - Extracts the linear order of floors and creates a mapping from floor
      name strings to numerical indices (0-based, from bottom to top).
    - Extracts the destination floor for each passenger from the initial state
      facts (assuming `(destin ?p ?f)` facts are in the initial state and are static).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the initial state facts (`task.initial_state`) and static facts (`task.static`)
       to determine the floor order and passenger destinations.
       - Identify all `(above f_upper f_lower)` facts from static facts.
       - Build a map `above_to_below` where `f_upper` maps to `f_lower`.
       - Find the top floor (a floor that is an `f_upper` but never an `f_lower`).
       - Traverse downwards from the top floor using the map to build an
         ordered list of floors from top to bottom.
       - Reverse the list to get the order from bottom to top.
       - Create a dictionary mapping each floor name to its 0-based index
         in the bottom-to-top order.
       - Identify all `(destin ?p ?f)` facts from initial state facts.
       - Build a dictionary mapping passenger name strings to their destination
         floor name strings.
    2. In the `__call__` method, for a given state (`node.state`):
       - Parse the current state facts to quickly identify:
         - The current floor of the lift (`(lift-at ?f)`).
         - The set of `served_passengers`.
         - The set of `boarded_passengers`.
         - A map `origin_map` from passenger name to their origin floor name
           for passengers currently waiting at an origin (`(origin ?p ?f)`).
       - Initialize `total_cost = 0`.
       - Get the index of the current lift floor.
       - Iterate through all passengers whose destination is known (from the
         map built in `__init__`).
       - For each passenger `p`:
         - If `p` is in `served_passengers`, the cost for this passenger is 0. Continue.
         - Get the destination floor `destin_f` and its index `destin_idx`.
         - If `p` is in `boarded_passengers`:
           - The passenger is in the lift. Cost is movement from current lift
             floor to `destin_f` plus the `depart` action.
           - Cost = `abs(current_floor_index - destin_idx) + 1`.
         - Elif `p` is in `origin_map`:
           - The passenger is waiting at their origin. Cost is movement from
             current lift floor to `origin_f`, plus `board` action,
             plus movement from `origin_f` to `destin_f`, plus `depart` action.
           - Find the origin floor `origin_f` from the `origin_map` and get its index `origin_idx`.
           - Cost = `abs(current_floor_index - origin_idx) + 1 + abs(origin_index - destin_idx) + 1`.
         - Else: The passenger is unserved but not at origin or boarded. This
           should not occur in valid states derived from the initial state.
           We assume valid states.
         - Add the calculated cost for passenger `p` to `total_cost`.
    4. Return `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state # Destinations are in initial state

        # 1. Determine floor order and create floor_to_index map
        above_to_below = {}
        all_floors = set()
        floors_below_something = set() # Floors that appear as f_lower

        # Collect all 'above' facts from static information
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_upper, f_lower = parts[1], parts[2]
                above_to_below[f_upper] = f_lower
                all_floors.add(f_upper)
                all_floors.add(f_lower)
                floors_below_something.add(f_lower)

        # Find the top floor: a floor that is mentioned but is never the lower floor in an 'above' fact.
        top_floor = None
        # Iterate through all floors found in 'above' facts
        for floor in all_floors:
             if floor not in floors_below_something:
                 top_floor = floor
                 break

        # If no 'above' facts exist but there are floors (e.g., single floor problem),
        # pick any floor as the "top" (and only) floor.
        if top_floor is None and all_floors:
             top_floor = next(iter(all_floors))
        elif top_floor is None and not all_floors:
             # No floors defined, likely an empty domain or problem.
             # This case shouldn't happen in typical miconic problems.
             # Initialize with empty maps/lists.
             self.ordered_floors = []
             self.floor_to_index = {}
             self.destin_map = {}
             return # Exit __init__ early

        # Build ordered list from top to bottom
        ordered_floors_top_down = []
        current = top_floor
        while current is not None:
            ordered_floors_top_down.append(current)
            current = above_to_below.get(current)

        # Reverse to get bottom to top and create index map
        self.ordered_floors = ordered_floors_top_down[::-1]
        self.floor_to_index = {floor: i for i, floor in enumerate(self.ordered_floors)}

        # 2. Determine passenger destinations from initial state facts
        self.destin_map = {}
        for fact in initial_state_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "destin":
                 passenger, floor = parts[1], parts[2]
                 self.destin_map[passenger] = floor


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state for quick lookups
        current_lift_floor = None
        boarded_passengers = set()
        origin_map = {} # passenger -> origin_floor
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "lift-at":
                current_lift_floor = parts[1]
            elif predicate == "boarded":
                boarded_passengers.add(parts[1])
            elif predicate == "origin":
                passenger, floor = parts[1], parts[2]
                origin_map[passenger] = floor
            elif predicate == "served":
                served_passengers.add(parts[1])

        # If no lift location is found, something is wrong with the state.
        # Or perhaps it's a goal state where lift location doesn't matter?
        # In miconic, lift-at is always present.
        if current_lift_floor is None:
             # This indicates an invalid state representation for this domain.
             # Returning infinity or a large number is safer for search.
             # However, for typical miconic problems, this shouldn't be reached.
             # Let's assume valid states have lift-at.
             # If there are no passengers and no goals, the heuristic should be 0.
             # The loop below handles this if destin_map is empty.
             # If there are passengers but no lift-at, the problem is unsolvable from here.
             return float('inf')


        current_floor_index = self.floor_to_index.get(current_lift_floor)
        # Handle case where lift is at a floor not in our map (shouldn't happen)
        if current_floor_index is None:
             return float('inf')


        total_cost = 0

        # Iterate through all passengers whose destination is known from initial state
        for passenger, destin_f in self.destin_map.items():
            # If passenger is already served, no cost for this passenger
            if passenger in served_passengers:
                continue

            # Get destination floor index
            destin_idx = self.floor_to_index.get(destin_f)
            # Handle case where destination floor is not in our map (shouldn't happen)
            if destin_idx is None:
                 return float('inf') # Problematic state/definition

            # Calculate cost based on passenger's current state
            if passenger in boarded_passengers:
                # Passenger is boarded, needs to be dropped off at destination
                # Cost = movement from current lift floor to destination + depart action
                cost = abs(current_floor_index - destin_idx) + 1
                total_cost += cost
            elif passenger in origin_map:
                # Passenger is at origin, needs pickup and dropoff
                # Cost = movement from current lift floor to origin + board action
                #        + movement from origin to destination + depart action
                origin_f = origin_map[passenger]
                origin_idx = self.floor_to_index.get(origin_f)
                # Handle case where origin floor is not in our map (shouldn't happen)
                if origin_idx is None:
                     return float('inf') # Problematic state/definition

                cost = abs(current_floor_index - origin_idx) + 1 + abs(origin_idx - destin_idx) + 1
                total_cost += cost
            # Else: Passenger is unserved but not at origin or boarded. This
            # state should not be reachable in a valid plan trace from initial state.
            # We assume valid states and skip this passenger if they are in such a state.
            # If a passenger is not in served, boarded, or origin facts, it implies
            # they were never in the initial state or were served and then somehow
            # became unserved without being at origin/boarded, which is invalid.

        return total_cost
