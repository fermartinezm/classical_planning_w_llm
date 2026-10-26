from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It counts 2 actions for each unserved passenger (board and depart)
    plus 1 action for each distinct floor that the lift must visit to pick up
    or drop off an unserved passenger.

    # Assumptions
    - Floors are ordered linearly based on the `above` predicates.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding is 1.
    - The cost of departing is 1.
    - The lift can carry multiple passengers.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the initial state.
    - Determines the linear order of floors based on `above` predicates and
      creates mappings between floor names and their numerical indices.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers who have not yet been served by checking for the
       absence of the `(served ?p)` predicate in the current state.
    2. If there are no unserved passengers, the goal is reached, and the heuristic is 0.
    3. Initialize the heuristic value `h`. Add 2 for each unserved passenger,
       representing the minimum board and depart actions required for them.
    4. Identify the set of unique floors where unserved passengers are currently
       waiting. These are the floors `f` for which `(origin ?p f)` is true for
       some unserved passenger `?p`. These are the required pickup stops.
    5. Identify the set of unique destination floors for all unserved passengers.
       These are the floors `self.destinations[?p]` for each unserved passenger `?p`.
       These are the required dropoff stops.
    6. Combine the sets from steps 4 and 5 to get the set of all distinct floors
       the lift must visit to serve the remaining passengers.
    7. Add the number of distinct required floors (the size of the combined set)
       to the heuristic value `h`. This component estimates the minimum number
       of stops or floor-visiting actions needed.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and floor order.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Extract passenger destinations from the initial state
        self.destinations = {}
        for fact in self.initial_state:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Build floor order from 'above' predicates
        all_floors = set()
        above_map = {} # maps lower_floor -> higher_floor
        is_higher_in_above = set() # floors that appear as the higher floor in an 'above' fact
        is_lower_in_above = set()  # floors that appear as the lower floor in an 'above' fact

        for fact in self.static_facts:
            if match(fact, "above", "*", "*"):
                _, higher_floor, lower_floor = get_parts(fact)
                above_map[lower_floor] = higher_floor
                is_higher_in_above.add(higher_floor)
                is_lower_in_above.add(lower_floor)
                all_floors.add(higher_floor)
                all_floors.add(lower_floor)

        # Also collect floors mentioned in the initial state if not in static facts
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["lift-at", "origin", "destin"]:
                 for part in parts[1:]:
                     # Simple check: assume objects starting with 'f' are floors
                     if isinstance(part, str) and part.startswith('f'):
                         all_floors.add(part)

        # Find the lowest floor: It's a floor that is mentioned but is never the 'higher_floor'
        # in any 'above' predicate.
        lowest_floor = None
        for floor in all_floors:
            if floor not in is_higher_in_above:
                 lowest_floor = floor
                 break # Assuming a single lowest floor

        if lowest_floor is None and all_floors:
             # Fallback/handle cases where the lowest floor might only be in initial state facts
             # or if the 'above' structure is unusual.
             # If a floor is a 'lower_floor' but never a 'higher_floor', it must be the lowest.
             candidates = is_lower_in_above - is_higher_in_above
             if candidates:
                 lowest_floor = list(candidates)[0]
             elif len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # If still no clear lowest floor, print warning and use lexicographical sort
                 print("Warning: Could not determine floor order from 'above' predicates. Falling back to lexicographical sort.")
                 sorted_floors = sorted(list(all_floors))
                 self.floor_to_idx = {floor: i for i, floor in enumerate(sorted_floors)}
                 self.idx_to_floor = {i: floor for i, floor in enumerate(sorted_floors)}
                 # Set lowest_floor to None to indicate fallback was used
                 lowest_floor = None


        if lowest_floor:
            # Build sorted floor list by following the 'above' chain from the lowest floor
            sorted_floors = []
            current = lowest_floor
            while current is not None:
                sorted_floors.append(current)
                current = above_map.get(current) # Get the floor immediately above

            self.floor_to_idx = {floor: i for i, floor in enumerate(sorted_floors)}
            self.idx_to_floor = {i: floor for i, floor in enumerate(sorted_floors)}
        elif not all_floors:
             # Handle case with no floors found
             self.floor_to_idx = {}
             self.idx_to_floor = {}
             print("Warning: No floors found.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved passengers
        # All passengers are those with a destination defined in the initial state
        all_passengers = set(self.destinations.keys())
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = all_passengers - served_passengers

        # If all passengers are served, the goal is reached, heuristic is 0.
        if not unserved_passengers:
            return 0

        # 2. Initialize heuristic with actions per passenger
        # Each unserved passenger needs a board and a depart action.
        h = 2 * len(unserved_passengers)

        # 3. Identify required pickup stops
        pickup_floors = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                # Only consider origins for passengers who are still unserved
                if passenger in unserved_passengers:
                    pickup_floors.add(floor)

        # 4. Identify required dropoff stops
        dropoff_floors = set()
        for passenger in unserved_passengers:
             # Destination is a required dropoff stop for any unserved passenger
             if passenger in self.destinations:
                 dropoff_floors.add(self.destinations[passenger])
             # else: This passenger doesn't have a destination in the initial state,
             # which indicates a problem definition issue if they are in all_passengers.

        # 5. Combine required stops
        required_stops = pickup_floors | dropoff_floors

        # 6. Add number of distinct required stops to heuristic
        # This counts the minimum number of floors the lift must visit.
        # It's a proxy for movement cost and stopping actions.
        h += len(required_stops)

        # Note: This heuristic is not admissible but aims to be informative for greedy search.

        return h
