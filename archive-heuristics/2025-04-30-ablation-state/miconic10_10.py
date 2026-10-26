from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe raise error or return empty list
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
    passengers. It sums the estimated cost for each unserved passenger,
    considering their current state (waiting at origin or boarded) and the
    lift's current location. The estimated cost for a passenger includes the
    lift movement to their pickup/dropoff floor and the board/depart actions.

    # Assumptions
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic sums individual passenger costs, potentially overestimating
      by double-counting shared lift movements. This is acceptable for a
      non-admissible heuristic used in greedy best-first search.
    - All passenger origins and destinations are available in the static facts or initial state.
    - The 'above' predicates define a linear order of floors.

    # Heuristic Initialization
    - Parses the 'above' predicates from static facts to determine the linear
      order of floors from highest to lowest and create a mapping from floor name to its index.
    - Extracts and stores the origin and destination floors for each passenger
      from the static facts and initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current floor of the lift from the state.
    2.  Initialize the total estimated cost to 0.
    3.  Iterate through all passengers whose origin and destination are known
        (parsed during initialization).
    4.  For each passenger:
        -   Check if the passenger is already served (predicate `(served p)` is true in the state). If yes, continue to the next passenger.
        -   If the passenger is not served, retrieve their origin and destination floors.
        -   Check if the passenger is currently waiting at their origin floor (predicate `(origin p o)` is true in the state).
            -   If yes, estimate the cost for this passenger as:
                `distance(current_lift_floor, origin_floor) + 1 (board) + distance(origin_floor, destination_floor) + 1 (depart)`.
                Add this cost to the total.
        -   Check if the passenger is currently boarded (predicate `(boarded p)` is true in the state).
            -   If yes, estimate the cost for this passenger as:
                `distance(current_lift_floor, destination_floor) + 1 (depart)`.
                Add this cost to the total.
        -   (A passenger should be either waiting at origin, boarded, or served in a valid state trajectory from a valid initial state).
    5.  Return the total estimated cost.

    The distance between two floors is calculated as the absolute difference
    of their indices in the ordered list of floors (highest floor is index 0).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, floor indices,
        and passenger origins/destinations.
        """
        super().__init__(task)

        # --- Parse Floor Order and Indices ---
        floor_names = set()
        # Build the 'below' map: floor -> floor directly below it
        below_map = {} # higher_floor -> lower_floor
        # Keep track of all floors that are 'lower' in an 'above' relation
        all_lower_floors_in_above = set()

        # Parse 'above' facts from static
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above' and len(parts) == 3:
                # (above floor_higher floor_lower)
                floor_higher, floor_lower = parts[1], parts[2]
                floor_names.add(floor_higher)
                floor_names.add(floor_lower)
                below_map[floor_higher] = floor_lower
                all_lower_floors_in_above.add(floor_lower)

        # Find the highest floor (a floor in floor_names that is not a value in below_map)
        highest_floor = None
        for floor in floor_names:
            if floor not in all_lower_floors_in_above:
                highest_floor = floor
                break

        self.floor_order = []
        if highest_floor:
            # Build the ordered list from highest to lowest
            current = highest_floor
            while current is not None:
                self.floor_order.append(current)
                current = below_map.get(current)
        elif floor_names:
             # Handle cases with potentially non-linear or single floor structure
             # If there's only one floor, it's the order.
             if len(floor_names) == 1:
                 self.floor_order = list(floor_names)
             else:
                 # This case indicates an unexpected 'above' structure (e.g., disconnected, cyclic)
                 # or maybe the highest floor wasn't found correctly.
                 # For robustness, sort alphabetically as a fallback, though this loses semantic order.
                 # print("Warning: Could not determine linear floor order from 'above' predicates. Using sorted names.")
                 self.floor_order = sorted(list(floor_names))


        # Create floor name to index map (highest floor is index 0)
        self.floor_to_index = {floor: index for index, floor in enumerate(self.floor_order)}

        # --- Extract Passenger Origins and Destinations ---
        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.all_passengers = set()

        # Collect origin and destination facts from static and initial state
        # Static facts are usually preferred for fixed problem properties
        facts_to_check = set(self.static) | set(task.initial_state)

        for fact in facts_to_check:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'origin' and len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    self.passenger_origins[passenger] = floor
                    self.all_passengers.add(passenger)
                elif parts[0] == 'destin' and len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    self.passenger_destinations[passenger] = floor
                    self.all_passengers.add(passenger)

    def dist(self, floor1, floor2):
        """Calculate the distance (number of floors to traverse) between two floors."""
        if floor1 not in self.floor_to_index or floor2 not in self.floor_to_index:
            # This should not happen if floors are correctly parsed and used.
            # Return a large value to indicate impossibility or high cost.
            # print(f"Warning: Unknown floor in distance calculation: {floor1} or {floor2}")
            return float('inf')

        index1 = self.floor_to_index[floor1]
        index2 = self.floor_to_index[floor2]
        return abs(index1 - index2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify the current floor of the lift.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # This state is likely invalid (no lift location or unknown floor)
             return float('inf') # Indicate unreachable or invalid state

        # 2. Initialize the total estimated cost to 0.
        total_cost = 0

        # Set of served passengers for quick lookup
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # 3. Iterate through all passengers identified during initialization
        for passenger in self.all_passengers:
            # 4. For each passenger:
            # Check if the passenger is already served.
            if passenger in served_passengers:
                continue # Passenger is served, cost is 0 for this passenger

            # Retrieve their origin and destination floors.
            origin_floor = self.passenger_origins.get(passenger)
            destination_floor = self.passenger_destinations.get(passenger)

            # This check is mostly defensive; passengers in all_passengers should have O/D
            if origin_floor is None or destination_floor is None:
                 # print(f"Warning: Origin or destination missing for unserved passenger {passenger}")
                 continue # Cannot estimate cost for this passenger

            # Check the passenger's current state
            is_waiting_at_origin = f"(origin {passenger} {origin_floor})" in state
            is_boarded = f"(boarded {passenger})" in state

            # Estimate cost based on state
            if is_waiting_at_origin:
                # Needs board at origin, then move to destination, then depart
                # Cost = move(L, o) + board + move(o, d) + depart
                cost_move_to_origin = self.dist(current_lift_floor, origin_floor)
                cost_move_origin_to_destin = self.dist(origin_floor, destination_floor)
                cost_actions = 1 + 1 # board + depart
                # Ensure distances are finite before adding
                if cost_move_to_origin == float('inf') or cost_move_origin_to_destin == float('inf'):
                     return float('inf') # Indicate unreachable origin/destination
                total_cost += cost_move_to_origin + cost_actions + cost_move_origin_to_destin

            elif is_boarded:
                # Needs move to destination, then depart
                # Cost = move(L, d) + depart
                cost_move_to_destin = self.dist(current_lift_floor, destination_floor)
                cost_actions = 1 # depart
                 # Ensure distance is finite before adding
                if cost_move_to_destin == float('inf'):
                     return float('inf') # Indicate unreachable destination
                total_cost += cost_move_to_destin + cost_actions

            # else: Unserved passenger is neither waiting at origin nor boarded.
            # This state is likely invalid according to domain logic.
            # In a valid state, an unserved passenger must be at their origin or boarded.
            # If this happens, the state is likely unreachable or problematic.
            # Returning infinity signals this state is bad.
            # Check if the passenger *should* have O/D based on initialization
            elif passenger in self.passenger_origins and passenger in self.passenger_destinations:
                 # Unserved, but not at origin and not boarded. Invalid state.
                 # print(f"Warning: Unserved passenger {passenger} is neither at origin {origin_floor} nor boarded.")
                 return float('inf')


        # 5. Return the total estimated cost.
        return total_cost
