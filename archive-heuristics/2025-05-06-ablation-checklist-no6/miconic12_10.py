# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class miconicHeuristic:
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the cost by summing the independent costs for each unserved passenger.
    The cost for a waiting passenger is:
        (moves from current lift floor to origin) + 1 (board) +
        (moves from origin to destination) + 1 (depart)
    The cost for a boarded passenger is:
        (moves from current lift floor to destination) + 1 (depart)

    Move cost between floors is the absolute difference in their indices.
    Floor indices are determined by the 'above' facts, assuming a linear floor structure.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Destination floor for each passenger.
        - Floor ordering and indexing.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Extract passenger destinations from static facts
        self.destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin' and len(parts) == 3:
                passenger, destination_floor = parts[1], parts[2]
                self.destinations[passenger] = destination_floor

        # 2. Determine floor ordering and create floor index map
        self.floor_indices = {}
        self.floor_list = [] # Ordered list of floors from lowest to highest

        above_map = {} # Map: floor_below -> floor_above
        floors_that_are_above_others = set()
        floors_that_are_below_others = set()
        all_floors = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above' and len(parts) == 3:
                floor_above, floor_below = parts[1], parts[2]
                above_map[floor_below] = floor_above
                floors_that_are_above_others.add(floor_above)
                floors_that_are_below_others.add(floor_below)
                all_floors.add(floor_above)
                all_floors.add(floor_below)

        # Find the lowest floor: it's a floor that is below something but not above anything else.
        lowest_floor = None
        for floor in floors_that_are_below_others:
            if floor not in floors_that_are_above_others:
                lowest_floor = floor
                break

        # Handle edge case: single floor or no 'above' facts
        if lowest_floor is None and len(all_floors) > 0:
             # If there's only one floor, it's the lowest.
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # Fallback for unexpected structures: pick one arbitrarily (e.g., alphabetically)
                 # or the one that is not a key in above_map (nothing below it)
                 potential_lowest = [f for f in all_floors if f not in above_map]
                 if potential_lowest:
                     lowest_floor = sorted(potential_lowest)[0] # Pick first alphabetically
                 elif all_floors:
                     # If all floors are keys in above_map, there's a cycle or disconnected component.
                     # This shouldn't happen in standard miconic. Pick one arbitrarily.
                     lowest_floor = sorted(list(all_floors))[0]


        if lowest_floor:
            current_floor = lowest_floor
            index = 0
            while current_floor is not None:
                self.floor_list.append(current_floor)
                self.floor_indices[current_floor] = index
                index += 1
                current_floor = above_map.get(current_floor) # Get the floor above the current one

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Find the current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        # If lift location is unknown or not indexed, heuristic is infinite (or a large number)
        # This implies an invalid state or problem definition for this heuristic.
        # Assuming valid states where lift-at is always present and floor is indexed.
        current_idx = self.floor_indices.get(current_lift_floor, -1)
        if current_idx == -1:
             # Should not happen in a solvable miconic problem with valid states
             # Return a large value to prune this state if encountered
             return float('inf')


        heuristic = 0

        # Track passengers' current status
        served_passengers = set()
        waiting_passengers = {} # {passenger: origin_floor}
        boarded_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'served' and len(parts) == 2:
                served_passengers.add(parts[1])
            elif parts[0] == 'origin' and len(parts) == 3:
                waiting_passengers[parts[1]] = parts[2]
            elif parts[0] == 'boarded' and len(parts) == 2:
                boarded_passengers.add(parts[1])

        # Iterate through all passengers known from destinations
        all_passengers = set(self.destinations.keys())

        for passenger in all_passengers:
            # If passenger is served, they contribute 0 to the heuristic
            if passenger in served_passengers:
                continue

            # Passenger is unserved. Get their destination.
            destination_floor = self.destinations.get(passenger)
            if destination_floor is None:
                 # Should not happen in a valid problem
                 continue

            destination_idx = self.floor_indices.get(destination_floor, -1)
            if destination_idx == -1:
                 # Should not happen in a valid problem
                 return float('inf') # Destination floor not indexed

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to go to destination and depart
                # Cost = moves from current lift floor to destination + 1 (depart)
                moves = abs(current_idx - destination_idx)
                heuristic += moves + 1
            elif passenger in waiting_passengers:
                 # Passenger is waiting at origin, needs pickup, travel, and depart
                 origin_floor = waiting_passengers[passenger]
                 origin_idx = self.floor_indices.get(origin_floor, -1)
                 if origin_idx == -1:
                      # Should not happen in a valid problem
                      return float('inf') # Origin floor not indexed

                 # Cost = moves from current lift floor to origin + 1 (board) +
                 #        moves from origin to destination + 1 (depart)
                 moves_to_origin = abs(current_idx - origin_idx)
                 moves_origin_to_destin = abs(origin_idx - destination_idx)
                 heuristic += moves_to_origin + 1 + moves_origin_to_destin + 1
            # Else: Passenger is unserved but neither waiting nor boarded.
            # This case indicates an invalid state according to standard miconic rules.
            # We assume valid states for heuristic computation.

        return heuristic
