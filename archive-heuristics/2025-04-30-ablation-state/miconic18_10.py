from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions (moves, board, depart)
    required to serve all passengers. It combines the estimated vertical travel
    needed to cover the range of floors where passengers need pickup or dropoff
    with the total number of board and depart actions required for unserved passengers.

    # Assumptions
    - Floors are ordered linearly by the `above` predicate, forming a single vertical stack.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - All passengers specified in the goal must be served.

    # Heuristic Initialization
    - Parses static facts (`above`) to determine the floor order and create a mapping
      from floor names to numerical indices (0 for the lowest floor, increasing upwards).
    - Parses static facts (`destin`) to store the destination floor for each passenger.
    - Identifies all passengers that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the goal state is reached (all goal passengers are served). If yes, return 0.
    2. Determine the current floor of the lift. If the lift's location is unknown or invalid, return infinity.
    3. Identify all unserved goal passengers.
    4. For each unserved goal passenger:
       - Check their current state: Are they waiting at their origin floor (`(origin p f)` is true) or are they boarded (`(boarded p)` is true)? (Assuming valid states, one of these must be true if not served).
       - If waiting at origin `f`, add `f` to the set of pickup floors and increment the count of unboarded passengers. If the origin floor is unknown or invalid, return infinity.
       - If boarded, add their destination floor to the set of dropoff floors and increment the count of boarded passengers. If the destination floor is unknown or invalid, return infinity.
       - If a passenger is unserved but neither waiting at origin nor boarded, the state is invalid; return infinity.
    5. Combine pickup and dropoff floors into a single set of service floors.
    6. If the set of service floors is empty (which should only happen if all goal passengers are served, handled in step 1), the heuristic is 0.
    7. Get the numerical indices for the current lift floor and all service floors using the floor-to-index map created during initialization.
    8. Find the minimum and maximum indices among the service floors.
    9. Estimate the movement cost: This is the minimum distance the lift must travel from its current floor to cover the range between the lowest and highest service floors. This is calculated as the distance from the current floor to the nearest end of the service floor range, plus the size of the range itself.
       `movement_cost = min(abs(idx_current - min_idx_service), abs(idx_current - max_idx_service)) + (max_idx_service - min_idx_service)`
    10. Estimate the action cost (board/depart): This is the total number of board actions needed (equal to the count of unboarded goal passengers) plus the total number of depart actions needed (equal to the count of boarded goal passengers).
    11. The total heuristic value is the sum of the estimated movement cost and the estimated action cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Determine floor order and create floor_to_index map
        above_map = {} # Maps a floor to the floor immediately above it
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        self.floor_to_index = {}
        self.index_to_floor = {}

        if all_floors:
            # Find the lowest floor (a floor that is a key in above_map but not a value)
            floors_below = set(above_map.keys())
            floors_above = set(above_map.values())
            lowest_floor = None
            for floor in floors_below:
                if floor not in floors_above:
                    lowest_floor = floor
                    break

            # Handle case with only one floor and no 'above' facts
            if lowest_floor is None and len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]

            # Build ordered list from lowest to highest if lowest_floor was found
            if lowest_floor:
                ordered_floors = []
                current = lowest_floor
                while current in above_map:
                    ordered_floors.append(current)
                    current = above_map[current]
                ordered_floors.append(current) # Add the highest floor

                for i, floor in enumerate(ordered_floors):
                    self.floor_to_index[floor] = i
                    self.index_to_floor[i] = floor
            # else: Domain has floors but no clear linear 'above' structure, or is empty.
            # The heuristic will return infinity if a floor is not found in the map.


        # Store passenger destinations for goal passengers
        self.passenger_destins = {}
        self.goal_passengers = set()

        # Extract goal passengers from goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                passenger = parts[1]
                self.goal_passengers.add(passenger)

        # Extract destinations for all passengers mentioned in static facts (includes goal passengers)
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "destin":
                 p, f = parts[1], parts[2]
                 self.passenger_destins[p] = f


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # If lift location is unknown or floor indexing failed, return infinity
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             return float('inf')

        idx_current = self.floor_to_index[current_lift_floor]

        # Identify floors needing service (pickup or dropoff) for unserved goal passengers
        floors_needing_service = set()
        unboarded_passengers_count = 0
        boarded_passengers_count = 0

        # Track passengers' current state (origin floor or boarded)
        # This assumes a passenger is either at their origin OR boarded OR served.
        # If not served, they must be at origin or boarded.
        passenger_is_boarded = set()
        passenger_origin_floor = {} # {passenger: origin_floor}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                p, f = parts[1], parts[2]
                # Only track state for goal passengers
                if p in self.goal_passengers:
                    passenger_origin_floor[p] = f
            elif parts[0] == "boarded":
                p = parts[1]
                 # Only track state for goal passengers
                if p in self.goal_passengers:
                    passenger_is_boarded.add(p)

        # Iterate through goal passengers to find unserved ones and their needs
        for p in self.goal_passengers:
            if (f"(served {p})") not in state:
                if p in passenger_is_boarded:
                    # Passenger is boarded, needs dropoff at destination
                    if p in self.passenger_destins and self.passenger_destins[p] in self.floor_to_index:
                        floors_needing_service.add(self.passenger_destins[p])
                        boarded_passengers_count += 1
                    # else: boarded passenger with unknown/invalid destination - treat as high cost
                    else:
                         return float('inf')
                elif p in passenger_origin_floor:
                    # Passenger is at origin floor, needs pickup
                    origin_floor = passenger_origin_floor[p]
                    if origin_floor in self.floor_to_index:
                        floors_needing_service.add(origin_floor)
                        unboarded_passengers_count += 1
                    # else: passenger at unknown/invalid origin floor - treat as high cost
                    else:
                         return float('inf')
                # else: passenger not served, not at origin, not boarded? Invalid state?
                # Treat as high cost
                else:
                     return float('inf')


        # If no floors need service, and we reached here, it means all goal passengers are served
        # (handled by the initial goal check).
        if not floors_needing_service:
             return 0

        # Calculate movement cost
        indices_service = {self.floor_to_index[f] for f in floors_needing_service}
        min_idx_service = min(indices_service)
        max_idx_service = max(indices_service)

        # Movement cost is distance to nearest end of the range + range size
        dist_to_min = abs(idx_current - min_idx_service)
        dist_to_max = abs(idx_current - max_idx_service)
        range_size = max_idx_service - min_idx_service

        movement_cost = min(dist_to_min, dist_to_max) + range_size


        # Calculate action cost (board/depart)
        # Each unboarded passenger needs a board action.
        # Each boarded passenger needs a depart action.
        action_cost = unboarded_passengers_count + boarded_passengers_count

        # Total heuristic is the sum of movement and action costs
        total_cost = movement_cost + action_cost

        return total_cost
