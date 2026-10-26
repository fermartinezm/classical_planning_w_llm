from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential non-string inputs or malformed facts gracefully
        # (though typically state facts are strings)
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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums two components:
    1. The number of unserved passengers (each requires at least one board and one depart action).
    2. An estimate of the vertical movement actions the lift must perform to visit all floors where pickups or dropoffs are needed.

    # Assumptions
    - The floors are ordered linearly based on the `above` predicates.
    - Each unserved passenger needs to be boarded (if not already) and then depart at their destination.
    - The lift can carry multiple passengers.
    - The cost of move, board, and depart actions is 1.

    # Heuristic Initialization
    - Parses the static facts to determine the linear ordering of floors and creates a mapping from floor object names to integer floor numbers (1-based).
    - Parses the static facts to store the destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current Lift Location:** Find the floor where the lift is currently located from the state facts.
    2.  **Identify Unserved Passengers:** Determine which passengers are not yet marked as `served`.
    3.  **Count Board/Depart Actions:** The number of unserved passengers represents the minimum number of board/depart actions remaining (each unserved passenger needs one final `depart` action, and one `board` action if not already boarded). This count is the first component of the heuristic.
    4.  **Identify Required Stops:** For each unserved passenger:
        - If the passenger is waiting at an origin floor (`origin ?p ?f_orig`), the lift must stop at `f_orig` for pickup.
        - If the passenger is boarded (`boarded ?p`), the lift must stop at their destination floor (`destin ?p ?f_dest`) for dropoff.
        Collect all these required pickup and dropoff floors into a set (`Floors_to_stop`).
    5.  **Estimate Movement Actions:**
        - If `Floors_to_stop` is empty, no movement is needed for unserved passengers (this should only happen in the goal state). Movement cost is 0.
        - If `Floors_to_stop` is not empty:
            - Find the minimum and maximum floor numbers among the `Floors_to_stop`.
            - Calculate the vertical span covered by these floors (`max_stop_num - min_stop_num`).
            - Calculate the distance from the current lift floor to the closest end of this span (`min(abs(current_floor_num - min_stop_num), abs(current_floor_num - max_stop_num))`).
            - The estimated movement cost is the span plus the distance to reach the span: `(max_stop_num - min_stop_num) + min(abs(current_floor_num - min_stop_num), abs(current_floor_num - max_stop_num))`. This estimates the moves needed to traverse the entire range of required floors, starting from the current position.
    6.  **Sum Components:** The total heuristic value is the sum of the count from step 3 and the estimated movement cost from step 5.
    7.  **Goal State:** If all passengers are served, the set of unserved passengers is empty, `Floors_to_stop` is empty, and the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Determine floor ordering and create floor_to_num mapping
        above_map = {} # Maps lower floor to immediately higher floor
        all_floors = set()
        floors_as_first_arg = set() # Floors that are the first arg of 'above' (higher floors)
        floors_as_second_arg = set() # Floors that are the second arg of 'above' (lower floors)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above' and len(parts) == 3:
                f_higher, f_lower = parts[1], parts[2]
                above_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                floors_as_first_arg.add(f_higher)
                floors_as_second_arg.add(f_lower)

        # Find the lowest floor: it appears as a second arg but never a first arg
        lowest_floor = None
        # Iterate through floors that are second args (potential lower floors)
        for f_lower in floors_as_second_arg:
             # Check if this floor is ever a first arg (meaning something is below it)
             if f_lower not in floors_as_first_arg:
                 lowest_floor = f_lower
                 break # Found the unique lowest floor

        if lowest_floor is None and all_floors:
             # Handle case with only one floor or circular 'above' (shouldn't happen in valid PDDL)
             # Or if the lowest floor is not involved in any 'above' as a second arg
             # A safer way: find a floor that is a second arg but not a first arg.
             # If that fails, maybe find a floor that is a second arg and its higher floor is a first arg, etc.
             # A simpler approach assuming a linear chain: find the floor that is never the first argument.
             potential_lowest = floors_as_second_arg - floors_as_first_arg
             if potential_lowest:
                 lowest_floor = potential_lowest.pop() # Assuming there's only one lowest
             elif all_floors:
                 # Fallback for single floor or unusual cases - pick any floor?
                 # Or assume the one not in above_map values is lowest?
                 # Let's assume linear chain and the above logic finds it.
                 # If still None, it might be a single floor problem or malformed.
                 # For robustness, if above_map is empty but there are floors, assume one floor.
                 if not above_map and all_floors:
                     lowest_floor = list(all_floors)[0] # Just pick one if no above facts

        if lowest_floor is None and all_floors:
             # Another fallback: find a floor that is a second arg but its higher floor is not in the map keys
             # This handles cases like (above f2 f1) where f1 is lowest but not a second arg of anything else.
             # Let's stick to the simpler logic based on the example structure.
             # The logic "appears as second arg but never first arg" works for the examples.
             pass # lowest_floor remains None if not found by the primary method

        # Build the ordered list of floors starting from the lowest
        ordered_floors = []
        current_floor = lowest_floor
        floor_rank = 1
        self.floor_to_num = {}

        # Traverse upwards from the lowest floor
        # Need a reverse map to traverse upwards: higher_to_lower
        lower_to_higher = {}
        higher_to_lower = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above' and len(parts) == 3:
                f_higher, f_lower = parts[1], parts[2]
                lower_to_higher[f_lower] = f_higher
                higher_to_lower[f_higher] = f_lower # Build reverse map

        # Find the lowest floor again using the reverse map logic
        # A floor 'f' is lowest if it is a value in higher_to_lower but not a key
        potential_lowest_set = set(higher_to_lower.values()) - set(higher_to_lower.keys())
        if potential_lowest_set:
             lowest_floor = potential_lowest_set.pop()
        elif all_floors:
             # Handle single floor case or disconnected components (unlikely in miconic)
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # This case might indicate a problem with parsing or domain structure
                 # For robustness, we could try to find any floor and build a partial map,
                 # but assuming connected linear floors, the above should work.
                 pass # lowest_floor remains None

        current_floor = lowest_floor
        floor_rank = 1
        ordered_floors = []
        self.floor_to_num = {}

        # Traverse upwards using the lower_to_higher map
        while current_floor is not None:
            ordered_floors.append(current_floor)
            self.floor_to_num[current_floor] = floor_rank
            floor_rank += 1
            current_floor = lower_to_higher.get(current_floor) # Get the floor immediately above

        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin' and len(parts) == 3:
                passenger, destination_floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = destination_floor

        # Also need passenger objects from goals to know who needs serving
        self.all_passengers = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served' and len(parts) == 2:
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid miconic state, but handle defensively
             return float('inf') # Cannot proceed without lift location

        current_floor_num = self.floor_to_num.get(current_lift_floor)
        if current_floor_num is None:
             # Should not happen if floor mapping is correct, but handle defensively
             return float('inf') # Unknown floor

        # Identify unserved passengers and their status (waiting or boarded)
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        waiting_passengers = {get_parts(fact)[1] for fact in state if match(fact, "origin", "*", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        unserved_passengers = self.all_passengers - served_passengers

        # If all passengers are served, the heuristic is 0
        if not unserved_passengers:
            return 0

        # Heuristic component 1: Number of board/depart actions needed
        # Each unserved passenger needs one final depart action.
        # Each waiting passenger also needs a board action.
        # A simple count of unserved passengers is a lower bound on board/depart actions.
        h_actions = len(unserved_passengers)

        # Heuristic component 2: Movement cost
        floors_to_stop = set()

        # Add origin floors for waiting passengers
        for p in unserved_passengers:
            if p in waiting_passengers:
                 # Find origin floor from state
                 origin_floor = None
                 for fact in state:
                     if match(fact, "origin", p, "*"):
                         origin_floor = get_parts(fact)[2]
                         break
                 if origin_floor:
                     floors_to_stop.add(origin_floor)
                 # else: passenger is unserved but not waiting or boarded? (malformed state)

        # Add destination floors for boarded passengers
        for p in unserved_passengers:
            if p in boarded_passengers:
                 # Destination is in static facts
                 dest_floor = self.passenger_destinations.get(p)
                 if dest_floor:
                     floors_to_stop.add(dest_floor)
                 # else: passenger is boarded but has no destination? (malformed state)

        # If no floors need visiting (should only happen if unserved passengers
        # are in a malformed state or already at destination and boarded?),
        # movement cost is 0.
        if not floors_to_stop:
             # This case might mean unserved passengers are already at their destination
             # and boarded, just need to depart. The h_actions covers the depart cost.
             # Movement cost is 0 in this specific scenario.
             h_movement = 0
        else:
            # Get floor numbers for all floors that need stops
            floors_to_stop_nums = {self.floor_to_num.get(f) for f in floors_to_stop if f in self.floor_to_num}

            if not floors_to_stop_nums:
                 # Should not happen if floors_to_stop is not empty and floor_to_num is correct
                 h_movement = float('inf') # Cannot calculate movement
            else:
                min_stop_num = min(floors_to_stop_nums)
                max_stop_num = max(floors_to_stop_nums)

                # Estimate movement cost: span of floors + distance from current to closest end
                span = max_stop_num - min_stop_num
                dist_to_min = abs(current_floor_num - min_stop_num)
                dist_to_max = abs(current_floor_num - max_stop_num)
                h_movement = span + min(dist_to_min, dist_to_max)

        # Total heuristic is the sum of action count and movement estimate
        total_heuristic = h_actions + h_movement

        return total_heuristic

