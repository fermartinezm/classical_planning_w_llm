from fnmatch import fnmatch
# Assuming heuristic_base is available in the specified path
from heuristics.heuristic_base import Heuristic


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
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
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of board actions needed, the number of depart actions needed,
    and the estimated number of move actions for the lift to visit all necessary floors.

    # Assumptions
    - Floors are arranged linearly and ordered by the `above` predicate.
    - Passenger destinations are fixed and available from the initial state or static facts.
    - The cost of each action (board, depart, move) is 1.

    # Heuristic Initialization
    - Parse the `above` facts from static information to build an ordered list of floors
      and a mapping from floor name to its index (level).
    - Parse the `destin` facts from the initial state and static information
      to build a mapping from passenger name to their destination floor.
    - Identify all passengers involved in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check for Goal State: If all passengers are served (i.e., `(served p)` is true for all passengers), the heuristic is 0.
    2. Find Current Lift Location: Identify the floor where the lift is currently located from the state facts `(lift-at ?f)`.
    3. Identify Unserved Passengers and Required Floors: Iterate through all known passengers. If a passenger `p` is not marked as `(served p)`:
       - If `(origin p f)` is true in the state, the passenger is waiting at floor `f`. Increment the count of board actions needed and add `f` to the set of floors the lift must visit for pickup.
       - If `(boarded p)` is true in the state, the passenger is inside the lift. Increment the count of depart actions needed and add the passenger's destination floor (looked up from the pre-calculated destination map) to the set of floors the lift must visit for dropoff.
    4. Calculate Estimated Move Actions:
       - If the set of required floors (pickup or dropoff) is empty, the estimated move cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the required floors using the pre-calculated floor-to-index map.
       - The estimated move cost is calculated as the minimum distance from the current lift floor's index to either the minimum or maximum required floor index, plus the total distance between the minimum and maximum required floor indices. This represents the travel needed to reach the range of necessary floors and traverse that range.
    5. Sum Costs: The total heuristic value is the sum of the total number of board actions needed, the total number of depart actions needed, and the estimated move actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state for destin facts

        # 1. Parse floors and build floor_to_index map
        above_map = {}
        all_floors = set()
        floors_that_are_above_another = set() # Floors that appear as the second argument in (above f_lower f_higher)

        # Collect all floors and the 'above' relationships
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_lower, f_higher = parts[1], parts[2]
                above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                floors_that_are_above_another.add(f_higher)

        # Find the lowest floor (a floor that is in all_floors but is NOT a 'higher' floor in any 'above' fact)
        lowest_floor = None
        # Check if all_floors is not empty to avoid issues with empty domains
        if all_floors:
            potential_lowest = all_floors - floors_that_are_above_another
            if len(potential_lowest) == 1:
                 lowest_floor = list(potential_lowest)[0]
            elif len(all_floors) == 1 and not above_map:
                 # Case with only one floor and no 'above' facts
                 lowest_floor = list(all_floors)[0]
            # Note: If len(potential_lowest) > 1, it implies multiple lowest floors or disconnected components.
            # Assuming a single linear tower for miconic, this case indicates an unusual problem structure.
            # The heuristic will proceed with lowest_floor = None if not found, leading to potential inf cost.


        self.floor_to_index = {}
        self.ordered_floors = []
        if lowest_floor:
            current_f = lowest_floor
            index = 0
            # Traverse the chain upwards
            while current_f is not None:
                self.floor_to_index[current_f] = index
                self.ordered_floors.append(current_f)
                index += 1
                current_f = above_map.get(current_f) # Get the floor immediately above

        # 2. Parse passenger destinations
        self.passenger_to_destin = {}
        # Destinations are typically in the initial state facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "destin":
                 passenger, floor = parts[1], parts[2]
                 self.passenger_to_destin[passenger] = floor

        # Also check static facts, although less common for dynamic info like destin
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "destin":
                 passenger, floor = parts[1], parts[2]
                 # Initial state takes precedence if duplicated
                 if passenger not in self.passenger_to_destin:
                     self.passenger_to_destin[passenger] = floor


        # 3. Identify all passengers involved in the problem
        # Passengers are those appearing in origin or destin facts in the initial state
        self.all_passengers = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "origin":
                 self.all_passengers.add(parts[1])
             if parts and parts[0] == "destin":
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check for Goal State
        # A state is a goal state if all passengers are served.
        all_served = True
        for passenger in self.all_passengers:
            if f"(served {passenger})" not in state:
                all_served = False
                break
        if all_served:
            return 0 # Heuristic is 0 at goal state

        # 2. Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # If lift location is not found, state is likely invalid or terminal (not goal)
        # Return infinity as it's not a solvable state from here.
        if current_lift_floor is None:
             return float('inf')

        current_idx = self.floor_to_index.get(current_lift_floor)
        # If lift is at a floor not in our map (shouldn't happen in valid problems)
        if current_idx is None:
             return float('inf')


        # 3. Identify unserved passengers and required floors
        passengers_at_origin = set()
        passengers_boarded = set()
        required_floors = set()

        for passenger in self.all_passengers:
            # Check if passenger is NOT served
            if f"(served {passenger})" not in state:
                is_boarded = f"(boarded {passenger})" in state

                if is_boarded:
                    passengers_boarded.add(passenger)
                    # Need to drop off at destination
                    dest_floor = self.passenger_to_destin.get(passenger)
                    if dest_floor: # Passenger must have a destination
                        required_floors.add(dest_floor)
                    # else: Invalid state - boarded passenger with no destination?
                else:
                    # Passenger is not boarded and not served, must be at origin
                    origin_floor = None
                    # Find origin floor from current state
                    for fact in state:
                        parts = get_parts(fact)
                        if parts and parts[0] == "origin" and parts[1] == passenger:
                            origin_floor = parts[2]
                            break
                    if origin_floor: # Passenger must have an origin if not boarded/served
                        passengers_at_origin.add(passenger)
                        required_floors.add(origin_floor)
                    # else: Invalid state - unserved, not boarded, not at origin?

        # 4. Calculate action counts
        board_actions_needed = len(passengers_at_origin)
        depart_actions_needed = len(passengers_boarded)

        # 5. Calculate estimated move actions
        estimated_moves = 0
        # Only calculate moves if there are floors to visit
        if required_floors:
            # Get indices for required floors, filter out any floors not in our map (shouldn't happen)
            req_indices = sorted([self.floor_to_index[f] for f in required_floors if f in self.floor_to_index])

            if req_indices: # Ensure we have valid indices
                min_req_idx = req_indices[0]
                max_req_idx = req_indices[-1]

                # Distance from current floor to the closest end of the required range
                dist_to_min = abs(current_idx - min_req_idx)
                dist_to_max = abs(current_idx - max_req_idx)
                travel_to_range = min(dist_to_min, dist_to_max)

                # Distance to traverse the entire required range
                range_distance = max_req_idx - min_req_idx

                estimated_moves = travel_to_range + range_distance

        # Total heuristic is sum of actions
        # Each board/depart is 1 action. Each move between adjacent floors is 1 action.
        total_cost = board_actions_needed + depart_actions_needed + estimated_moves

        return total_cost
