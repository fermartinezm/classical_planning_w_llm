from fnmatch import fnmatch

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    # Ensure fact is a string before attempting to slice
    if not isinstance(fact, str):
        return False # Cannot match non-string facts

    parts = fact[1:-1].split()
    # Check if the number of parts matches the number of args, unless args contains wildcards that consume multiple parts (not the case here)
    # A simpler check is just to zip and compare
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


# Define the heuristic class
# Inherit from Heuristic base class if available in the planning framework
# class miconicHeuristic(Heuristic):
class miconicHeuristic:
    """
    A domain-dependent heuristic for the Miconic domain.

    Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the number of board actions needed (for unboarded passengers at their origin),
    the number of depart actions needed (for boarded passengers at their destination),
    and estimates the minimum number of lift movements required to visit all necessary floors.

    Assumptions
    - Floors are linearly ordered as defined by the 'above' predicates.
    - The cost of each action (board, depart, up, down) is 1.
    - Passengers must be picked up at their origin and dropped off at their destination.
    - The lift can only carry passengers that have boarded.
    - All passengers requiring service are listed in the goal conditions.

    Heuristic Initialization
    - Parses 'above' predicates from static facts to create a mapping from floor names to integer indices.
    - Parses 'destin' predicates from static facts to store the destination floor for each passenger.
    - Identifies all passengers that need to be served from the goal conditions.

    Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state satisfies the goal conditions. If yes, the heuristic is 0.
    2. Identify the current floor of the lift from the state.
    3. Identify the status of each passenger (served, boarded, or at origin) from the state.
    4. Determine the set of floors the lift *must* visit to serve unserved passengers:
       - The origin floor for each unserved, not-boarded passenger.
       - The destination floor for each unserved, boarded passenger.
    5. Count the number of 'board' actions needed (number of unserved, not-boarded passengers).
    6. Count the number of 'depart' actions needed (number of unserved, boarded passengers).
    7. If there are floors that need visiting (from step 4), estimate the minimum number of 'move' actions (up/down) required. A simple estimate is the maximum absolute difference between the current floor's index and the indices of all needed floors. If no floors need visiting, the move estimate is 0.
    8. The total heuristic value is the sum of needed board actions, needed depart actions, and estimated move actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal information."""
        static_facts = task.static

        # 1. Build floor mapping (name -> index)
        above_facts = [get_parts(fact) for fact in static_facts if match(fact, "above", "*", "*")]

        all_floors = set()
        floors_with_something_above_them = set()
        above_map = {} # f1 -> f2 if (above f1 f2)

        for _, f1, f2 in above_facts:
            all_floors.add(f1)
            all_floors.add(f2)
            floors_with_something_above_them.add(f2)
            above_map[f1] = f2

        self.floor_to_index = {}
        self.index_to_floor = {}

        if all_floors:
            # The lowest floor is one that is in all_floors but not in floors_with_something_above_them
            # Assumes there is exactly one such floor forming the base of the tower
            lowest_floor_candidates = all_floors - floors_with_something_above_them
            if lowest_floor_candidates:
                lowest_floor = lowest_floor_candidates.pop()

                # Build the ordered list of floors and the mapping
                current_floor = lowest_floor
                index = 1
                while current_floor is not None:
                    self.floor_to_index[current_floor] = index
                    self.index_to_floor[index] = current_floor
                    current_floor = above_map.get(current_floor)
                    index += 1
            # else: No clear lowest floor found, floor mapping will be empty.
            # This might happen in malformed problems or if floors are not linearly ordered.


        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.passenger_destinations[passenger] = floor

        # 3. Identify all passengers that need to be served from the goal conditions
        self.all_passengers = set()
        # Assuming the goal is a conjunction of served predicates or a single served predicate
        goal_facts = []
        # task.goals can be a single fact string or a frozenset of fact strings
        if isinstance(task.goals, frozenset):
             goal_facts = list(task.goals) # Assume it's a set of individual goal facts
        elif isinstance(task.goals, str) and match(task.goals, "and", "*"):
             # Crude parsing: remove (and ) and the final ) and split by )(
             goal_str = task.goals[5:-1].strip()
             if goal_str: # Handle empty goal (and)
                 # Split by ') (' and add back parentheses
                 goal_facts = ['(' + fact_str + ')' for fact_str in goal_str.split(') (')]
        elif isinstance(task.goals, str):
             goal_facts = [task.goals] # Single goal fact

        for goal_fact_str in goal_facts:
             parts = get_parts(goal_fact_str)
             if parts and parts[0] == 'served':
                 if len(parts) > 1:
                    self.all_passengers.add(parts[1])
             # Add other goal types if necessary, but 'served' is the main one here.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal is reached
        # The goal is that all passengers in self.all_passengers are served.
        # Check if '(served p)' is in state for all p in self.all_passengers
        if all(f'(served {p})' in state for p in self.all_passengers):
             return 0 # Goal state

        # 2. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # If lift-at is not found, state is likely invalid or initial state parsing failed.
        # Assuming lift-at is always present in valid states.
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # This state is likely unreachable or malformed for this domain
             # Return a large value indicating it's far from goal or invalid
             return float('inf') # Or some large integer

        current_floor_idx = self.floor_to_index[current_lift_floor]

        # 3. Identify passenger status
        passengers_at_origin = {} # passenger -> floor
        passengers_boarded = set()
        passengers_served = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any
            if parts[0] == 'origin' and len(parts) == 3:
                passengers_at_origin[parts[1]] = parts[2]
            elif parts[0] == 'boarded' and len(parts) == 2:
                passengers_boarded.add(parts[1])
            elif parts[0] == 'served' and len(parts) == 2:
                passengers_served.add(parts[1])

        # 4. Determine needed floors and count actions
        needed_pickup_floors = set()
        needed_dropoff_floors = set()
        board_actions_needed = 0
        depart_actions_needed = 0

        for passenger in self.all_passengers:
            if passenger not in passengers_served:
                if passenger in passengers_at_origin: # Unserved, not boarded, at origin
                    needed_pickup_floors.add(passengers_at_origin[passenger])
                    board_actions_needed += 1
                elif passenger in passengers_boarded: # Unserved, boarded
                    # Passenger must have a destination defined in static facts
                    dest_floor = self.passenger_destinations.get(passenger)
                    if dest_floor: # Should always exist for passengers in goal
                       needed_dropoff_floors.add(dest_floor)
                       depart_actions_needed += 1
                    # else: passenger boarded but no destination? Invalid state?

        all_needed_floors = needed_pickup_floors.union(needed_dropoff_floors)

        # 5. Calculate move estimate
        move_estimate = 0
        if all_needed_floors: # Only calculate moves if there are floors to visit
            needed_indices = [self.floor_to_index[f] for f in all_needed_floors if f in self.floor_to_index] # Ensure floor exists in mapping
            if needed_indices: # Check if any valid floors were found
                min_needed_idx = min(needed_indices)
                max_needed_idx = max(needed_indices)

                # Estimate moves as distance to furthest needed floor
                dist_to_min = abs(current_floor_idx - min_needed_idx)
                dist_to_max = abs(current_floor_idx - max_needed_idx)
                move_estimate = max(dist_to_min, dist_to_max)
            # else: all_needed_floors contained floors not in the floor mapping? Invalid state? Move estimate remains 0.


        # 6. Total heuristic
        total_cost = board_actions_needed + depart_actions_needed + move_estimate

        return total_cost
