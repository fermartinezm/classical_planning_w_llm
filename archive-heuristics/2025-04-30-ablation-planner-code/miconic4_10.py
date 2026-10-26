from fnmatch import fnmatch
# Assume Heuristic base class is available and imported elsewhere if needed
# from heuristics.heuristic_base import Heuristic

# Helper functions
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
    # Basic check for predicate name
    if not parts or (args and not fnmatch(parts[0], args[0])):
        return False
    # Check remaining arguments if provided
    if len(args) > 1:
        if len(parts) < len(args): # Not enough parts to match all args
            return False
        return all(fnmatch(parts[i], args[i]) for i in range(1, len(args)))
    return True # Only predicate was checked and matched


class miconicHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all
    passengers to their destination floors. It considers the actions required
    for each unserved passenger (boarding and departing) and the minimum
    movement cost for the lift to visit all necessary floors (origins for
    waiting passengers and destinations for boarded passengers).

    # Assumptions
    - Floors are linearly ordered by the 'above' predicate.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - All passengers mentioned in the goal state are defined with 'origin' and 'destin' in the static facts.
    - The state representation includes 'lift-at' for the current lift location.

    # Heuristic Initialization
    - Parses the 'above' facts from static information to establish the floor
      order and create a mapping from floor names to integer indices. The lowest
      floor is assigned index 0, and indices increase for floors higher up.
    - Parses 'origin' and 'destin' facts from static information to store
      the required routes for each passenger.
    - Identifies all passengers that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is a goal state. A state is a goal state if all passengers identified in the goal conditions are currently in the 'served' state. If so, the heuristic is 0.
    2. Identify the current floor of the lift by finding the fact matching '(lift-at ?f)' in the state.
    3. Initialize the total estimated action cost (`action_cost`) to 0.
    4. Initialize sets for required pickup floors (`required_pickup_floors`) and required dropoff floors (`required_dropoff_floors`). These sets will store the names of the floors the lift must visit.
    5. Identify the sets of passengers currently 'boarded' and 'served' by examining the state facts.
    6. Iterate through all passengers that need to be served (identified during initialization from the goal state).
    7. For each unserved passenger (a goal passenger not found in the 'served' set):
       - Check if the passenger is currently in the 'boarded' set. If yes:
         - This passenger needs to be dropped off. Look up their destination floor from the passenger routes stored during initialization.
         - Add this destination floor to the `required_dropoff_floors` set.
         - Add 1 to the `action_cost` (representing the 'depart' action needed).
       - Else (the passenger is unserved and not boarded, implying they are waiting at their origin):
         - This passenger needs to be picked up and then dropped off. Look up their origin floor from the passenger routes.
         - Add this origin floor to the `required_pickup_floors` set.
         - Add 2 to the `action_cost` (representing the 'board' and 'depart' actions needed).
    8. Calculate the 'movement cost':
       - Combine `required_pickup_floors` and `required_dropoff_floors` into a single set `required_floors_to_visit`.
       - If `required_floors_to_visit` is empty, the movement cost is 0 (no more floors need visiting for unserved passengers).
       - Otherwise, convert the names of the required floors to their integer indices using the `floor_to_int` mapping created during initialization.
       - Find the minimum (`min_req_int`) and maximum (`max_req_int`) floor indices among the required floors.
       - Get the integer index of the current lift floor (`current_lift_floor_int`).
       - The movement cost is estimated as the distance from the current lift floor index to the closest end of the required range (`min_req_int` or `max_req_int`), plus the span of the required range (`max_req_int - min_req_int`). This formula estimates the minimum travel needed to reach the range of relevant floors and traverse it to visit all required floors within that range.
    9. The total heuristic value for the state is the sum of the calculated `action_cost` and `movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Facts that are not affected by actions.

        # Parse floor order from 'above' facts
        # Map f_higher -> f_lower
        floor_below = {}
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_higher, f_lower = get_parts(fact)[1:]
                floor_below[f_higher] = f_lower
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Create reverse map: f_lower -> f_higher
        floor_above_map = {v: k for k, v in floor_below.items()}

        # Find the lowest floor (the one not appearing as a value in floor_above_map)
        lowest_floor = None
        for floor in all_floors:
             if floor not in floor_above_map:
                  lowest_floor = floor
                  break

        # Build floor_to_int mapping by following the chain up from the lowest floor
        self.floor_to_int = {}
        current_floor = lowest_floor
        floor_index = 0
        while current_floor is not None:
             self.floor_to_int[current_floor] = floor_index
             # Find the floor that is above the current_floor
             next_floor = floor_above_map.get(current_floor)
             current_floor = next_floor
             floor_index += 1

        # Store passenger origin and destination floors from static facts
        self.passenger_routes = {}
        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1:]
                if p not in self.passenger_routes:
                    self.passenger_routes[p] = {}
                self.passenger_routes[p]['origin'] = f
            elif match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1:]
                if p not in self.passenger_routes:
                    self.passenger_routes[p] = {}
                self.passenger_routes[p]['destin'] = f

        # Identify all passengers that need to be served (from goal state)
        self.goal_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 p = get_parts(goal)[1]
                 self.goal_passengers.add(p)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Check if goal is reached
        # A state is a goal state if all goal passengers are served.
        all_goal_passengers_served = True
        for p in self.goal_passengers:
            if f"(served {p})" not in state:
                all_goal_passengers_served = False
                break

        if all_goal_passengers_served:
             return 0

        # Get current lift location
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # This should not happen in a valid state according to domain rules,
        # but return a large value if lift location is unknown.
        if lift_floor is None:
             return float('inf')

        current_lift_floor_int = self.floor_to_int[lift_floor]

        # Identify passengers currently boarded or served
        boarded_passengers_in_state = set()
        served_passengers_in_state = set()

        for fact in state:
            if match(fact, "boarded", "*"):
                p = get_parts(fact)[1]
                boarded_passengers_in_state.add(p)
            elif match(fact, "served", "*"):
                 p = get_parts(fact)[1]
                 served_passengers_in_state.add(p)


        # Identify required pickup and dropoff floors for unserved passengers
        required_pickup_floors = set()
        required_dropoff_floors = set()
        action_cost = 0

        for p in self.goal_passengers:
            if p in served_passengers_in_state:
                continue # Passenger is already served

            if p in boarded_passengers_in_state:
                # Passenger is unserved and boarded, needs depart
                # Get destination from static routes
                destin_floor = self.passenger_routes.get(p, {}).get('destin')
                if destin_floor: # Ensure destin exists
                    required_dropoff_floors.add(destin_floor)
                    action_cost += 1 # Needs depart action
                # else: Invalid state? Passenger boarded but no destination known.
            else:
                 # Passenger is unserved and not boarded, implying waiting at origin
                 # Get origin from static routes
                 origin_floor = self.passenger_routes.get(p, {}).get('origin')
                 if origin_floor: # Ensure origin exists
                     required_pickup_floors.add(origin_floor)
                     action_cost += 2 # Needs board and depart action
                 # else: Invalid state? Passenger unserved/unboarded but no origin known.


        # Calculate movement cost
        required_floors_to_visit = required_pickup_floors.union(required_dropoff_floors)

        movement_cost = 0
        if required_floors_to_visit:
            # Ensure all required floors are in our floor_to_int map
            valid_required_floor_ints = {self.floor_to_int[f] for f in required_floors_to_visit if f in self.floor_to_int}

            if valid_required_floor_ints:
                min_req_int = min(valid_required_floor_ints)
                max_req_int = max(valid_required_floor_ints)

                # Distance from current floor to the closest required floor
                dist_to_min = abs(current_lift_floor_int - min_req_int)
                dist_to_max = abs(current_lift_floor_int - max_req_int)
                dist_to_closest_required = min(dist_to_min, dist_to_max)

                # Span of the required floors
                span = max_req_int - min_req_int

                # Movement cost estimate: get to the range + traverse the range
                movement_cost = dist_to_closest_required + span
            # else: Required floors are somehow not in the floor map - indicates a problem
            # with parsing or domain definition, but for robustness, movement_cost remains 0.


        # Total heuristic is the sum of actions and estimated movements
        total_heuristic = action_cost + movement_cost

        return total_heuristic
