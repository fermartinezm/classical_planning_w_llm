from fnmatch import fnmatch
# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # This line should be present in the actual environment

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    The heuristic estimates the number of actions needed to serve all passengers.
    It combines the estimated travel cost for the lift to visit all necessary
    floors with the number of board and deboard actions required for unserved
    passengers.

    # Assumptions
    - Passengers are either waiting at their origin, boarded in the lift, or served.
    - The lift has sufficient capacity for all passengers.
    - Actions have unit cost.
    - The floor structure is a linear sequence defined by `(above f_i f_j)` facts.

    # Heuristic Initialization
    - Parses the `(above f_i f_j)` facts to create an ordered list of floors
      and a mapping from floor names to integer indices.
    - Parses the `(destin p f)` facts to store the destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers who are not yet served.
    2. For each unserved passenger:
       - If the passenger is waiting at their origin floor, their origin floor is a required pickup stop.
       - If the passenger is boarded in the lift, their destination floor is a required dropoff stop.
    3. Collect the set of all unique required pickup and dropoff floors.
    4. If there are no required floors, the heuristic value is 0 (all relevant passengers are served or don't require lift action).
    5. Map the current lift floor and all required floors to their integer indices based on the floor order.
    6. Calculate the estimated travel cost: This is the minimum distance the lift must travel to reach one extreme of the required floors' range from its current position, plus the distance needed to sweep across the entire range of required floors.
       - Let `current_idx` be the index of the lift's current floor.
       - Let `min_req_idx` and `max_req_idx` be the minimum and maximum indices among the required floors.
       - Estimated travel cost = `min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)`.
    7. Calculate the estimated board/deboard cost:
       - Count the number of unserved passengers who are waiting at their origin (`N_waiting_unserved`). Each needs one `board` action.
       - Count the number of unserved passengers who are boarded (`N_boarded_unserved`). Each needs one `deboard` action.
       - Estimated board/deboard cost = `N_waiting_unserved + N_boarded_unserved`.
    8. The total heuristic value is the sum of the estimated travel cost and the estimated board/deboard cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        self.static = task.static

        # Parse floor order from (above f_i f_j) facts
        above_facts = [fact for fact in self.static if match(fact, "above", "*", "*")]
        floor_pairs = [(get_parts(fact)[1], get_parts(fact)[2]) for fact in above_facts]

        # Find the lowest floor (a floor that is not the second argument of any 'above' fact)
        all_floors = set()
        floors_above_something = set()
        # Create a mapping from floor_below -> floor_above
        above_map = {}
        for f1, f2 in floor_pairs:
            all_floors.add(f1)
            all_floors.add(f2)
            floors_above_something.add(f2)
            above_map[f1] = f2

        lowest_floor = None
        for floor in all_floors:
            if floor not in floors_above_something:
                lowest_floor = floor
                break

        # Build the ordered list of floors
        self.ordered_floors = []
        self.floor_to_index = {}

        # Handle case with no floors or single floor or complex structure
        if lowest_floor is None:
             if all_floors:
                 if len(all_floors) == 1:
                     lowest_floor = list(all_floors)[0]
                 else:
                     # Fallback: Sort floors alphabetically if order cannot be determined
                     # print("Warning: Could not determine lowest floor from 'above' facts. Using alphabetical sort.")
                     self.ordered_floors = sorted(list(all_floors))
                     self.floor_to_index = {f: i for i, f in enumerate(self.ordered_floors)}
                     self._parse_destinations()
                     return # Skip the rest of floor parsing
             else:
                 # No floors found at all
                 # print("Warning: No floors found in 'above' facts.")
                 self.ordered_floors = []
                 self.floor_to_index = {}
                 self._parse_destinations()
                 return # Skip the rest of floor parsing


        # Build the ordered list of floors starting from the lowest
        current_floor = lowest_floor
        index = 0
        visited_floors = set() # To detect cycles

        while current_floor is not None and current_floor in all_floors:
            if current_floor in visited_floors:
                 # print(f"Warning: Cyclic 'above' facts detected involving floor {current_floor}. Stopping floor parsing.")
                 break
            visited_floors.add(current_floor)
            self.ordered_floors.append(current_floor)
            self.floor_to_index[current_floor] = index
            index += 1
            current_floor = above_map.get(current_floor) # Get the floor directly above

        # Parse passenger destinations from (destin p f) facts
        self._parse_destinations()

    def _parse_destinations(self):
         self.destinations = {}
         for fact in self.static:
             if match(fact, "destin", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3: # Ensure correct number of parts
                     _, passenger, floor = parts
                     self.destinations[passenger] = floor
                 # else: print(f"Warning: Malformed destin fact: {fact}") # Suppress


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        current_lift_floor = None
        waiting_passengers = {} # {passenger: origin_floor}
        boarded_passengers = set()
        served_passengers = set()

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == "lift-at":
                if len(parts) == 2:
                    current_lift_floor = parts[1]
            elif predicate == "origin":
                if len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    waiting_passengers[passenger] = floor
            elif predicate == "boarded":
                if len(parts) == 2:
                    passenger = parts[1]
                    boarded_passengers.add(passenger)
            elif predicate == "served":
                if len(parts) == 2:
                    passenger = parts[1]
                    served_passengers.add(passenger)

        # Identify unserved passengers
        # We need the set of all passengers to find unserved ones.
        # Passengers appear in origin, destin (static), boarded, or served facts.
        all_passengers = set(waiting_passengers.keys()) | boarded_passengers | served_passengers | set(self.destinations.keys())
        unserved_passengers = all_passengers - served_passengers

        # Identify required floors and count pending actions
        required_floors = set()
        n_waiting_unserved = 0
        n_boarded_unserved = 0

        for passenger in unserved_passengers:
            if passenger in waiting_passengers:
                # Passenger is waiting at origin, needs pickup
                origin_floor = waiting_passengers[passenger]
                if origin_floor in self.floor_to_index: # Ensure floor is valid
                    required_floors.add(origin_floor)
                    n_waiting_unserved += 1
            elif passenger in boarded_passengers:
                # Passenger is boarded, needs dropoff at destination
                if passenger in self.destinations:
                    destin_floor = self.destinations[passenger]
                    if destin_floor in self.floor_to_index: # Ensure floor is valid
                        required_floors.add(destin_floor)
                        n_boarded_unserved += 1
                # else: Passenger is boarded but has no destination? Malformed problem?

        # If no required floors, all relevant passengers are served or don't need lift
        if not required_floors:
            return 0

        # Calculate estimated travel cost
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # This state might be unreachable or malformed if lift-at is missing/invalid
             # Return a simplified cost (only board/deboard actions)
             # print(f"Warning: Current lift floor {current_lift_floor} is invalid or not found in floor list. Cannot calculate travel cost accurately.") # Suppress
             return n_waiting_unserved + n_boarded_unserved


        current_idx = self.floor_to_index[current_lift_floor]
        required_indices = {self.floor_to_index[f] for f in required_floors}
        min_req_idx = min(required_indices)
        max_req_idx = max(required_indices)

        # Estimate travel as distance to nearest required extreme + distance to sweep range
        travel_cost = min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)

        # Calculate estimated board/deboard cost
        # Each waiting passenger needs 1 board action
        # Each boarded passenger needs 1 deboard action
        board_deboard_cost = n_waiting_unserved + n_boarded_unserved

        # Total heuristic is sum of estimated travel and estimated actions at floors
        heuristic_value = travel_cost + board_deboard_cost

        return heuristic_value
