from fnmatch import fnmatch
# Assume Heuristic base class is provided elsewhere, e.g., in heuristics/heuristic_base.py
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe return empty list or raise error
        # print(f"Warning: Unexpected fact format: {fact}")
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    It counts the number of 'board' actions needed (for waiting passengers),
    the number of 'depart' actions needed (for boarded passengers), and
    estimates the lift movement cost to visit all necessary floors (origin floors
    of waiting passengers and destination floors of boarded passengers).

    # Assumptions
    - Floors are linearly ordered, defined by the 'above' predicate.
    - The cost of each action (move, board, depart) is 1.
    - The lift can carry multiple passengers simultaneously.
    - The estimated movement cost is the distance to the nearest required extreme floor
      plus the distance spanning the range of all required floors.

    # Heuristic Initialization
    - Parses 'above' facts from static information to determine the linear order
      of floors and create a mapping from floor names to integer indices (0-based).
    - Extracts destination floors for all passengers from static information.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the goal state is reached. If all passengers are marked as 'served', the heuristic is 0.
    2. Identify the lift's current floor from the state and find its corresponding integer index using the pre-calculated floor mapping. If the floor is unknown, return infinity.
    3. Iterate through the state facts to identify passengers by their status: 'served', 'waiting' (origin fact), or 'boarded'. Store origin floors for waiting passengers.
    4. Identify the set of unserved passengers by subtracting the 'served' passengers from the total set of passengers (derived from static destinations and goals).
    5. Filter the waiting and boarded passengers identified in step 3 to include only those who are unserved.
    6. Collect the floor indices for required stops:
       - For each unserved waiting passenger, add their origin floor index (retrieved from step 3) to the set of required pickup floor indices. Handle cases where origin floor is unknown or not in the floor map.
       - For each unserved boarded passenger, find their destination floor index (using pre-calculated destinations from initialization) and add it to the set of required dropoff floor indices. Handle cases where destination is unknown or not in the floor map.
    7. Calculate the action cost: This is the sum of the number of unserved waiting passengers (each needs a 'board' action) and the number of unserved boarded passengers (each needs a 'depart' action).
    8. Determine the set of all required floor indices by combining the required pickup and dropoff floor indices.
    9. Calculate the movement cost:
       - If the set of required floor indices is empty, the movement cost is 0.
       - If not empty, find the minimum (`min_idx`) and maximum (`max_idx`) indices among the required floors.
       - The estimated movement cost is calculated as:
         `min(abs(current_floor_index - min_idx), abs(current_floor_index - max_idx)) + (max_idx - min_idx)`
         This formula estimates the minimum moves to reach either the lowest or highest required floor from the current position, plus the moves needed to traverse the distance between the lowest and highest required floors.
    10. The total heuristic value is the sum of the action cost and the movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Determine floor order and create floor_to_index mapping
        all_floors = set()
        above_map = {} # f_below -> f_above
        below_map = {} # f_above -> f_below

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_below, f_above = parts[1], parts[2]
                all_floors.add(f_below)
                all_floors.add(f_above)
                above_map[f_below] = f_above
                below_map[f_above] = f_below

        self.floor_order = []
        self.floor_to_index = {}

        if all_floors:
            # Find the bottom floor (a floor that no other floor is immediately above)
            bottom_floor = None
            # Check floors mentioned in 'above' facts
            floors_in_above = set(above_map.keys()).union(set(above_map.values()))
            for floor in floors_in_above:
                 if floor not in below_map: # This floor is not immediately above anything else
                     bottom_floor = floor
                     break

            # If a bottom floor was found using 'above' facts, build the ordered list
            if bottom_floor is not None:
                current = bottom_floor
                while current is not None:
                    self.floor_order.append(current)
                    current = above_map.get(current)
            else:
                 # Fallback: If 'above' facts didn't define a clear linear order or were missing,
                 # try to find all floors from any relevant facts and sort them by name if possible.
                 potential_floors = set()
                 for fact in static_facts:
                     parts = get_parts(fact)
                     if len(parts) > 1 and parts[0] in ['lift-at', 'origin', 'destin', 'above']:
                         for part in parts[1:]:
                             # Simple check if it looks like a floor name (starts with f)
                             if isinstance(part, str) and part.startswith('f'):
                                 potential_floors.add(part)

                 # Also check initial state for floors if static didn't provide them
                 if not potential_floors and task.initial_state:
                      for fact in task.initial_state:
                          parts = get_parts(fact)
                          if len(parts) > 1 and parts[0] in ['lift-at', 'origin', 'destin']:
                              for part in parts[1:]:
                                  if isinstance(part, str) and part.startswith('f'):
                                      potential_floors.add(part)

                 if potential_floors:
                     try:
                         # Attempt to sort floors like f1, f10, f2 numerically
                         sorted_floors = sorted(list(potential_floors), key=lambda f: int(f[1:]))
                         self.floor_order = sorted_floors
                     except (ValueError, IndexError):
                         # If names are not f<number> or list is empty, just use the set (order won't be meaningful)
                         self.floor_order = list(potential_floors)
                 # else: self.floor_order remains empty

            # Create floor_to_index mapping from the determined order
            self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_order)}

        # 2. Extract passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                passenger, destination_floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = destination_floor

        # Get all possible passengers from destinations and goals
        self.all_passengers = set(self.passenger_destinations.keys())
        # Add passengers from goals, just in case a passenger is in goal but not static destin (unlikely for miconic)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served" and len(parts) == 2:
                  self.all_passengers.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Get lift's current floor index
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        current_floor_idx = self.floor_to_index.get(current_floor)
        # If lift location is unknown or floor is not in our map, heuristic is infinite
        if current_floor_idx is None:
             # This should not happen in a valid state/domain with correctly parsed floors
             # print(f"Error: Lift at unknown floor '{current_floor}' or floor map is empty/incomplete.")
             return float('inf') # Cannot compute heuristic


        # 2. Identify passengers by status
        served_passengers_in_state = set()
        waiting_passengers_in_state = set()
        boarded_passengers_in_state = set()
        passenger_origin_floors = {} # Store origin floors for waiting passengers

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "served" and len(parts) == 2:
                served_passengers_in_state.add(parts[1])
            elif predicate == "origin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                waiting_passengers_in_state.add(p)
                passenger_origin_floors[p] = f
            elif predicate == "boarded" and len(parts) == 2:
                 p = parts[1]
                 boarded_passengers_in_state.add(p)


        # 3. Identify unserved passengers and required stops
        unserved_passengers = self.all_passengers - served_passengers_in_state

        waiting_passengers = waiting_passengers_in_state.intersection(unserved_passengers)
        boarded_passengers = boarded_passengers_in_state.intersection(unserved_passengers)

        # If no unserved passengers, goal is reached (already checked above, but defensive)
        if not unserved_passengers:
             return 0

        pickup_floor_indices = set()
        dropoff_floor_indices = set()

        for p in waiting_passengers:
             origin_f = passenger_origin_floors.get(p)
             if origin_f:
                 f_idx = self.floor_to_index.get(origin_f)
                 if f_idx is not None:
                     pickup_floor_indices.add(f_idx)
             # else: Origin floor for waiting passenger not found or not in floor map - problem with state/domain

        for p in boarded_passengers:
             dest_f = self.passenger_destinations.get(p)
             if dest_f:
                 dest_f_idx = self.floor_to_index.get(dest_f)
                 if dest_f_idx is not None:
                     dropoff_floor_indices.add(dest_f_idx)
             # else: Destination for boarded passenger not found or not in floor map - problem with static/domain


        # 4. Calculate action costs (board + depart)
        action_cost = len(waiting_passengers) + len(boarded_passengers)

        # 5. Calculate movement cost
        required_floor_indices = pickup_floor_indices.union(dropoff_floor_indices)

        movement_cost = 0
        if required_floor_indices:
            min_idx = min(required_floor_indices)
            max_idx = max(required_floor_indices)

            # Estimate movement cost
            # This is min(|curr - min_req|, |curr - max_req|) + (max_req - min_req)
            # It represents the distance to the nearest extreme required floor
            # plus the distance needed to traverse the entire range of required floors.
            movement_cost = min(abs(current_floor_idx - min_idx), abs(current_floor_idx - max_idx)) + (max_idx - min_idx)

        # Total heuristic is sum of action costs and movement cost
        total_heuristic = action_cost + movement_cost

        return total_heuristic
