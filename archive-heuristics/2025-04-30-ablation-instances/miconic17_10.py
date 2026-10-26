from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(predicate arg1 arg2)" -> ["predicate", "arg1", "arg2"]
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to transport
    all unserved passengers to their destination floors. It sums the estimated
    cost for each unserved passenger independently, considering their current
    state (waiting at origin or boarded) and the lift's current location.

    # Assumptions
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic calculates the Manhattan distance (absolute difference in floor levels)
      for lift movement.
    - The heuristic assumes that each unserved passenger requires a sequence of
      actions (potentially including lift movement, board, lift movement, depart)
      regardless of other passengers. This is a relaxation and makes the heuristic non-admissible.
    - All unserved passengers are either waiting at their origin floor or are boarded.
    - The floor structure defined by `(above f_higher f_lower)` facts forms a single linear sequence.

    # Heuristic Initialization
    - The heuristic extracts the destination floor for each passenger from the
      static `(destin ?p ?f)` facts.
    - It determines the ordering of floors and assigns a numerical level to each
      floor based on the static `(above ?f_higher ?f_lower)` facts. This allows
      calculating the distance between any two floors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state facts to identify:
       - The current floor of the lift.
       - Which passengers are currently boarded.
       - Which passengers are waiting at their origin floors.
       - Which passengers are already served.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through all passengers who have a defined destination (from static facts).
    4. For each passenger:
       - Check if the passenger is already `served`. If yes, this passenger requires no further actions, so skip them.
       - If the passenger is `boarded`:
         - Get their destination floor.
         - Calculate the movement cost for the lift to travel from its current floor
           to the passenger's destination floor (absolute difference in floor levels).
         - Add this movement cost plus 1 (for the `depart` action) to the total cost.
       - If the passenger is waiting at their `origin` floor:
         - Get their origin floor and their destination floor.
         - Calculate the movement cost for the lift to travel from its current floor
           to the passenger's origin floor.
         - Add this movement cost plus 1 (for the `board` action) to the total cost.
         - Calculate the movement cost for the lift to travel from the passenger's
           origin floor to their destination floor.
         - Add this movement cost plus 1 (for the `depart` action) to the total cost.
       - If the passenger is neither served, boarded, nor at their origin, they are
         ignored in this heuristic calculation (assuming such states are not typical
         or the cost is implicitly handled elsewhere).
    5. The total heuristic value is the sum of the costs calculated for each unserved passenger.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals # Goal conditions are needed to check if a passenger is served
        static_facts = task.static # Static facts contain floor structure and destinations

        self.goal_locations = {}
        # above_relations maps f_lower -> f_higher based on (above f_higher f_lower)
        above_relations = {}

        # Parse static facts to get destinations and above relations
        all_floors_mentioned = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "destin":
                passenger, floor = parts[1], parts[2]
                self.goal_locations[passenger] = floor
                all_floors_mentioned.add(floor)
            elif predicate == "above":
                f_higher, f_lower = parts[1], parts[2]
                above_relations[f_lower] = f_higher
                all_floors_mentioned.add(f_higher)
                all_floors_mentioned.add(f_lower)
            # Floors can also be mentioned in initial state origin/lift-at facts,
            # but static facts should ideally define the full relevant floor structure.


        self.floor_levels = {}

        # Determine floor order and levels
        # Find the lowest floor: a floor that is a f_lower but never a f_higher
        all_above_first_args = set(above_relations.values())
        all_above_second_args = set(above_relations.keys())

        # The lowest floor is in all_above_second_args but not in all_above_first_args
        lowest_floors_candidates = list(all_above_second_args - all_above_first_args)

        lowest_floor = None
        if len(lowest_floors_candidates) == 1:
             lowest_floor = lowest_floors_candidates[0]
        elif not above_relations:
             # Case with only one floor and no above facts
             if len(all_floors_mentioned) == 1:
                 lowest_floor = list(all_floors_mentioned)[0]
             else:
                 # Cannot determine floor order without above facts and multiple floors exist
                 # This heuristic is not applicable.
                 self.floor_levels = {} # Indicate failure to build levels
                 return # Cannot initialize heuristic properly
        else:
             # Handle cases with multiple lowest floors or disconnected floors if necessary
             # For typical miconic, there's one lowest floor.
             if lowest_floors_candidates:
                 lowest_floor = lowest_floors_candidates[0]
             else:
                 # This might happen if all floors form a cycle or there are no floors mentioned in above
                 self.floor_levels = {} # Indicate failure
                 return


        # Traverse upwards from the lowest floor to build levels
        current_floor = lowest_floor
        level = 0
        # Use above_relations map (f_lower -> f_higher) to traverse upwards
        while current_floor is not None:
            if current_floor in self.floor_levels:
                 # Cycle detected or already visited - should not happen in linear structure
                 self.floor_levels = {} # Indicate failure
                 return
            self.floor_levels[current_floor] = level
            # Find the floor immediately above the current floor
            current_floor = above_relations.get(current_floor) # Use above_relations directly
            level += 1

        # Check if all floors mentioned in destinations are in floor_levels
        # If not, the floor structure might be incomplete.
        # This check is not strictly necessary for heuristic correctness (we return inf if level is None),
        # but can help identify problem definition issues.
        # for floor in self.goal_locations.values():
        #     if floor not in self.floor_levels:
        #          pass # Warning already handled implicitly by returning inf in __call__


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # If floor levels weren't initialized successfully, return a large value
        if not self.floor_levels:
             return float('inf') # Cannot compute heuristic

        # Parse relevant facts from the current state in a single pass
        lift_at_floor = None
        boarded_passengers = set()
        origin_locations = {} # passenger -> floor
        served_passengers = set() # passengers who are served

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            if predicate == "lift-at":
                lift_at_floor = parts[1]
            elif predicate == "boarded":
                boarded_passengers.add(parts[1])
            elif predicate == "origin":
                passenger, floor = parts[1], parts[2]
                origin_locations[passenger] = floor
            elif predicate == "served":
                 served_passengers.add(parts[1])


        # Ensure lift location is found (should always be true in valid states)
        if lift_at_floor is None:
             # This state might be invalid. Return infinity.
             return float('inf')

        total_cost = 0

        # Get lift level, handle case where lift is at an unknown floor (shouldn't happen in valid problems)
        current_lift_level = self.floor_levels.get(lift_at_floor)
        if current_lift_level is None:
             # Lift is at a floor not in our level map. Cannot compute distance.
             return float('inf')


        # Iterate through all passengers whose destination is known
        for passenger, destin_floor in self.goal_locations.items():
            # Check if the passenger is already served (goal reached for this passenger)
            if passenger in served_passengers:
                 continue # This passenger is done

            # Get destination level, handle case where destination floor is unknown
            destin_level = self.floor_levels.get(destin_floor)
            if destin_level is None:
                 # Destination floor not in our level map. Cannot compute distance.
                 return float('inf')


            # Handle passengers who are boarded
            if passenger in boarded_passengers:
                # Cost = move from current lift floor to destination + depart action
                move_cost = abs(current_lift_level - destin_level)
                total_cost += move_cost + 1 # +1 for depart action

            # Handle passengers waiting at their origin
            elif passenger in origin_locations:
                origin_floor = origin_locations[passenger]
                origin_level = self.floor_levels.get(origin_floor)

                # Handle case where origin floor is unknown
                if origin_level is None:
                     # Origin floor not in our level map. Cannot compute distance.
                     return float('inf')

                # Cost = move from current lift floor to origin + board action
                #      + move from origin to destination + depart action
                move_to_origin_cost = abs(current_lift_level - origin_level)
                move_to_destin_cost = abs(origin_level - destin_level)
                total_cost += move_to_origin_cost + 1 + move_to_destin_cost + 1 # +1 board, +1 depart

            # If passenger is neither served, boarded, nor at origin_locations,
            # they are not contributing to the heuristic in this state.
            # This assumes the state is well-formed and unserved passengers
            # are always in one of these two states (at origin or boarded).
            # If such a passenger exists, the heuristic might underestimate.
            # For a greedy search, ignoring them is a simple approach.


        return total_cost
