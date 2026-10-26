import re

class miconicHeuristic:
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
        Estimates the cost to reach the goal state (all passengers served)
        by summing two components:
        1. An estimate of the minimum number of floor movements required
           to visit all floors where unserved passengers need to be picked up
           or dropped off.
        2. An estimate of the minimum number of board/depart actions required
           for all unserved passengers.

    Assumptions:
        - Floors are named 'f' followed by a number (e.g., f1, f2, f10).
        - The 'above' predicates define a linear ordering of floors,
          where `(above fi fj)` implies floor fi is physically above floor fj,
          and the numerical order of floors (f1, f2, ...) corresponds to this physical order
          (f1 is the lowest, f2 is above f1, etc.).
        - The problem is solvable (finite number of steps to reach the goal).
        - The state representation is a frozenset of strings like '(predicate arg1 arg2)'.
        - Static facts include all 'destin' and 'above' predicates relevant to the problem instance.
        - The initial state includes a '(lift-at ?f)' fact and '(origin ?p ?f)' facts for waiting passengers.
        - All floors mentioned in the problem can be ordered numerically based on their names.

    Heuristic Initialization:
        The constructor pre-processes static information:
        - Parses 'destin' facts from `task.static` to create a mapping from passenger to destination floor (`self.destinations`).
        - Collects all floor names mentioned in `task.static` ('above', 'destin') and `task.initial_state` ('lift-at', 'origin').
        - Determines the linear order of floors by sorting the collected floor names numerically based on the number part (e.g., f1, f2, f10 -> f1, f2, ..., f10).
        - Creates mappings between floor names and numerical indices (`self.floor_to_index`) and vice versa (`self.index_to_floor`).

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the input 'state' (a frozenset of strings) to identify:
           - The current floor of the lift (from the '(lift-at ?f)' fact).
           - Passengers waiting at origin floors (from '(origin ?p ?f)' facts).
           - Passengers currently boarded in the lift (from '(boarded ?p)' facts).
           - Passengers already served (from '(served ?p)' facts).
        2. Retrieve the destination floor for each passenger using the 'destinations' map
           created during initialization from static facts.
        3. Determine the set of all passengers known from 'destin' facts.
        4. Determine the set of unserved passengers by excluding served passengers
           from the set of all passengers.
        5. If the set of unserved passengers is empty, the state is a goal state,
           and the heuristic value is 0.
        6. If there are unserved passengers, separate them into two groups:
           - Waiting unserved: Passengers who are unserved and have an '(origin ?p ?f)' fact in the state.
           - Boarded unserved: Passengers who are unserved and have a '(boarded ?p)' fact in the state.
        7. Identify the set of 'required floors' that the lift must visit:
           - This set includes the origin floor for each waiting unserved passenger.
           - This set includes the destination floor for each boarded unserved passenger.
        8. Calculate the action cost estimate:
           - Each waiting unserved passenger requires at least one 'board' action and one 'depart' action (minimum 2 actions).
           - Each boarded unserved passenger requires at least one 'depart' action (minimum 1 action).
           - The estimated action cost is the sum of these minimum actions for all unserved passengers:
             `2 * (number of waiting unserved passengers) + 1 * (number of boarded unserved passengers)`.
        9. Calculate the movement cost estimate:
           - If `required_floors` is empty (which should only happen if `unserved_passengers` is also empty, handled in step 5), the movement cost is 0.
           - Otherwise, convert the current lift floor and all required floors to their numerical indices
             using the 'floor_to_index' map created during initialization.
           - Find the minimum and maximum floor indices among the required floors.
           - The estimated movement cost is the minimum distance required to travel from the
             current lift floor to either the minimum or maximum required floor index,
             plus the distance required to traverse the entire range between the minimum
             and maximum required floor indices. This is calculated as
             `min(abs(current_floor_idx - min_req_idx), abs(current_floor_idx - max_req_idx)) + (max_req_idx - min_req_idx)`.
             Each unit of distance corresponds to one 'up' or 'down' action.
             (Note: If the state is malformed, e.g., missing lift location or referring to unknown floors,
             the heuristic might return infinity, indicating an unreachable/invalid state).
        10. The total heuristic value for the state is the sum of the movement cost estimate
            and the action cost estimate.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        @param task: The planning task object (instance of the Task class).
        """
        self.destinations = {}
        self.floor_to_index = {}
        self.index_to_floor = {}

        # 1. Process static facts to get destinations and floor names
        all_floors_set = set()

        for fact_str in task.static:
            # Remove parentheses and split
            parts = fact_str[1:-1].split()
            if not parts: # Skip empty strings if any
                continue
            predicate = parts[0]

            if predicate == 'destin':
                # Expecting '(destin passenger floor)'
                if len(parts) == 3:
                    passenger = parts[1]
                    floor = parts[2]
                    self.destinations[passenger] = floor
                    all_floors_set.add(floor) # Collect floors from destinations too
            elif predicate == 'above':
                # Expecting '(above floor1 floor2)'
                 if len(parts) == 3:
                    floor1 = parts[1]
                    floor2 = parts[2]
                    all_floors_set.add(floor1)
                    all_floors_set.add(floor2)

        # Also collect floors from initial state lift-at and origin facts
        for fact_str in task.initial_state:
             parts = fact_str[1:-1].split()
             if not parts:
                 continue
             predicate = parts[0]
             if predicate == 'lift-at' and len(parts) == 2:
                 all_floors_set.add(parts[1])
             elif predicate == 'origin' and len(parts) == 3:
                 all_floors_set.add(parts[2])


        # 2. Determine floor order and create index maps
        # Assuming floor names are f1, f2, ..., fn and (above fi fj) implies i > j
        # We can sort floors numerically based on the number part of the name.
        floor_names = [f for f in all_floors_set if f.startswith('f')]

        # Sort floors numerically based on the number part
        try:
            # Use regex to extract the number part robustly
            all_floors_sorted = sorted(floor_names, key=lambda f: int(re.search(r'\d+', f).group()))
        except (ValueError, AttributeError):
            # Handle cases where floor names might not be strictly fN format
            # or regex fails. This heuristic relies on numerical floor order.
            # If sorting fails, the heuristic cannot be computed as designed.
            raise ValueError("Floor names are not in expected 'fN' format for numerical sorting.")


        if not all_floors_sorted:
             # This should not happen in a valid problem with floors
             raise ValueError("Could not identify any floors.")

        for i, floor in enumerate(all_floors_sorted):
            self.floor_to_index[floor] = i + 1 # Use 1-based indexing
            self.index_to_floor[i + 1] = floor

        # Basic check that we found at least one floor
        if not self.floor_to_index:
             raise ValueError("Could not determine floor order from static facts.")


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal.
        """
        # Parse state facts
        current_lift_floor = None
        waiting_passengers = [] # List of (passenger, floor)
        boarded_passengers = [] # List of passenger
        served_passengers = []  # List of passenger

        for fact_str in state:
            # Remove parentheses and split
            parts = fact_str[1:-1].split()
            if not parts: # Skip empty strings
                continue
            predicate = parts[0]

            if predicate == 'lift-at' and len(parts) == 2:
                current_lift_floor = parts[1]
            elif predicate == 'origin' and len(parts) == 3:
                waiting_passengers.append((parts[1], parts[2]))
            elif predicate == 'boarded' and len(parts) == 2:
                boarded_passengers.append(parts[1])
            elif predicate == 'served' and len(parts) == 2:
                served_passengers.append(parts[1])

        # Get all passengers defined in the problem (from destinations)
        all_passengers = set(self.destinations.keys())

        # Identify unserved passengers
        unserved_passengers = {p for p in all_passengers if p not in served_passengers}

        # If all passengers are served, it's a goal state
        if not unserved_passengers:
            return 0

        # Separate unserved passengers by status
        waiting_unserved = [(p, f) for p, f in waiting_passengers if p in unserved_passengers]
        boarded_unserved = [p for p in boarded_passengers if p in unserved_passengers]

        # Identify required floors (origin for waiting, destination for boarded)
        pickup_floors = {f for p, f in waiting_unserved}
        dropoff_floors = {self.destinations[p] for p in boarded_unserved}
        required_floors = pickup_floors | dropoff_floors

        # Calculate action cost estimate
        # Each waiting unserved passenger needs board (1) + depart (1) = 2 actions
        # Each boarded unserved passenger needs depart (1) = 1 action
        action_cost_estimate = 2 * len(waiting_unserved) + 1 * len(boarded_unserved)

        # Calculate movement cost estimate
        # If there are unserved passengers but no required floors, this is unexpected
        # in a valid problem. It would mean unserved passengers are neither waiting
        # nor boarded. In this case, the movement cost is 0.
        if not required_floors:
             movement_estimate = 0
        else:
            # Ensure current_lift_floor is found and is a valid floor
            if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
                 # This state is malformed if lift-at fact is missing or refers to unknown floor
                 # Returning infinity indicates an unreachable/invalid state from this point.
                 return float('inf')

            current_lift_floor_idx = self.floor_to_index[current_lift_floor]

            required_indices = {self.floor_to_index[f] for f in required_floors}
            min_required_idx = min(required_indices)
            max_required_idx = max(required_indices)

            # Estimate movement to reach the range of required floors and traverse it
            # Distance from current floor to the minimum required floor
            dist_to_min = abs(current_lift_floor_idx - min_required_idx)
            # Distance from current floor to the maximum required floor
            dist_to_max = abs(current_lift_floor_idx - max_required_idx)
            # Distance to traverse the entire range of required floors
            dist_range = max_required_idx - min_required_idx

            # The minimum movement to visit all floors in the range [min_required_idx, max_required_idx]
            # starting from current_lift_floor_idx is the distance to reach one end of the range
            # plus the distance to traverse the range.
            movement_estimate = min(dist_to_min, dist_to_max) + dist_range


        # Total heuristic
        h = movement_estimate + action_cost_estimate

        return h
