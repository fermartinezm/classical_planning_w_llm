class miconicHeuristic:
    """
    Summary:
    This heuristic estimates the number of actions required to reach the goal
    in the miconic domain. The goal is to have all passengers served.
    The heuristic is the sum of three components:
    1. The number of passengers waiting at their origin floors (each requires a board action).
    2. The number of passengers currently boarded in the lift who need to depart
       at their destination floors (each requires a depart action).
    3. An estimate of the lift movement actions required to visit all necessary
       floors (origin floors for waiting passengers and destination floors for
       boarded passengers).

    Assumptions:
    - The PDDL instance is solvable.
    - Floor names follow a consistent naming convention (e.g., f1, f2, f3, ...)
      such that alphabetical/numerical sorting corresponds to the physical order
      defined by the `above` predicate. The `above` facts in the static information
      confirm this total order.
    - The lift can move freely between adjacent floors.

    Heuristic Initialization:
    The constructor pre-processes the static information:
    - It identifies all floor objects by examining `above`, `destin`, `lift-at`,
      and `origin` facts in the static information and initial state. It sorts
      these floors numerically based on their names (assuming f<number> format)
      to create a mapping from floor name to a numerical index (0-based). This
      allows calculating distances between floors.
    - It extracts the destination floor for each passenger from the `destin` facts
      in the static information.
    - It counts the total number of passengers based on the number of destination facts.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Count the number of passengers currently marked as `served`. If this count
       equals the total number of passengers in the problem, the state is a goal
       state, and the heuristic is 0.
    2. If it's not a goal state, parse the state facts to identify:
       - The current floor of the lift (`lift-at` predicate).
       - Passengers waiting at their origin floors (`origin` predicate) and their specific origin floors.
       - Passengers currently boarded in the lift (`boarded` predicate).
       - Passengers already served (`served` predicate).
    3. Calculate the number of passengers who are waiting at their origin floors
       and are not yet served (`num_waiting`). This contributes `num_waiting` to the heuristic.
    4. Calculate the number of passengers who are boarded but not yet served
       (`num_boarded_not_served`). This contributes `num_boarded_not_served` to the heuristic.
    5. Determine the set of "required" floors the lift must visit:
       - Add the origin floor for each passenger identified in step 3.
       - Add the destination floor for each passenger identified in step 4
         (using the pre-processed destination information from initialization).
    6. If the set of required floors is empty, the estimated travel is 0.
    7. If the set of required floors is not empty:
       - Find the minimum and maximum floor indices among the required floors
         using the floor-to-index mapping.
       - Calculate the estimated lift travel cost. This is the minimum number
         of move actions (`up` or `down`) needed to travel from the current
         lift floor to visit all required floors. A lower bound estimate is
         the distance from the current floor to the nearest end of the required
         floor range (min or max required floor), plus the distance to traverse
         the entire required range (max_idx - min_idx).
         Estimated travel = min(abs(current_idx - min_idx), abs(current_idx - max_idx)) + (max_idx - min_idx).
    8. The total heuristic value is the sum: `num_waiting` + `num_boarded_not_served` + Estimated travel.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        @param task: The planning task object (instance of the Task class).
        """
        self.task = task
        self.floor_to_index = {}
        self.index_to_floor = {}
        self.passenger_destin = {}

        # Extract floor names from static and initial state facts
        floor_names = set()
        for fact in task.static | task.initial_state:
            parts = fact.strip("()").split()
            predicate = parts[0]
            if predicate in ('above', 'lift-at', 'origin', 'destin'):
                 # Floors are typically the second or third argument
                 if len(parts) > 1:
                     # Check if argument looks like a floor (starts with 'f')
                     if parts[1].startswith('f'):
                         floor_names.add(parts[1])
                 if len(parts) > 2:
                     if parts[2].startswith('f'):
                         floor_names.add(parts[2])


        # Sort floors based on numerical part (assuming f1, f2, ...)
        # This relies on the naming convention and the above predicates confirming the order.
        # A more robust way would be to build a graph from 'above' and find the topological sort,
        # but simple numerical sorting is sufficient for typical miconic benchmarks.
        try:
            sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
        except ValueError:
             # Fallback if floor names are not in f<number> format, sort alphabetically
             sorted_floor_names = sorted(list(floor_names))


        for i, floor_name in enumerate(sorted_floor_names):
            self.floor_to_index[floor_name] = i
            self.index_to_floor[i] = floor_name

        # Extract passenger destinations from static facts
        for fact in task.static:
            if fact.startswith('(destin '):
                parts = fact.strip("()").split()
                passenger = parts[1]
                destin_floor = parts[2]
                self.passenger_destin[passenger] = destin_floor

        # Store total number of passengers for goal check
        self.total_passengers = len(self.passenger_destin)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal.
        """
        served_count = 0
        current_lift_floor = None
        passengers_waiting = set() # Passengers at origin
        passengers_boarded_raw = set() # All boarded passengers
        passengers_served = set()
        origin_floors_in_state = {} # Map passenger -> origin floor from state

        # Parse state facts in a single pass
        for fact in state:
            parts = fact.strip("()").split()
            predicate = parts[0]

            if predicate == 'served':
                passenger = parts[1]
                passengers_served.add(passenger)
                served_count += 1
            elif predicate == 'lift-at':
                current_lift_floor = parts[1]
            elif predicate == 'origin':
                passenger = parts[1]
                origin_floor = parts[2]
                passengers_waiting.add(passenger)
                origin_floors_in_state[passenger] = origin_floor
            elif predicate == 'boarded':
                passenger = parts[1]
                passengers_boarded_raw.add(passenger)

        # Check if goal is reached (all passengers served)
        if served_count == self.total_passengers:
             return 0 # Goal state

        # Filter passengers: only count those not yet served
        passengers_waiting_not_served = passengers_waiting - passengers_served
        passengers_boarded_not_served = passengers_boarded_raw - passengers_served

        # Component 1: Number of board actions needed
        num_waiting = len(passengers_waiting_not_served)

        # Component 2: Number of depart actions needed
        num_boarded_not_served = len(passengers_boarded_not_served)

        # Component 3: Estimated lift travel
        required_floors = set()

        # Add origin floors for waiting passengers
        for p in passengers_waiting_not_served:
             # Origin floor is in the state fact
             origin_floor = origin_floors_in_state.get(p)
             if origin_floor: # Should always be found if p is in passengers_waiting_not_served
                 required_floors.add(origin_floor)

        # Add destination floors for boarded passengers
        for p in passengers_boarded_not_served:
             # Destination is in static info
             destin_floor = self.passenger_destin.get(p)
             if destin_floor: # Should always be found for a valid passenger
                 required_floors.add(destin_floor)

        estimated_travel = 0
        if required_floors:
            # Ensure current_lift_floor was found (should be in any valid state)
            if current_lift_floor is None:
                 # This indicates an invalid state representation, return infinity or a large value
                 # For solvable problems, lift-at should always be present.
                 # Returning a large value helps prune this path if encountered.
                 return float('inf') # Or a large integer

            current_idx = self.floor_to_index.get(current_lift_floor)
            if current_idx is None:
                 # This indicates an invalid state representation (lift at unknown floor)
                 return float('inf') # Or a large integer

            required_indices = sorted([self.floor_to_index[f] for f in required_floors if f in self.floor_to_index])

            if required_indices: # Check again in case required_floors contained unknown floors
                min_idx = required_indices[0]
                max_idx = required_indices[-1]

                # Estimate travel: distance to nearest end + span
                dist_to_min = abs(current_idx - min_idx)
                dist_to_max = abs(current_idx - max_idx)
                span = max_idx - min_idx

                estimated_travel = min(dist_to_min, dist_to_max) + span
            else:
                 # Required floors were identified but none were valid floor names?
                 # Or maybe required_floors was not empty but contained only invalid floors?
                 # This case should ideally not happen in valid problems.
                 # If it does, and num_waiting/num_boarded_not_served > 0,
                 # the heuristic will still be > 0. If both are 0, it should be goal.
                 # Let's assume this branch implies no valid required floors were found.
                 # If num_waiting and num_boarded_not_served are > 0, this might be an underestimate.
                 # If num_waiting and num_boarded_not_served are 0, it should be goal (handled above).
                 # For safety, if required_floors was non-empty but resulted in empty required_indices,
                 # it might indicate an issue, but let's assume 0 travel in this specific sub-case
                 # as no valid floors were identified to travel to.
                 estimated_travel = 0 # This case should be rare/impossible in valid problems


        # Total heuristic
        h_value = num_waiting + num_boarded_not_served + estimated_travel

        # Ensure heuristic is 0 only at goal (already handled by the initial check)
        # and finite for solvable states (components are finite).

        return h_value

