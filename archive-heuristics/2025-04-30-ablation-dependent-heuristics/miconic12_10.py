class MiconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the number of unserved passengers (representing the board/depart actions
        needed) and the estimated minimum number of lift movement actions
        required to visit all necessary floors (origins of waiting passengers
        and destinations of boarded passengers).

    Assumptions:
        - The PDDL instance is valid for the miconic domain.
        - Floor names are structured as 'f<number>' (e.g., 'f1', 'f10').
        - The 'above' predicates define a linear order of floors, consistent
          with the numerical order of the floor names (e.g., (above f2 f1)
          implies f2 is immediately above f1, and f1 < f2 numerically).
        - The problem is solvable.

    Heuristic Initialization:
        In the constructor, the heuristic pre-processes the static information
        from the task.
        1. It extracts all 'destin' facts to build a dictionary mapping each
           passenger to their destination floor.
        2. It extracts all floor names involved in 'above' predicates.
        3. It sorts the floor names numerically based on their suffix to
           establish the floor order from lowest to highest.
        4. It creates a dictionary mapping each floor name to its corresponding
           integer index (0 for the lowest floor, 1 for the next, and so on).

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Check if the state is a goal state using task.goal_reached(). If true,
           the heuristic is 0. (This check is typically done by the search algorithm,
           but the heuristic should return 0 for goal states).
        2. Identify the current floor of the lift by finding the fact
           '(lift-at ?f)' in the state.
        3. Identify all passengers that are not yet served. These are passengers
           'p' for whom '(served p)' is a goal fact but is not present in the state.
        4. Initialize counters for passengers waiting at their origin (`num_origin`)
           and passengers currently boarded (`num_boarded`).
        5. Initialize sets for required stop floors: `waiting_origins` and
           `boarded_destins`.
        6. Iterate through the unserved passengers:
           - If the fact '(origin p f)' is in the state for some floor 'f',
             increment `num_origin` and add 'f' to `waiting_origins`.
           - If the fact '(boarded p)' is in the state, increment `num_boarded`,
             look up the passenger's destination floor 'd' using the pre-computed
             destinations map, and add 'd' to `boarded_destins`.
        7. Combine the required stop floors: `all_required_stops = waiting_origins | boarded_destins`.
        8. If `all_required_stops` is empty (meaning all unserved passengers are
           neither waiting at an origin nor boarded, which implies an unsolvable
           state or an inconsistency, but assuming solvable problems, this case
           should ideally not be reached for unserved passengers unless h=0),
           return 0 (though the goal check handles the h=0 case).
        9. Get the integer indices for all floors in `all_required_stops` using
           the pre-computed floor index map. Find the minimum (`min_req_idx`)
           and maximum (`max_req_idx`) indices among these required stops.
        10. Get the index of the current lift floor (`current_lift_idx`).
        11. Calculate the minimum number of floor changes (`min_floor_changes`)
            required for the lift to travel from its current floor to visit all
            floors in `all_required_stops`. This is estimated as the distance
            from the current floor to the closest end of the required floor range
            (`[min_req_idx, max_req_idx]`) plus the length of that range.
            - If `current_lift_idx` is below `min_req_idx`, moves = `max_req_idx - current_lift_idx`.
            - If `current_lift_idx` is above `max_req_idx`, moves = `current_lift_idx - min_req_idx`.
            - If `current_lift_idx` is within the range (`min_req_idx <= current_lift_idx <= max_req_idx`),
              moves = `(max_req_idx - min_req_idx) + min(current_lift_idx - min_req_idx, max_req_idx - current_lift_idx)`.

        12. The final heuristic value is the sum: `num_origin + num_boarded + min_floor_changes`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing static task information.

        @param task: The planning task object.
        """
        self.task = task
        self.destinations = {}
        self.floor_indices = {}
        self._parse_static_info()

    def _parse_static_info(self):
        """
        Parses static facts to build destination map and floor index map.
        Assumes floor names are f<number> and sorted numerically correspond to floor order.
        """
        all_floor_names_from_static = set()
        for fact_str in self.task.static:
             parts = fact_str.strip("()").split()
             if len(parts) >= 3: # Ensure enough parts before accessing indices
                 predicate = parts[0]
                 if predicate == 'destin':
                     # Fact is like '(destin p1 f2)'
                     if len(parts) == 3:
                         passenger = parts[1]
                         floor = parts[2]
                         self.destinations[passenger] = floor
                 elif predicate == 'above':
                     # Fact is like '(above f1 f2)' meaning f1 is immediately above f2
                     if len(parts) == 3:
                         floor_above = parts[1]
                         floor_below = parts[2]
                         all_floor_names_from_static.add(floor_above)
                         all_floor_names_from_static.add(floor_below)

        if all_floor_names_from_static:
            # Sort floor names numerically based on the number suffix
            # Assumes floor names are like 'f1', 'f10', 'f2'
            try:
                sorted_floor_names = sorted(list(all_floor_names_from_static), key=lambda f: int(f[1:]))
                for i, floor in enumerate(sorted_floor_names):
                    self.floor_indices[floor] = i
            except ValueError:
                # Fallback for unexpected floor names - might lead to incorrect heuristic
                print(f"Warning: Floor names not in expected 'f<number>' format. Using alphabetical sort. Floors: {all_floor_names_from_static}")
                sorted_floor_names = sorted(list(all_floor_names_from_static)) # Fallback sort
                for i, floor in enumerate(sorted_floor_names):
                    self.floor_indices[floor] = i


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal.
        """
        # The search algorithm typically checks for goal state before calling heuristic.
        # However, the heuristic must return 0 for goal states by definition.
        # We can check task.goal_reached(state) here, but it might be redundant
        # depending on the planner's implementation. Let's rely on the logic
        # that if no unserved passengers exist, the heuristic will be 0.

        # 2. Identify current lift floor
        current_lift_floor = None
        # Convert state frozenset to set for faster lookups if state is large
        state_facts_set = set(state)

        for fact_str in state_facts_set:
            if fact_str.startswith('(lift-at '):
                parts = fact_str.strip("()").split()
                if len(parts) == 2:
                    current_lift_floor = parts[1]
                    break

        if current_lift_floor is None:
             # Lift location unknown, indicates invalid state.
             return float('inf')

        current_lift_idx = self.floor_indices.get(current_lift_floor)
        if current_lift_idx is None:
             # Current lift floor name found in state but not in floor indices map.
             return float('inf')


        # 3. Identify unserved passengers
        all_passengers = set()
        # Assuming goal is always (and (served p1) (served p2) ...)
        # Extract passengers from goal facts
        for goal_fact in self.task.goals:
             if goal_fact.startswith('(served '):
                 parts = goal_fact.strip("()").split()
                 if len(parts) == 2:
                     all_passengers.add(parts[1])

        unserved_passengers = {p for p in all_passengers if '(served ' + p + ')' not in state_facts_set}

        # If no unserved passengers, goal is reached.
        if not unserved_passengers:
             return 0

        # 4, 5, 6. Categorize unserved passengers and find required stops
        num_origin = 0
        num_boarded = 0
        waiting_origins = set()
        boarded_destins = set()

        for p in unserved_passengers:
            is_origin = False
            is_boarded = False

            # Check if waiting at origin
            origin_floor = None
            # Efficiently check for '(origin p f)' facts
            origin_fact_prefix = '(origin ' + p + ' '
            for fact_str in state_facts_set:
                if fact_str.startswith(origin_fact_prefix):
                    parts = fact_str.strip("()").split()
                    if len(parts) == 3:
                        origin_floor = parts[2]
                        is_origin = True
                        break

            if is_origin:
                num_origin += 1
                waiting_origins.add(origin_floor)
            elif '(boarded ' + p + ')' in state_facts_set:
                is_boarded = True
                num_boarded += 1
                # Find destination from pre-computed map
                destin_floor = self.destinations.get(p)
                if destin_floor:
                    boarded_destins.add(destin_floor)
                else:
                     # Passenger boarded but destination unknown? Invalid state.
                     return float('inf') # Unsolvable

            # else: passenger is unserved but neither at origin nor boarded? Invalid state.
            # Assuming valid states, this else should not be reached for unserved passengers.
            # If it is, it's likely unsolvable.
            # Note: A passenger could be at origin AND boarded in an invalid state,
            # but the domain actions prevent this. Assuming valid states.


        # 7. Combine required stops
        all_required_stops = waiting_origins | boarded_destins

        # 8. If no required stops, but unserved passengers exist, something is wrong.
        if not all_required_stops:
             # This case should ideally not be reached in a solvable problem
             # with unserved passengers.
             return float('inf') # Unsolvable or inconsistent state


        # 9. Get indices of required stops
        required_indices = set()
        for f in all_required_stops:
            idx = self.floor_indices.get(f)
            if idx is not None:
                required_indices.add(idx)
            else:
                 # Required floor name found but not in floor indices map.
                 return float('inf') # Unsolvable

        if not required_indices:
             # This implies all floors in all_required_stops were not found in floor_indices.
             return float('inf') # Unsolvable

        min_req_idx = min(required_indices)
        max_req_idx = max(required_indices)

        # 10. Current lift index is already found

        # 11. Calculate minimum floor changes
        moves = 0
        if current_lift_idx < min_req_idx: # Strictly less than
            moves = max_req_idx - current_lift_idx
        elif current_lift_idx > max_req_idx: # Strictly greater than
            moves = current_lift_idx - min_req_idx
        else: # min_req_idx <= current_lift_idx <= max_req_idx
            # Distance to cover the range + distance from current to closest end of range
            moves = (max_req_idx - min_req_idx) + min(current_lift_idx - min_req_idx, max_req_idx - current_lift_idx)


        # 12. Calculate total heuristic value
        heuristic_value = num_origin + num_boarded + moves

        return heuristic_value

# Note: The heuristic is implemented as a class that is initialized with the task
# and then called with states. This is a standard pattern for heuristics that
# require pre-processing static information. The planner is expected to
# instantiate the class once with the task object and then call the instance
# whenever a heuristic value is needed for a state. For example:
#
# heuristic_object = MiconicHeuristic(task)
# h_value = heuristic_object(current_state)
