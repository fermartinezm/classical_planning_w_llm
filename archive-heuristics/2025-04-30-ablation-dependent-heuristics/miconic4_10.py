import math

class miconicHeuristic:
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    Estimates the cost to reach the goal state (all passengers served)
    by summing the number of board actions needed, the number of depart
    actions needed, and an estimate of the minimum number of lift movement
    actions required to visit all necessary floors.

    Assumptions:
    - The PDDL domain is miconic as provided.
    - Floors are totally ordered by the 'above' predicate. If no 'above'
      facts are present, floors found in other static/initial facts are
      ordered alphabetically as a fallback.
    - Action costs are uniform (each action costs 1).
    - The state representation is a frozenset of strings as described.
    - Static facts include 'destin' for all passengers and 'above' defining
      the floor order (if applicable).

    Heuristic Initialization:
    The constructor processes the static facts from the task:
    1. Identifies all floors and determines their order based on the 'above'
       predicate. It counts how many floors each floor is directly stated
       to be 'above'. Sorting floors by this count in ascending order gives
       the floor sequence from lowest to highest. A mapping from floor name
       (string) to integer index (0-based) is created. If no 'above' facts
       are present, floors found in 'destin', 'lift-at', and 'origin' facts
       are collected and ordered alphabetically as a fallback.
    2. Extracts the destination floor for each passenger from 'destin' facts
       and stores this in a dictionary (passenger name to floor name).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the current floor of the lift by finding the fact '(lift-at ?f)'.
    2. Identify passengers who are waiting at their origin floor by finding
       facts '(origin ?p ?f)'. Store them as a set of (passenger, floor) tuples.
    3. Identify passengers who are currently boarded in the lift by finding
       facts '(boarded ?p)'. Store them as a set of passenger names.
    4. Identify passengers who are already served by finding facts '(served ?p)'.
       (Note: Served passengers do not contribute to the heuristic, but identifying
       them helps understand the state).
    5. Calculate the number of board actions needed: This is equal to the number
       of waiting passengers.
    6. Calculate the number of depart actions needed: This is equal to the number
       of boarded passengers.
    7. Determine the set of 'required floors': These are the origin floors of
       waiting passengers and the destination floors of boarded passengers.
    8. Estimate the number of movement actions (up/down) needed:
       - If there are no required floors, the movement cost is 0.
       - If there are required floors, convert their names to integer indices
         using the precomputed floor-to-integer map. Find the minimum and
         maximum integer indices among the required floors. Let the current
         lift floor's index be `f_current_int`, the minimum required index be
         `min_req_int`, and the maximum required index be `max_req_int`.
       - The estimated movement cost is the minimum number of moves required
         to travel from the current floor to visit all required floors. A lower
         bound estimate for this is the minimum of the distance from the current
         floor to the lowest required floor and the distance from the current
         floor to the highest required floor, plus the total vertical distance
         between the lowest and highest required floors. This is calculated as:
         `min(abs(f_current_int - min_req_int), abs(f_current_int - max_req_int)) + (max_req_int - min_req_int)`.
         If the current lift floor is not found in the floor map (e.g., due to
         an unexpected state fact), the movement cost is estimated as just the
         span distance (`max_req_int - min_req_int`).
    9. The total heuristic value is the sum of the number of board actions needed,
       the number of depart actions needed, and the estimated movement cost.
       This value is 0 if and only if the state is a goal state (all passengers
       served, meaning no waiting or boarded passengers, and thus no required
       floors).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        @param task: The planning task object.
        """
        self.task = task
        self.destin_map = {}
        self.floor_to_int = {}
        self._process_static_facts()

    def _process_static_facts(self):
        """
        Processes static facts to build floor ordering and destination map.
        """
        all_floors_from_above = set()
        floor_above_counts = {} # Count how many floors each floor is stated to be *above*

        # First pass: process 'above' facts to find floors and ordering info
        for fact_str in self.task.static:
            parts = fact_str.strip("()").split()
            if not parts: continue # Skip empty strings

            if parts[0] == 'above':
                # Fact is like '(above f1 f2)' meaning f1 is above f2
                if len(parts) > 2:
                    floor1 = parts[1]
                    floor2 = parts[2]
                    all_floors_from_above.add(floor1)
                    all_floors_from_above.add(floor2)
                    # Initialize count if floor not seen before
                    floor_above_counts.setdefault(floor1, 0)
                    floor_above_counts.setdefault(floor2, 0)
                    # Increment count for the floor that is above another
                    floor_above_counts[floor1] += 1

        # If 'above' facts exist, use them for ordering
        if all_floors_from_above:
            # Sort floors based on the count of floors they are above (ascending).
            # The floor above the fewest others is the lowest.
            sorted_floors = sorted(list(all_floors_from_above), key=lambda f: floor_above_counts.get(f, 0))
            self.floor_to_int = {floor: i for i, floor in enumerate(sorted_floors)}
        else:
            # If no 'above' facts, collect all floors mentioned in static/initial state
            # and order them alphabetically as a fallback.
            all_floors_fallback = set()
            for fact_str in self.task.static:
                 parts = fact_str.strip("()").split()
                 if not parts: continue
                 if parts[0] == 'destin':
                     if len(parts) > 2:
                         all_floors_fallback.add(parts[2])

            for fact_str in self.task.initial_state:
                 parts = fact_str.strip("()").split()
                 if not parts: continue
                 if parts[0] == 'lift-at':
                     if len(parts) > 1:
                         all_floors_fallback.add(parts[1])
                 elif parts[0] == 'origin':
                     if len(parts) > 2:
                         all_floors_fallback.add(parts[2])

            if all_floors_fallback:
                 sorted_floors_fallback = sorted(list(all_floors_fallback))
                 self.floor_to_int = {floor: i for i, floor in enumerate(sorted_floors_fallback)}
            # else: No floors found at all, floor_to_int remains empty.

        # Process 'destin' facts
        for fact_str in self.task.static:
            parts = fact_str.strip("()").split()
            if not parts: continue
            if parts[0] == 'destin':
                # Fact is like '(destin p1 f2)'
                if len(parts) > 2:
                    passenger = parts[1]
                    dest_floor = parts[2]
                    self.destin_map[passenger] = dest_floor


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        current_lift_floor = None
        waiting_passengers = set() # set of (passenger, origin_floor)
        boarded_passengers = set() # set of passenger names
        # served_passengers = set() # Not needed for heuristic calculation

        # Parse current state
        for fact_str in state:
            parts = fact_str.strip("()").split()
            if not parts: continue # Skip empty strings resulting from malformed facts

            if parts[0] == 'lift-at':
                if len(parts) > 1:
                    current_lift_floor = parts[1]
            elif parts[0] == 'origin':
                if len(parts) > 2:
                    passenger = parts[1]
                    origin_floor = parts[2]
                    waiting_passengers.add((passenger, origin_floor))
            elif parts[0] == 'boarded':
                 if len(parts) > 1:
                    passenger = parts[1]
                    boarded_passengers.add(passenger)
            # elif parts[0] == 'served':
            #    if len(parts) > 1:
            #        passenger = parts[1]
            #        served_passengers.add(passenger)

        # Heuristic components
        h_board = len(waiting_passengers)
        h_depart = len(boarded_passengers)

        # Determine required floors for movement
        required_floors = set()
        for passenger, origin_floor in waiting_passengers:
            required_floors.add(origin_floor)
        for passenger in boarded_passengers:
            # Get destination from static info
            if passenger in self.destin_map:
                 required_floors.add(self.destin_map[passenger])
            # else: A boarded passenger should have a destination in static facts.
            # If not, this indicates an issue with the problem definition or state.
            # We ignore passengers without a known destination for the heuristic.


        h_movement = 0
        # Only calculate movement if there are floors to visit and we know the lift's location
        if required_floors and current_lift_floor in self.floor_to_int:
            # Convert required floors to integer indices, keeping only known floors
            required_indices = sorted([self.floor_to_int[f] for f in required_floors if f in self.floor_to_int])

            if required_indices: # Ensure there are valid floors after mapping
                min_req_int = required_indices[0]
                max_req_int = required_indices[-1]

                # Get current lift floor index
                current_lift_int = self.floor_to_int[current_lift_floor]

                # Estimate movement cost: min moves to get into the required range
                # and then traverse the range.
                dist_to_min = abs(current_lift_int - min_req_int)
                dist_to_max = abs(current_lift_int - max_req_int)
                span_dist = max_req_int - min_req_int

                # Minimum moves to visit all required floors starting from current
                # This is the min of going to the lowest required then sweeping up,
                # or going to the highest required then sweeping down.
                h_movement = min(dist_to_min + span_dist, dist_to_max + span_dist)
            # else: required_floors had names not in floor_to_int (shouldn't happen if floor_to_int includes all relevant floors)
            # h_movement remains 0
        elif required_floors and current_lift_floor not in self.floor_to_int:
             # This case should ideally not happen in a well-formed problem/state,
             # but as a fallback, if we know floors need visiting but don't know
             # the current floor's index, we can estimate movement as just the span.
             # This requires getting min/max required indices even without current.
             required_indices = sorted([self.floor_to_int[f] for f in required_floors if f in self.floor_to_int])
             if required_indices:
                 min_req_int = required_indices[0]
                 max_req_int = required_indices[-1]
                 h_movement = max_req_int - min_req_int


        # Total heuristic is sum of components
        heuristic_value = h_board + h_depart + h_movement

        # The goal_reached check at the start ensures h=0 for goal states.
        # If not a goal state, either h_board > 0 or h_depart > 0 (or both).
        # If h_board + h_depart > 0, then required_floors is not empty (unless
        # unserved passengers have no origin/destination facts, which is invalid PDDL).
        # If required_floors is not empty, h_movement will be >= 0 (assuming valid floor data).
        # Thus, heuristic_value will be > 0 for non-goal states.

        return heuristic_value
