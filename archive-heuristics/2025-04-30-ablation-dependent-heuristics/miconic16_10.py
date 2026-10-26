import math

class miconicHeuristic:
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    Estimates the cost to reach the goal state (all passengers served) by summing
    the number of passengers still at their origin (representing required 'board' actions),
    the number of unserved passengers (representing required 'depart' actions),
    and the estimated minimum lift travel required to visit all necessary floors.
    The estimated travel is based on the vertical distance the lift must cover
    to reach the minimum and maximum floor levels where unserved passengers
    need pickup or dropoff, plus the distance from the current lift position
    to the closer of these two extreme floors.

    Assumptions:
    - Floor names are in the format 'f<number>'.
    - The `(above f_above f_below)` static facts define a linear ordering of floors.
    - The lowest floor is the one that never appears as the first argument (`f_above`)
      in any `(above)` fact.
    - The highest floor is the one that never appears as the second argument (`f_below`)
      in any `(above)` fact.
    - Floors between the lowest and highest are ordered numerically based on their names.
    - The numerical order of floors (f1, f2, ...) is consistent with the vertical order
      defined by `(above)` facts, potentially in reverse. The heuristic detects and
      corrects for reversed numerical order relative to vertical order.

    Heuristic Initialization:
    The constructor parses the static facts to determine the floor order and
    create mappings between floor names and their corresponding integer levels (1 to N).
    It also extracts the destination floor for each passenger from the static
    facts, as this information is static. It also identifies all passenger names.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the state is a goal state using `self.task.goal_reached(state)`. If yes, return 0.
    2. Find the current floor of the lift from the state fact `(lift-at ?f)`.
    3. Map the current lift floor to its level using the precomputed `floor_to_level` map.
       If the floor is not in the map, return infinity (indicating an unexpected state).
    4. Identify all unserved passengers by checking which passengers do not have the `(served ?p)` fact in the state.
    5. If there are no unserved passengers, return 0 (redundant due to step 1, but safe).
    6. Identify passengers who are still at their origin floor by checking for `(origin p f)` facts in the state for unserved passengers.
    7. Identify the set of "required" floors that the lift must visit:
       - For each unserved passenger `p` at their origin `f_origin(p)`, add `f_origin(p)` to required floors.
       - For each unserved passenger `p`, get their destination `f_destin(p)` from the precomputed `passenger_destinations` map and add it to required floors.
    8. Map the required floors to their levels using the `floor_to_level` map, filtering out any floors not found in the map.
    9. If there are no required levels (e.g., due to parsing issues or unexpected state), the travel cost is 0.
    10. If there are required levels, find the minimum required level ($L_{min\_req}$) and maximum required level ($L_{max\_req}$).
    11. Calculate the estimated minimum lift travel actions:
        $L_{current}$ is the level of the current lift floor.
        Estimated travel = $(L_{max\_req} - L_{min\_req}) + \min(abs(L_{current} - L_{min\_req}), abs(L_{current} - L_{max\_req}))$.
    12. The total heuristic value is the sum of the number of passengers at origin, the number of unserved passengers, and the estimated minimum lift travel actions.
        Heuristic = `len(passengers_at_origin)` + `len(unserved_passengers)` + `lift_travel_cost`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.static_facts = task.static

        # Parse floor levels from static facts
        self.floor_to_level, self.level_to_floor = self._parse_floor_levels(self.static_facts, self.task.initial_state)

        # Extract passenger destinations from static facts
        self.passenger_destinations = self._parse_passenger_destinations(self.static_facts)

        # Get all passenger names from static facts (e.g., destin facts)
        self.all_passengers = self._get_all_passengers(self.static_facts)


    def _parse_floor_levels(self, static_facts, initial_state):
        """
        Parses (above f1 f2) facts to determine floor order and levels.
        (above f1 f2) means f1 is above f2.
        Identifies the lowest floor (never appears as f1) and highest floor (never appears as f2).
        Assumes floors are named f<number> and are linearly ordered between lowest and highest.
        Assigns levels 1 to N based on this order.
        Includes fallback for single floor or complex/missing above facts.
        """
        floors = set()
        floors_above_arg = set() # Floors that appear as the first argument in (above)
        floors_below_arg = set() # Floors that appear as the second argument in (above)

        for fact in static_facts:
            if fact.startswith('(above '):
                parts = fact[1:-1].split()
                f_above, f_below = parts[1], parts[2]
                floors.add(f_above)
                floors.add(f_below)
                floors_above_arg.add(f_above)
                floors_below_arg.add(f_below)
            # Also collect floors mentioned in destin facts
            if fact.startswith('(destin '):
                 parts = fact[1:-1].split()
                 floors.add(parts[2])


        # Also collect floors mentioned in initial state (like lift-at, origin)
        for fact in initial_state:
             if fact.startswith('(lift-at '):
                 floors.add(fact[1:-1].split()[1])
             if fact.startswith('(origin '):
                 floors.add(fact[1:-1].split()[2])


        if not floors:
             # No floors found anywhere
             return {}, {}

        if len(floors) == 1:
             # Single floor problem
             floor_name = list(floors)[0]
             return {floor_name: 1}, {1: floor_name}

        # Identify lowest and highest floors based on (above) facts
        lowest_floors_candidates = list(floors_below_arg - floors_above_arg)
        highest_floors_candidates = list(floors_above_arg - floors_below_arg)

        if len(lowest_floors_candidates) != 1 or len(highest_floors_candidates) != 1:
             # Not a simple linear order or missing endpoints in (above) facts
             # Fallback to numerical sort if structure is not simple linear
             # print(f"Warning: Could not determine unique lowest/highest floor from (above) facts. Lowest candidates: {lowest_floors_candidates}, Highest candidates: {highest_floors_candidates}. Using numerical sort fallback.")
             def floor_sort_key(floor_name):
                 try:
                     return int(floor_name[1:])
                 except ValueError:
                     return float('inf') # Should not happen with f<number> format

             sorted_floors = sorted(list(floors), key=floor_sort_key)
             floor_to_level = {floor: i + 1 for i, floor in enumerate(sorted_floors)}
             level_to_floor = {i + 1: floor for i, floor in enumerate(sorted_floors)}
             return floor_to_level, level_to_floor


        lowest_floor = lowest_floors_candidates[0]
        highest_floor = highest_floors_candidates[0]

        # Assume floors are f<number> and the number determines the order between lowest/highest
        # Get all floor names and sort them numerically
        def floor_sort_key(floor_name):
            try:
                return int(floor_name[1:])
            except ValueError:
                return float('inf') # Should not happen with f<number> format

        sorted_floors_numerically = sorted(list(floors), key=floor_sort_key)

        # The order is lowest_floor, ..., highest_floor.
        # Find the numerical index of lowest and highest floors in the sorted list.
        try:
            lowest_idx = sorted_floors_numerically.index(lowest_floor)
            highest_idx = sorted_floors_numerically.index(highest_floor)
        except ValueError:
             # Should not happen if lowest/highest were found in the set of all floors
             print("Error: Lowest or highest floor not found in sorted floor list.")
             return {}, {}


        # Assign levels based on the order from lowest to highest
        floor_to_level = {}
        level_to_floor = {}
        level = 1
        if lowest_idx < highest_idx: # e.g., f1 is lowest, f20 is highest
             for i in range(lowest_idx, highest_idx + 1):
                 floor = sorted_floors_numerically[i]
                 floor_to_level[floor] = level
                 level_to_floor[level] = floor
                 level += 1
        else: # e.g., f20 is lowest, f1 is highest
             for i in range(lowest_idx, highest_idx - 1, -1):
                 floor = sorted_floors_numerically[i]
                 floor_to_level[floor] = level
                 level_to_floor[level] = floor
                 level += 1

        # Check consistency with a sample (above) fact and reverse levels if needed
        # This handles cases like f1 below f2, but numerical sort puts f1 first.
        sample_fact = next((fact for fact in static_facts if fact.startswith('(above ')), None)
        if sample_fact:
             parts = sample_fact[1:-1].split()
             f_above, f_below = parts[1], parts[2]
             if f_above in floor_to_level and f_below in floor_to_level:
                 # If (above f_above f_below) is true, level(f_above) should be > level(f_below)
                 if floor_to_level[f_above] <= floor_to_level[f_below]:
                     # Levels are currently assigned in reverse order relative to 'above'
                     # print(f"Info: Assigned levels ({floor_to_level[f_above]} for {f_above}, {floor_to_level[f_below]} for {f_below}) inconsistent with (above {f_above} {f_below}). Reversing levels.")
                     max_level = len(floor_to_level)
                     new_floor_to_level = {}
                     new_level_to_floor = {}
                     for floor, level in floor_to_level.items():
                         new_level = max_level - level + 1
                         new_floor_to_level[floor] = new_level
                         new_level_to_floor[new_level] = floor
                     floor_to_level = new_floor_to_level
                     level_to_floor = new_level_to_floor


        # Include any floors found in state/destin but not in the linear above chain?
        # If a floor exists but wasn't part of the main linear chain identified by lowest/highest,
        # it won't be in floor_to_level. The heuristic calculation handles this by filtering.
        # This assumes such floors are irrelevant or indicate an ill-formed problem for this heuristic.

        return floor_to_level, level_to_floor


    def _parse_passenger_destinations(self, static_facts):
        """
        Parses (destin p f) facts to map passengers to their destinations.
        """
        passenger_destinations = {}
        for fact in static_facts:
            if fact.startswith('(destin '):
                parts = fact[1:-1].split()
                p, f = parts[1], parts[2]
                passenger_destinations[p] = f
        return passenger_destinations

    def _get_all_passengers(self, static_facts):
         """
         Gets all passenger names from static facts (e.g., destin facts).
         """
         passengers = set()
         for fact in static_facts:
             if fact.startswith('(destin '):
                 parts = fact[1:-1].split()
                 passengers.add(parts[1])
         return list(passengers)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal.
        """
        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # 2. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_lift_floor = fact[1:-1].split()[1]
                break

        # 3. Map current lift floor to level
        if current_lift_floor is None or current_lift_floor not in self.floor_to_level:
             # Should not happen in a valid miconic state, or floor wasn't parsed
             # Return infinity as heuristic is unreliable/state might be invalid
             return float('inf')

        current_lift_level = self.floor_to_level[current_lift_floor]


        # 4. Identify unserved passengers and passengers at origin
        served_passengers = {fact[1:-1].split()[1] for fact in state if fact.startswith('(served ')}
        unserved_passengers = []
        passengers_at_origin = []
        required_floors = set()

        for p in self.all_passengers:
            if p not in served_passengers:
                unserved_passengers.append(p)

                # Check if passenger is at origin in the current state
                origin_fact = next((fact for fact in state if fact.startswith(f'(origin {p} ')), None)
                if origin_fact:
                    origin_floor = origin_fact[1:-1].split()[2]
                    passengers_at_origin.append(p)
                    required_floors.add(origin_floor) # Need to visit origin to pick up

                # Add destination floor for all unserved passengers
                if p in self.passenger_destinations:
                    required_floors.add(self.passenger_destinations[p])


        # 5. If no unserved passengers, return 0 (redundant check)
        if not unserved_passengers:
             return 0

        # 8. Map required floors to levels, filtering out unknown floors
        required_levels = {self.floor_to_level[f] for f in required_floors if f in self.floor_to_level}

        # 9. Calculate estimated lift travel actions
        lift_travel_cost = 0
        if required_levels: # Only calculate travel if there are floors to visit
            min_req_level = min(required_levels)
            max_req_level = max(required_levels)

            # Estimated travel: distance to cover the range [min_req, max_req] starting from current level
            # This is the span (max - min) plus the distance from current to the closer end of the span.
            lift_travel_cost = (max_req_level - min_req_level) + min(abs(current_lift_level - min_req_level), abs(current_lift_level - max_req_level))


        # 12. Total heuristic = Number of passengers at origin + Number of unserved passengers + Estimated minimum travel
        total_heuristic = len(passengers_at_origin) + len(unserved_passengers) + lift_travel_cost

        return total_heuristic
