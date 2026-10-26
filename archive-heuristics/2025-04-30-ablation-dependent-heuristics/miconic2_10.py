class miconicHeuristic:
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    Estimates the cost to reach the goal (all passengers served) by summing
    the number of board/depart actions needed for unserved passengers and
    an estimate of the travel cost for the lift to visit necessary floors.
    The travel cost is estimated based on the range of floor levels the lift
    must traverse to reach all origin floors of waiting passengers and all
    destination floors of boarded passengers, plus the distance from the
    current floor to the nearest extreme required floor.

    Assumptions:
    - The 'above' predicates define a total order on floors.
    - Floor names like 'f1', 'f2', etc., are identifiers, and their physical
      level is determined solely by the 'above' facts.
    - The heuristic assumes the lift can pick up and drop off multiple
      passengers on a single trip covering the required floors.
    - Input states and static information are valid according to the PDDL domain.

    Heuristic Initialization:
    - Parses static facts from the task.
    - Extracts passenger destinations from '(destin ?p ?f)' facts into a dictionary `self.destinations`.
    - Builds a mapping from floor names to integer levels `self.floor_levels` based on '(above ?f1 ?f2)' facts.
      The level of a floor `f` is determined by counting how many other floors
      `f_other` exist such that the fact '(above f f_other)' is present in the
      static information. This count represents the number of floors physically
      below `f`, providing a consistent level mapping where the lowest floor
      has level 0.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state `state` is a goal state by comparing it against `self.task.goals`. If `self.task.goals` is a subset of `state`, return 0.
    2. Identify the current floor of the lift by finding the fact matching '(lift-at ?f)' in `state`. Store the floor name as `current_lift_floor`.
    3. Identify all passengers who are not yet served. This is done by taking the set of all passengers (from `self.destinations`) and removing those for whom a '(served ?p)' fact exists in `state`.
    4. Separate the unserved passengers into two groups based on the current state:
       - `unserved_waiting`: Passengers `p` who are unserved and for whom an '(origin p f)' fact exists in `state`. Store their origin floors.
       - `unserved_boarded`: Passengers `p` who are unserved and for whom a '(boarded p)' fact exists in `state`.
    5. Calculate the base cost: This is the minimum number of board and depart actions required for the unserved passengers. Each unserved waiting passenger needs one 'board' action. Each unserved boarded passenger needs one 'depart' action. Base cost = `len(unserved_waiting) + len(unserved_boarded)`.
    6. Identify the set of floors the lift *must* visit to serve the remaining passengers. This set `required_stops` includes:
       - The origin floor for each passenger in `unserved_waiting`.
       - The destination floor for each passenger in `unserved_boarded` (retrieved from `self.destinations`).
    7. If the set `required_stops` is empty, no further travel is needed for unserved passengers, so the travel cost is 0.
    8. If `required_stops` is not empty, calculate the travel cost estimate:
       - Get the integer level for the `current_lift_floor` and all floors in `required_stops` using the `self.floor_levels` map. Collect these into a set `all_relevant_levels`.
       - Find the minimum level `min_L` and maximum level `max_L` within `all_relevant_levels`.
       - The travel cost is estimated as the total vertical range covered (`max_L - min_L`) plus the minimum distance from the `current_level` to either the minimum or maximum level (`min(abs(current_level - min_L), abs(current_level - max_L))`). This estimates the travel needed to reach one extreme of the required range and then traverse the full range.
    9. The total heuristic value is the sum of the `base_cost` and the calculated `travel_cost`.
    """

    def __init__(self, task):
        self.task = task
        self.destinations = {}
        self.floor_levels = {}
        self._process_static_info()

    def _process_static_info(self):
        # Extract destinations
        for fact_str in self.task.static:
            if fact_str.startswith('(destin '):
                parts = self._parse_fact(fact_str)
                passenger = parts[1]
                floor = parts[2]
                self.destinations[passenger] = floor

        # Build floor levels based on 'above' facts
        above_facts = [self._parse_fact(f) for f in self.task.static if f.startswith('(above ')]
        all_floors = set()
        for fact in above_facts:
            all_floors.add(fact[1]) # floor1
            all_floors.add(fact[2]) # floor2

        # Determine levels: level of floor f is the count of floors f_other
        # such that (above f f_other) is true. This counts floors below f.
        # This assumes a total order defined by 'above'.
        for f in all_floors:
            floors_below_f_count = 0
            for fact in above_facts:
                # fact is (above f_i f_j)
                if fact[1] == f: # f is f_i, so f is above f_j
                     floors_below_f_count += 1
            self.floor_levels[f] = floors_below_f_count

        # Handle edge case: if no above facts but floors exist (e.g., single floor problem)
        if not self.floor_levels and all_floors:
             # Assign level 0 to all floors if no 'above' relations are given
             for f in all_floors:
                 self.floor_levels[f] = 0


    def _parse_fact(self, fact_str):
        # Parses a fact string like '(predicate arg1 arg2)' into a tuple ('predicate', 'arg1', 'arg2')
        # Handles potential extra spaces and ensures correct splitting
        parts = fact_str.strip().strip('()').split()
        return tuple(parts)

    def _extract_objects_from_fact(self, fact_str):
        # Helper to extract objects from a fact string like '(predicate obj1 obj2)'
        parts = self._parse_fact(fact_str)
        return parts[1:] # Return list of objects


    def __call__(self, state):
        # Check for goal state
        if self.task.goals <= state:
            return 0

        # Extract current lift floor
        current_lift_floor = None
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                current_lift_floor = self._extract_objects_from_fact(fact_str)[0]
                break # Assuming only one lift-at fact

        # If for some reason lift-at is not in state (invalid state?), return infinity
        # Based on problem description, assuming valid states where lift-at exists.
        # If current_lift_floor is None, accessing self.floor_levels below would fail.
        # Let's assume it's always present in valid states.

        # Identify unserved passengers
        all_passengers = set(self.destinations.keys())
        served_passengers = set()
        waiting_passengers = {} # {passenger: origin_floor}
        boarded_passengers = set()

        for fact_str in state:
            if fact_str.startswith('(served '):
                p = self._extract_objects_from_fact(fact_str)[0]
                served_passengers.add(p)
            elif fact_str.startswith('(origin '):
                p, f = self._extract_objects_from_fact(fact_str)
                waiting_passengers[p] = f
            elif fact_str.startswith('(boarded '):
                p = self._extract_objects_from_fact(fact_str)[0]
                boarded_passengers.add(p)

        unserved_passengers = all_passengers - served_passengers
        unserved_waiting = {p for p in unserved_passengers if p in waiting_passengers}
        unserved_boarded = {p for p in unserved_passengers if p in boarded_passengers}

        # Calculate base cost (board/depart actions)
        base_cost = len(unserved_waiting) + len(unserved_boarded)

        # Identify required stops (floors to visit)
        required_stops = set()
        for p in unserved_waiting:
            # Origin floor must exist for a waiting passenger
            required_stops.add(waiting_passengers[p])
        for p in unserved_boarded:
            # Destination must exist for a boarded passenger
            # Assuming valid states and static info
            required_stops.add(self.destinations[p])

        # Calculate travel cost
        travel_cost = 0
        if required_stops:
            # Get levels for current floor and required stops
            # Filter out any floors not found in floor_levels (shouldn't happen in valid problems)
            floors_involved = required_stops | {current_lift_floor}
            levels_involved = {self.floor_levels[f] for f in floors_involved if f in self.floor_levels}

            if levels_involved:
                 current_level = self.floor_levels.get(current_lift_floor, None) # Use .get for safety
                 if current_level is not None: # Ensure current floor level was found
                     min_L = min(levels_involved)
                     max_L = max(levels_involved)

                     # Travel cost estimate: range + distance to nearest extreme
                     # This estimates the minimum vertical travel to cover the range [min_L, max_L] starting from current_level
                     travel_cost = (max_L - min_L) + min(abs(current_level - min_L), abs(current_level - max_L))
                 # else: current_lift_floor not in floor_levels, indicates problem with input
                 # In a real planner, might log a warning or return infinity.
                 # For this problem, assume valid input.
            # else: required_stops was not empty, but none of the floors were in floor_levels.
            # Indicates problem with input. Assume valid input.

        # Total heuristic
        h_value = base_cost + travel_cost

        return h_value
