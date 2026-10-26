from heuristics.heuristic_base import Heuristic
from task import Task


class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.
    """

    def __init__(self, task: Task):
        """
        Initializes the miconic heuristic by precomputing static information
        about passenger destinations and floor levels.

        Keyword arguments:
        task -- an instance of the Task class containing domain information.
        """
        super().__init__()
        self.task = task

        # Data structures for static info
        self.passenger_to_dest_floor = {}
        self.floor_to_level = {}
        self.all_floors = set()
        self.all_passengers = set()

        # --- Precomputation Step 1: Collect all objects ---
        # This is a simplification based on typical PDDL object naming conventions
        # in miconic instances (f* for floors, p* for passengers).
        # A proper PDDL parser would provide typed objects directly.
        all_objects = set()
        # Collect objects mentioned in static, initial, and goal facts
        for fact_str in task.static | task.initial_state | task.goals:
             # Remove surrounding brackets and split by space
             parts = fact_str[1:-1].split()
             # Assume arguments after predicate are objects
             for arg in parts[1:]:
                 all_objects.add(arg)

        # Filter objects by assumed type prefix
        self.all_floors = {obj for obj in all_objects if obj.startswith('f')}
        self.all_passengers = {obj for obj in all_objects if obj.startswith('p')}


        # --- Precomputation Step 2: Parse static facts for destinations and floor ordering ---
        above_map = {} # Maps lower floor to higher floor (f_lower -> f_higher)

        for fact_str in task.static:
            # Parse fact string
            parts = fact_str[1:-1].split() # Remove brackets and split
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'destin':
                # Format: (destin passenger floor)
                if len(args) == 2:
                    passenger, floor = args
                    self.passenger_to_dest_floor[passenger] = floor
            elif predicate == 'above':
                # Format: (above f_higher f_lower)
                if len(args) == 2:
                    f_higher, f_lower = args
                    # Store mapping from lower to higher
                    above_map[f_lower] = f_higher


        # --- Precomputation Step 3: Determine floor levels ---
        # Interpretation: (above f_higher f_lower) means f_higher is immediately above f_lower.
        # The lowest floor is the one that is never a 'f_higher' in any 'above' fact.
        floors_that_are_above_others = set(above_map.values()) # These are the higher floors mentioned in 'above'
        lowest_floors = [f for f in self.all_floors if f not in floors_that_are_above_others]

        if len(lowest_floors) != 1:
             # This heuristic assumes a single linear tower of floors.
             # If there isn't exactly one lowest floor, the structure is unexpected.
             # Handle cases like zero floors or disconnected floors.
             if not self.all_floors:
                 # No floors defined, cannot compute heuristic.
                 pass # floor_to_level remains empty
             elif len(self.all_floors) == 1:
                 # Single floor case
                 self.floor_to_level[list(self.all_floors)[0]] = 0
             else:
                 # Disconnected or non-linear floors. The heuristic's travel cost
                 # calculation relies on a linear structure. This is a limitation.
                 # We cannot reliably assign levels in this case.
                 # Accessing floor_to_level later will raise KeyError, signaling the issue.
                 pass # floor_to_level remains incomplete


        else:
            # Found a unique lowest floor, proceed with linear level assignment
            lowest_floor = lowest_floors[0]
            current_floor = lowest_floor
            level = 0
            # Traverse upwards using the above_map (lower -> higher)
            while current_floor in above_map:
                self.floor_to_level[current_floor] = level
                current_floor = above_map[current_floor]
                level += 1
            # Assign level to the highest floor (the last one in the chain)
            self.floor_to_level[current_floor] = level


    def __call__(self, node):
        """
        Computes the miconic domain-dependent heuristic for a given state.

        Summary:
        The heuristic estimates the remaining actions needed to serve all passengers.
        It considers the travel cost for the lift to visit all necessary floors
        (origins of unboarded passengers and destinations of boarded passengers)
        plus the number of board and depart actions required.

        Assumptions:
        - The floors form a single linear tower, ordered by the 'above' predicate,
          where (above f_higher f_lower) means f_higher is immediately above f_lower.
        - Standard miconic domain predicates and actions are used.
        - The PDDL instance is well-formed according to the domain.
        - Object names starting with 'f' are floors, 'p' are passengers (simplification).

        Heuristic Initialization:
        In the constructor (__init__), the heuristic precomputes static information:
        1. It collects all potential floor and passenger objects based on naming conventions.
        2. It parses the 'destin' facts from the static information to map each
           passenger to their destination floor.
        3. It parses the 'above' facts to determine the linear ordering of floors
           and assigns a numerical level to each floor (starting from 0 for the lowest).
           It assumes (above f_higher f_lower) means f_higher is immediately above f_lower.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift from the state fact '(lift-at ?f)'.
           Get the numerical level of the current floor using the precomputed map.
           If the lift location or its level is unknown (e.g., due to invalid PDDL structure), return infinity.
        2. Identify all unserved passengers. A passenger is unserved if the fact
           '(served ?p)' is not in the current state. If there are no unserved
           passengers, the goal is reached, return 0.
        3. Determine the set of floors the lift *must* visit to serve the unserved
           passengers. This set includes:
           - The origin floor for every unserved passenger who is currently waiting
             at their origin (fact '(origin ?p ?f)' is true in the state).
           - The destination floor for every unserved passenger who is currently
             boarded in the lift (fact '(boarded ?p)' is true in the state).
           If a passenger's destination is unknown (not in static facts), return infinity.
        4. Count the number of unboarded unserved passengers (those with '(origin ...)' fact)
           and the number of boarded unserved passengers (those with '(boarded ...)' fact).
           These counts represent the minimum number of 'board' and 'depart' actions needed, respectively.
        5. If the set of required stops is empty, it means all unserved passengers
           must be already boarded and at their destination floors. The remaining
           actions are just the 'depart' actions for these passengers. The heuristic
           value is the number of such unserved passengers (which equals the count
           of boarded unserved passengers in this specific case).
        6. If the set of required stops is not empty:
           - Get the levels for all floors in the required stops set. If any level
             is missing (e.g., due to invalid floor structure), return infinity.
           - Find the minimum and maximum floor levels among the required stops.
           - Estimate the minimum travel cost (number of up/down moves) required
             for the lift to start at the current floor and visit all required stops.
             A lower bound for this is the minimum moves to reach either the
             minimum or maximum required level, plus the distance between the
             minimum and maximum required levels. This is calculated as
             `min(abs(current_level - min_req_level), abs(current_level - max_req_level)) + (max_req_level - min_req_level)`.
           - The total heuristic value is the sum of the estimated travel cost
             and the estimated action cost (number of board actions + number of depart actions).
        """
        state = node.state

        # 1. Find current lift floor and level
        current_floor = None
        # Using set for faster lookup
        state_facts_set = set(state)
        for fact_str in state_facts_set:
            if fact_str.startswith('(lift-at '):
                # Extract floor name from '(lift-at floor)'
                parts = fact_str[1:-1].split()
                if len(parts) == 2:
                    current_floor = parts[1]
                break

        # Check if lift location is found and its level is mapped
        if current_floor is None or current_floor not in self.floor_to_level:
             # Lift location not found or level mapping failed for this floor during init
             return float('inf')

        current_level = self.floor_to_level[current_floor]

        # 2. Identify unserved passengers
        served_passengers = set()
        for fact_str in state_facts_set:
            if fact_str.startswith('(served '):
                # Extract passenger name from '(served passenger)'
                parts = fact_str[1:-1].split()
                if len(parts) == 2:
                    served_passengers.add(parts[1])

        unserved_passengers = {p for p in self.all_passengers if p not in served_passengers}

        if not unserved_passengers:
            # Goal state reached (all passengers served)
            return 0

        # 3. Determine required stops and action counts
        pickup_floors = set()
        dropoff_floors = set()
        num_unboarded = 0
        num_boarded_unserved = 0

        for passenger in unserved_passengers:
            dest_floor = self.passenger_to_dest_floor.get(passenger)
            if dest_floor is None:
                 # Destination not found for an unserved passenger (invalid instance?)
                 return float('inf')

            origin_fact_prefix = f'(origin {passenger} '
            boarded_fact = f'(boarded {passenger})'

            is_at_origin = False
            # Check for origin fact
            for fact_str in state_facts_set:
                 if fact_str.startswith(origin_fact_prefix):
                      # Extract origin floor from '(origin passenger floor)'
                      parts = fact_str[1:-1].split()
                      if len(parts) == 3 and parts[1] == passenger:
                          origin_floor = parts[2]
                          pickup_floors.add(origin_floor)
                          num_unboarded += 1
                          is_at_origin = True
                          break # Found origin fact for this passenger

            # Check for boarded fact
            is_boarded = boarded_fact in state_facts_set

            if is_boarded:
                 dropoff_floors.add(dest_floor)
                 num_boarded_unserved += 1

            # Basic validation: unserved passenger must be at origin or boarded
            # If not, the state is likely unreachable or a dead end.
            # We don't explicitly return inf here, assuming valid states are reachable.


        required_stops = pickup_floors | dropoff_floors

        # 4. Calculate heuristic
        action_cost = num_unboarded + num_boarded_unserved

        if not required_stops:
            # Case 5: All unserved passengers are boarded and at their destination
            # The only remaining actions are 'depart'.
            # The number of such passengers is num_boarded_unserved (which equals action_cost here).
            heuristic_value = action_cost
        else:
            # Case 6: Need to visit floors
            required_levels = []
            for f in required_stops:
                 level = self.floor_to_level.get(f)
                 if level is None:
                      # Level mapping failed for a required floor during init
                      return float('inf')
                 required_levels.append(level)

            min_req_level = min(required_levels)
            max_req_level = max(required_levels)

            # Estimate travel cost: min moves to reach the range [min_req_level, max_req_level]
            # from current_level, and then traverse the range.
            travel_cost = min(abs(current_level - min_req_level), abs(current_level - max_req_level)) + (max_req_level - min_req_level)

            # Total heuristic
            heuristic_value = travel_cost + action_cost

        return heuristic_value
