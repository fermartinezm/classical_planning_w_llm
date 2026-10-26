from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse a PDDL fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the cost to reach the goal by summing:
        1. The number of passengers currently waiting at their origin (representing board actions).
        2. The number of passengers currently boarded (representing depart actions).
        3. The estimated minimum movement cost for the lift to visit all floors where passengers are waiting.
        4. The sum of the vertical distances each unserved passenger needs to travel from their origin to their destination while inside the lift.

    Assumptions:
        - The 'above' predicate defines a linear order of floors, where (above f_higher f_lower) means f_higher is immediately above f_lower.
        - The cost of 'up' and 'down' actions is 1 per floor level change.
        - The cost of 'board' and 'depart' actions is 1.
        - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
        - Parses static facts to build a map from floor names to integer levels based on the 'above' predicate. Assumes a linear floor structure.
        - Parses initial state facts to store the origin and destination floor for each passenger.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift from the state.
        2. Identify all passengers who have not yet been served.
        3. For each unserved passenger:
           - Determine if they are waiting at their origin or are boarded.
           - Store their origin and destination floors (retrieved from initialization).
        4. Calculate the number of waiting passengers and the number of boarded passengers among the unserved ones. These counts contribute directly to the heuristic (representing board/depart actions).
        5. Determine the set of 'pickup floors': the origin floors of all currently waiting passengers.
        6. Calculate the estimated movement cost for the lift to visit all pickup floors:
           - If there are no pickup floors, this cost is 0.
           - Otherwise, find the minimum and maximum floor levels among the pickup floors.
           - Calculate the minimum vertical distance the lift must travel from its current floor to reach either the minimum or maximum pickup floor level, plus the distance to traverse the range between the minimum and maximum pickup floor levels. This is `min(abs(L_current - L_min_pickup), abs(L_current - L_max_pickup)) + (L_max_pickup - L_min_pickup)`.
        7. Calculate the total vertical distance required for all unserved passengers to travel from their origin to their destination. Sum `abs(level_map[origin] - level_map[destination])` for each unserved passenger.
        8. The total heuristic value is the sum of (number of waiting passengers) + (number of boarded passengers) + (estimated movement cost to pickup floors) + (total passenger trip distance).
        9. If all passengers are served, the heuristic is 0.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Build floor level map from static facts
        # below_map maps floor_higher -> floor_lower (if f_higher is immediately above f_lower)
        below_map = {}
        all_floors_in_above = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_higher, f_lower = parts[1], parts[2]
                below_map[f_higher] = f_lower
                all_floors_in_above.add(f_higher)
                all_floors_in_above.add(f_lower)

        self.level_map = {}
        if all_floors_in_above:
            # Find the highest floor: a floor that is a key in below_map but not a value.
            potential_highest_floors = list(below_map.keys() - below_map.values())

            highest_floor = None
            if potential_highest_floors:
                 highest_floor = potential_highest_floors[0] # Assuming a single highest floor

            if highest_floor:
                # Build level map starting from the highest floor
                current_floor = highest_floor
                level = len(all_floors_in_above) # Assign highest level to the highest floor
                while current_floor in below_map: # Traverse downwards
                    if current_floor in self.level_map: break # Avoid infinite loops
                    self.level_map[current_floor] = level
                    current_floor = below_map[current_floor]
                    level -= 1
                # Add the lowest floor which is not a key in below_map
                if current_floor not in self.level_map:
                     self.level_map[current_floor] = level

            # Handle case with single floor not involved in 'above' facts but present in initial state/goals
            # Collect all floors mentioned in initial state and goals
            all_mentioned_floors = set(self.level_map.keys())
            for fact in initial_state:
                 parts = get_parts(fact)
                 # Check for floor objects based on type or naming convention 'f...'
                 # Assuming floors start with 'f' and are the 2nd or 3rd part of a fact
                 if len(parts) > 1 and parts[1].startswith('f'):
                      all_mentioned_floors.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('f'):
                      all_mentioned_floors.add(parts[2])
            for goal in self.goals:
                 parts = get_parts(goal)
                 if len(parts) > 1 and parts[1].startswith('f'):
                      all_mentioned_floors.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('f'):
                      all_mentioned_floors.add(parts[2])

            # Assign level 1 to any mentioned floor not in level_map (e.g., single floor problem)
            # This assumes any floor not in the 'above' chain is the lowest or a disconnected floor.
            # For simplicity in miconic, a single floor problem would have no 'above' facts,
            # and this loop would assign level 1 to that floor.
            for floor in all_mentioned_floors:
                 if floor not in self.level_map:
                      self.level_map[floor] = 1 # Assign level 1 as a fallback


        # 2. Store initial origins and destinations
        self.initial_origins = {}
        self.initial_destinations = {}
        self.all_passengers = set()
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                p, f = parts[1], parts[2]
                self.initial_origins[p] = f
                self.all_passengers.add(p)
            elif parts[0] == "destin":
                p, f = parts[1], parts[2]
                self.initial_destinations[p] = f
                self.all_passengers.add(p)

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        # The goal is a conjunction of (served ?p) for all initial passengers
        # We can check if all initial passengers are served.
        all_served = True
        for p in self.all_passengers:
            if '(served ' + p + ')' not in state:
                all_served = False
                break
        if all_served:
             return 0

        # 1. Find current lift floor
        current_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                current_floor = parts[1]
                break

        # If lift location is unknown, return a high value (shouldn't happen in valid states)
        if current_floor is None:
             return float('inf') # Or some large number

        current_level = self.level_map.get(current_floor, 0) # Default to 0 if floor not in map

        # 2. Identify unserved passengers and their state (waiting/boarded)
        unserved_passengers = set()
        waiting_passengers = set()
        boarded_passengers = set()
        pickup_floors = set()
        total_passenger_trip_distance = 0

        for p in self.all_passengers:
            if '(served ' + p + ')' not in state:
                unserved_passengers.add(p)

                origin_floor = self.initial_origins.get(p)
                destin_floor = self.initial_destinations.get(p)

                if origin_floor and destin_floor:
                    origin_level = self.level_map.get(origin_floor, 0)
                    destin_level = self.level_map.get(destin_floor, 0)
                    total_passenger_trip_distance += abs(origin_level - destin_level)

                    if '(boarded ' + p + ')' in state:
                        boarded_passengers.add(p)
                    elif '(origin ' + p + ' ' + origin_floor + ')' in state:
                         waiting_passengers.add(p)
                         pickup_floors.add(origin_floor)
                    # else: unserved but not waiting or boarded? (e.g. just departed, but served not added yet?)
                    # Assuming valid states where unserved passengers are either waiting or boarded.


        # 4. Calculate counts for action cost part
        num_waiting = len(waiting_passengers)
        num_boarded = len(board_passengers)

        # 5. Calculate estimated movement cost to visit pickup floors
        movement_to_pickup = 0
        if pickup_floors:
            # Only consider pickup floors that are in our level map
            valid_pickup_levels = {self.level_map[f] for f in pickup_floors if f in self.level_map}

            if valid_pickup_levels:
                L_min_pickup = min(valid_pickup_levels)
                L_max_pickup = max(valid_pickup_levels)
                movement_to_pickup = min(abs(current_level - L_min_pickup), abs(current_level - L_max_pickup)) + (L_max_pickup - L_min_pickup)

        # 8. Total heuristic
        heuristic_value = num_waiting + num_boarded + movement_to_pickup + total_passenger_trip_distance

        return heuristic_value
