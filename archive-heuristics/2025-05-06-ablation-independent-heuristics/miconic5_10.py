from heuristics.heuristic_base import Heuristic
# Assuming Task class is available in the environment where this heuristic runs
# from task import Task # Not strictly needed for the code itself, but good for context

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the number of actions required to reach the goal state
        (all passengers served). The heuristic is the sum of the number
        of unserved passengers and an estimate of the necessary lift
        movement actions to visit all required pickup and dropoff floors.

    Assumptions:
        - The PDDL domain is miconic as provided.
        - The 'above' predicates define a total order on floors forming a single chain.
        - Facts in the state and static information are strings in the format
          '(predicate arg1 arg2 ...)'.

    Heuristic Initialization:
        In the constructor, the static information from the task is processed:
        1. All 'above' predicates are parsed to determine the floor order and
           build a mapping from floor names to numerical indices (floor_to_index).
           The lowest floor (the one not appearing as the first argument in any
           'above' predicate) is assigned index 0, the next floor up index 1, and so on.
           A mapping from lower floor to immediately higher floor is built to traverse
           the floor chain.
        2. All 'destin' predicates are parsed to build a mapping from passenger
           names to their destination floor names (passenger_destin).

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Identify the current floor of the lift by finding the '(lift-at ?f)' fact.
        2. Identify all passengers who are not yet served by checking for the
           absence of '(served ?p)' facts.
        3. For each unserved passenger:
           - If the passenger is at their origin floor (check for '(origin ?p ?f)' fact),
             they are unboarded and unserved. Add their origin floor to the set of
             required pickup stops (S_board).
           - If the passenger is boarded (check for '(boarded ?p)' fact), they are
             boarded and unserved. Add their destination floor (looked up from
             passenger_destin) to the set of required dropoff stops (S_depart).
        4. Calculate the 'base cost': This is the number of unserved passengers.
           Each unserved passenger represents a remaining goal fact `(served ?p)`
           that needs to be achieved. This serves as a simple count of remaining
           high-level tasks (serving a passenger).
           Base cost = count of unserved passengers.
        5. Calculate the 'movement cost': This estimates the lift travel needed.
           - Determine the set of all floors the lift must visit (F_required_stops)
             by taking the union of S_board and S_depart.
           - If F_required_stops is empty, movement cost is 0.
           - Otherwise, find the minimum and maximum floor indices among the
             required stops (min_idx, max_idx) using the floor_to_index map.
           - Get the index of the current lift floor (f_current_idx).
           - The minimum movement to cover the range [min_idx, max_idx] starting
             from f_current_idx is estimated as the distance from f_current_idx
             to the closest end of the range (min_idx or max_idx) plus the length
             of the range (max_idx - min_idx). This is calculated as
             min(abs(f_current_idx - min_idx), abs(f_current_idx - max_idx)) + (max_idx - min_idx).
           - If the current floor or any required stop floor is not found in the
             floor_to_index map (indicating an issue with the problem definition
             or floor indexing logic), the heuristic returns infinity.
        6. The total heuristic value is the sum of the base cost and the movement cost.
    """

    def __init__(self, task):
        super().__init__()
        self.floor_to_index = {}
        self.passenger_destin = {}
        self.floors = set()
        above_pairs = []

        # Parse static facts
        for fact_str in task.static:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'above':
                f_higher, f_lower = args
                above_pairs.append((f_higher, f_lower))
                self.floors.add(f_higher)
                self.floors.add(f_lower)
            elif predicate == 'destin':
                p, f = args
                self.passenger_destin[p] = f

        # Build floor index map from above_pairs
        if self.floors:
            # Map lower floor to immediately higher floor
            lower_to_higher = {}
            # Set of floors that appear as the first argument in an 'above' predicate
            floors_that_are_above_others = set()

            for f_higher, f_lower in above_pairs:
                lower_to_higher[f_lower] = f_higher
                floors_that_are_above_others.add(f_higher)

            # Find the lowest floor: a floor in self.floors that is not in floors_that_are_above_others
            lowest_floors_candidates = list(self.floors - floors_that_are_above_others)

            if not lowest_floors_candidates:
                 # This might mean a single floor or a cycle or disconnected components
                 if len(self.floors) == 1:
                     lowest_floor = list(self.floors)[0]
                 else:
                     # Handle cases where lowest floor isn't uniquely identifiable
                     # Fallback: alphabetical sort - highly domain dependent!
                     # This might be incorrect if floor names don't sort according to 'above'.
                     print("Warning: Could not determine a unique lowest floor from 'above' predicates. Falling back to alphabetical sort.")
                     sorted_floors = sorted(list(self.floors))
                     self.floor_to_index = {f: i for i, f in enumerate(sorted_floors)}
                     return # Exit __init__

            else:
                # Assuming a single chain, there should be exactly one lowest floor
                lowest_floor = lowest_floors_candidates[0] # Pick the first one if multiple (problematic)

                # Build map upwards starting from the lowest floor
                current = lowest_floor
                index = 0
                # Use a visited set to prevent infinite loops in case of cycles (though unlikely in miconic)
                visited_floors = set()
                while current in self.floors and current not in visited_floors:
                    visited_floors.add(current)
                    self.floor_to_index[current] = index
                    if current in lower_to_higher:
                        current = lower_to_higher[current]
                        index += 1
                    else:
                        break # Reached the highest floor

                # Check if all floors that appeared in 'above' predicates were indexed
                all_above_floors = floors_that_are_above_others | set(lower_to_higher.keys())
                if not all_above_floors.issubset(set(self.floor_to_index.keys())):
                     print("Warning: Not all floors mentioned in 'above' predicates were indexed. 'above' predicates might not form a single chain covering all relevant floors.")
                     # This might lead to KeyErrors later if unindexed floors are encountered.

        # If self.floors is empty, floor_to_index remains empty, handled in __call__


    def _parse_fact(self, fact_str):
        """Helper to parse a fact string like '(predicate arg1 arg2)'."""
        # Assumes fact_str is like '(predicate arg1 ...)'
        parts = fact_str.strip('()').split()
        if not parts: # Handle empty string case
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        # Goal is (served ?p) for all passengers identified during init
        all_served = True
        # Iterate through all passengers identified during init
        for p in self.passenger_destin.keys():
            if f'(served {p})' not in state:
                all_served = False
                break
        if all_served:
            return 0 # Goal reached

        f_current = None
        unserved_passengers = set(self.passenger_destin.keys()) # Start with all passengers

        # Collect state information and identify served passengers
        origin_facts = {} # Map passenger to origin floor
        boarded_facts = set() # Set of boarded passenger names

        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'lift-at':
                f_current = args[0]
            elif predicate == 'served':
                p = args[0]
                unserved_passengers.discard(p) # Remove served passenger from consideration
            elif predicate == 'origin':
                p, f = args
                origin_facts[p] = f
            elif predicate == 'boarded':
                p = args[0]
                boarded_facts.add(p)

        # Identify required stops for unserved passengers
        S_board = set() # Origin floors for unboarded unserved passengers
        S_depart = set() # Destination floors for boarded unserved passengers

        for p in unserved_passengers:
            if p in origin_facts:
                # Unserved and at origin -> needs boarding
                S_board.add(origin_facts[p])
            elif p in boarded_facts:
                # Unserved and boarded -> needs departing
                # Look up destination from pre-calculated map
                if p in self.passenger_destin:
                    S_depart.add(self.passenger_destin[p])
                # else: This case shouldn't happen in a valid problem instance
                # where all passengers have destinations.

        # Calculate base cost
        # Base cost is the number of unserved passengers.
        base_cost = len(unserved_passengers)

        # Calculate movement cost
        F_required_stops = S_board | S_depart

        movement_cost = 0
        if F_required_stops:
            # Ensure current floor is known and indexed
            if f_current is None or f_current not in self.floor_to_index:
                 print(f"Error: Current lift floor '{f_current}' not found or not indexed.")
                 return float('inf') # Indicate problematic state

            f_current_idx = self.floor_to_index[f_current]

            # Get indices of required stops, checking if floors are indexed
            required_indices = set()
            for f in F_required_stops:
                if f in self.floor_to_index:
                    required_indices.add(self.floor_to_index[f])
                else:
                    # Required stop floor not in index map. Problematic.
                    print(f"Error: Required stop floor '{f}' not found or not indexed.")
                    return float('inf') # Indicate problematic state

            if required_indices: # Ensure there are valid required indices
                min_idx = min(required_indices)
                max_idx = max(required_indices)

                # Movement cost: distance to nearest end of range + range span
                dist_to_min = abs(f_current_idx - min_idx)
                dist_to_max = abs(f_current_idx - max_idx)
                range_span = max_idx - min_idx

                movement_cost = min(dist_to_min, dist_to_max) + range_span
            # else: required_indices was empty, movement_cost remains 0. This case
            # should not happen if F_required_stops was not empty and all floors indexed.


        return base_cost + movement_cost
