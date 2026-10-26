import re

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unserved passenger independently. For each
    unserved passenger, it calculates the cost of moving the lift to their
    current location (origin or current lift), boarding them (if not already
    boarded), moving the lift to their destination, and departing them.
    The movement cost between floors is estimated by the absolute difference
    in their floor indices, assuming a linear ordering of floors derived
    from the 'above' predicates. This heuristic is non-admissible as it
    ignores the fact that the lift can pick up/drop off multiple passengers
    at the same floor and that lift movements can serve multiple passengers'
    needs simultaneously.

    Assumptions:
    - The 'above' predicates define a total linear order on floors.
    - Floor names can be ordered based on the 'above' relations.
    - Passenger destinations are static and provided in the initial state/static facts.
    - The goal is to have all passengers 'served'.

    Heuristic Initialization:
    The heuristic is initialized with the planning task. During initialization,
    it parses the static facts to:
    1. Determine the linear order of floors based on the 'above' predicates
       and create a mapping from floor names to integer indices.
    2. Create a mapping from passenger names to their destination floor names.
    3. Identify all passenger names in the problem.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the state is a goal state (all passengers served). If yes, the heuristic is 0.
    2. Identify the current floor of the lift by finding the fact '(lift-at ?f)' in the state.
    3. Create a temporary mapping of passengers currently at their origin floor by
       finding all facts '(origin ?p ?f)' in the state.
    4. Initialize the total heuristic value `h` to 0.
    5. Iterate through all passengers identified during initialization.
    6. For each passenger `p`:
        a. Check if the fact '(served p)' is present in the current state. If yes, this passenger is served, so continue to the next passenger.
        b. If the passenger is not served, retrieve their destination floor `f_destin` from the precomputed destination map.
        c. Check if the fact '(boarded p)' is present in the current state.
            i. If '(boarded p)' is in the state:
               - The passenger is in the lift and needs to be dropped off.
               - The estimated cost for this passenger is the distance from the current lift floor to `f_destin` plus 1 action (depart).
               - Distance is calculated as the absolute difference between the floor indices of the current lift floor and `f_destin`.
               - Add this cost to `h`.
            ii. If '(boarded p)' is NOT in the state:
               - The passenger is at their origin floor. Find their origin floor `f_origin` from the temporary origins map created in step 3.
               - The estimated cost for this passenger is the distance from the current lift floor to `f_origin` plus 1 action (board), plus the distance from `f_origin` to `f_destin` plus 1 action (depart).
               - Distances are calculated as the absolute differences between the respective floor indices.
               - Add this cost to `h`.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static task information.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.task = task
        self.static = task.static
        self.goals = task.goals

        # Precompute floor order and destination map from static facts
        self.floor_to_index = self._parse_floor_order(self.static)
        self.destin_map, self.passengers = self._parse_destinations(self.static)

    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove surrounding brackets and split by spaces
        parts = fact_string[1:-1].split()
        return parts[0], parts[1:]

    def _parse_floor_order(self, static_facts):
        """
        Parses 'above' facts to determine floor order and create index map.
        Assumes a linear order where (above f_i f_j) means f_i is immediately
        above f_j if there's no floor between them, or f_i is some number
        of floors above f_j. We build a simple linear index.
        """
        floor_above_map = {} # f_below -> f_above
        floor_below_map = {} # f_above -> f_below
        all_floors = set()

        for fact_string in static_facts:
            pred, args = self._parse_fact(fact_string)
            if pred == 'above':
                f1, f2 = args # f1 is above f2
                floor_below_map[f1] = f2
                floor_above_map[f2] = f1
                all_floors.add(f1)
                all_floors.add(f2)

        if not all_floors:
             # Handle case with no floors or no above facts (e.g., empty problem)
             return {}

        # Find the lowest floor (a floor that is not a value in floor_above_map)
        # Or, a floor that is not a key in floor_below_map (nothing is below it)
        lowest_floor = None
        # Find a floor that is not a value in floor_above_map
        potential_lowest = all_floors.copy()
        for f_above in floor_above_map.values():
             if f_above in potential_lowest:
                  potential_lowest.remove(f_above)

        if potential_lowest:
             # If multiple candidates, pick one (e.g., alphabetically)
             lowest_floor = sorted(list(potential_lowest))[0]
        else:
             # This case should ideally not happen in a well-formed miconic problem
             # with above facts defining a chain, but handle defensively.
             # If all floors are above some other floor, maybe it's a cycle or single floor?
             # For linear chains, there must be one floor not above anything else.
             # Let's assume the lowest is the one not found as a value in floor_above_map
             # (i.e., nothing is immediately below it in the map).
             # A simpler approach for standard miconic: find the floor that is not a key in floor_below_map
             # (nothing is above it, meaning it's the highest) and traverse down.
             # Let's rebuild the map direction for easier traversal from bottom up.
             floor_above_map = {} # f_below -> f_above
             floor_below_map = {} # f_above -> f_below
             for fact_string in static_facts:
                 pred, args = self._parse_fact(fact_string)
                 if pred == 'above':
                     f1, f2 = args # f1 is above f2
                     floor_above_map[f2] = f1 # f2 is below f1

             # Find the lowest floor: it's not a key in floor_above_map
             lowest_floor_candidates = all_floors.copy()
             for f_below in floor_above_map.keys():
                 if f_below in lowest_floor_candidates:
                     lowest_floor_candidates.remove(f_below)

             if lowest_floor_candidates:
                 lowest_floor = sorted(list(lowest_floor_candidates))[0] # Pick one if multiple
             else:
                 # Still no lowest floor found, maybe only one floor or cycle?
                 # If only one floor, it's the lowest.
                 if len(all_floors) == 1:
                     lowest_floor = list(all_floors)[0]
                 else:
                     # This indicates an issue with the 'above' facts not forming a clear linear order.
                     # Return empty map or raise error. Returning empty map makes heuristic 0.
                     print("Warning: Could not determine linear floor order from 'above' facts.")
                     return {}


        # Build the ordered list of floors starting from the lowest
        ordered_floors = []
        current_floor = lowest_floor
        while current_floor is not None:
            ordered_floors.append(current_floor)
            # Find the floor immediately above the current one
            next_floor = None
            for f_above, f_below in floor_above_map.items():
                if f_below == current_floor:
                    next_floor = f_above
                    break # Assuming unique floor above
            current_floor = next_floor

        # Create the floor_to_index map
        floor_to_index = {floor: index for index, floor in enumerate(ordered_floors)}

        return floor_to_index


    def _parse_destinations(self, static_facts):
        """Parses 'destin' facts to create a destination map and passenger list."""
        destin_map = {}
        passengers = set()
        for fact_string in static_facts:
            pred, args = self._parse_fact(fact_string)
            if pred == 'destin':
                p, f = args
                destin_map[p] = f
                passengers.add(p)
        return destin_map, sorted(list(passengers)) # Return sorted list for consistent iteration

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: A frozenset of fact strings representing the current state.

        Returns:
            An integer estimating the remaining cost to the goal.
        """
        # 1. Check if goal state
        if self.task.goal_reached(state):
            return 0

        # Handle case where floor order couldn't be determined
        if not self.floor_to_index:
             # Cannot compute meaningful distance, return 0 or number of unserved passengers
             # Number of unserved passengers is a weak but valid non-admissible heuristic
             unserved_count = 0
             for p in self.passengers:
                 if f"'(served {p})'" not in state:
                     unserved_count += 1
             return unserved_count # Or just 0 if we can't estimate movement

        # 2. Find current lift floor and origins in the state
        current_lift_floor = None
        current_origins = {} # passenger -> origin_floor
        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'lift-at':
                current_lift_floor = args[0]
            elif pred == 'origin':
                p, f = args
                current_origins[p] = f

        if current_lift_floor is None:
             # This should not happen in a valid miconic state, but handle defensively
             # If lift location is unknown, cannot estimate movement cost.
             # Fallback to number of unserved passengers or a large value.
             # Let's use number of unserved passengers * a constant (e.g., 3 for board/depart/move)
             unserved_count = 0
             for p in self.passengers:
                 if f"'(served {p})'" not in state:
                     unserved_count += 1
             return unserved_count * 3 # Or some other fallback

        current_lift_floor_index = self.floor_to_index.get(current_lift_floor)
        if current_lift_floor_index is None:
             # Unknown floor, fallback
             print(f"Warning: Lift is at unknown floor {current_lift_floor}.")
             unserved_count = 0
             for p in self.passengers:
                 if f"'(served {p})'" not in state:
                     unserved_count += 1
             return unserved_count * 3


        # 4. Initialize heuristic
        h = 0

        # 5. Iterate through all passengers
        for p in self.passengers:
            # 6a. Check if served
            if f"'(served {p})'" in state:
                continue # Passenger is served

            # Passenger is not served
            f_destin = self.destin_map.get(p)
            if f_destin is None:
                 # Should not happen if passengers list is built from destin map
                 continue # Cannot estimate cost without destination

            f_destin_index = self.floor_to_index.get(f_destin)
            if f_destin_index is None:
                 # Destination floor not in floor index map, cannot estimate distance
                 # Add a base cost for this passenger (e.g., 2 for board/depart)
                 h += 2
                 continue


            # 6c. Check if boarded
            if f"'(boarded {p})'" in state:
                # Passenger is boarded, needs to depart at destination
                cost_p = abs(current_lift_floor_index - f_destin_index) + 1 # move + depart
                h += cost_p
            else:
                # Passenger is at origin, needs board and depart
                f_origin = current_origins.get(p)
                if f_origin is None:
                    # Passenger is not served, not boarded, and not at an origin?
                    # This state might be unreachable or malformed.
                    # Add a base cost or a large penalty. Add base cost (board + depart).
                    h += 2
                    continue

                f_origin_index = self.floor_to_index.get(f_origin)
                if f_origin_index is None:
                     # Origin floor not in floor index map, cannot estimate distance
                     # Add a base cost for this passenger (e.g., 2 for board/depart)
                     h += 2
                     continue

                cost_p = abs(current_lift_floor_index - f_origin_index) + 1 + \
                         abs(f_origin_index - f_destin_index) + 1 # move to origin + board + move to destin + depart
                h += cost_p

        # 7. Return total heuristic value
        return h

# Example usage (assuming 'task' object is available):
# heuristic = miconicHeuristic(task)
# h_value = heuristic(current_state)
