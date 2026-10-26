# Add necessary imports
from heuristics.heuristic_base import Heuristic
import re
import math # For float('inf')

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding brackets and split by spaces
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function to extract floor index from name
def get_floor_index(floor_name):
    """Extracts the numerical index from a floor name like 'f12'."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        return int(match.group(1))
    # Handle cases where floor name might not match fN pattern, though unlikely in miconic
    # Returning a large value or raising an error are options.
    # Assuming valid miconic floor names for now.
    raise ValueError(f"Invalid floor name format: {floor_name}")


class miconicHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the miconic domain. It estimates the number
    of actions required to serve all passengers. The estimate is the sum of:
    1. The number of passengers currently at their origin (requiring a board action).
    2. The number of unserved passengers (requiring a depart action).
    3. An estimate of the minimum number of move actions (up/down) required
       to visit all necessary floors (origins of passengers not yet boarded,
       and destinations of all unserved passengers). The travel estimate is
       calculated as the distance from the current lift floor to the closest
       end of the range of required floors, plus the span of that range.

    Assumptions:
    - Floor names are in the format 'fN' where N is an integer, and higher N
      corresponds to a higher floor, consistent with the 'above' predicate.
    - The 'destin' predicate is static.
    - The 'origin' predicate is dynamic.
    - The goal is to serve all passengers mentioned in the initial goal state.

    Heuristic Initialization:
    - Parses static facts to identify all passengers and their destinations.
    - Builds a mapping from floor names (e.g., 'f1', 'f20') to integer indices
      based on the numerical part of the name, assuming 'f1' is the lowest floor.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current floor of the lift from the state.
    2. Identify all passengers that need to be served (those in the task goals).
    3. For each unserved passenger:
        a. Determine if they are at their origin (check state for '(origin p f)').
        b. Determine if they are boarded (check state for '(boarded p)').
        c. Determine their destination (check static facts for '(destin p f)').
        d. Count passengers at origin (`N_origin`).
        e. Count total unserved passengers (`N_unserved`).
        f. Collect the set of origin floors for passengers at origin (`F_pickup`).
        g. Collect the set of destination floors for all unserved passengers (`F_dropoff`).
    4. The set of floors the lift must visit is `F_visit = F_pickup U F_dropoff`.
    5. If there are no unserved passengers (`N_unserved == 0`), the heuristic is 0.
    6. Otherwise, map the floor names in `F_visit` and the current lift floor to integer indices using the precomputed mapping.
    7. Find the minimum (`idx_min_visit`) and maximum (`idx_max_visit`) indices among the required floors (`F_visit`).
    8. Calculate the estimated travel cost: `min(abs(current_floor_idx - idx_min_visit), abs(current_floor_idx - idx_max_visit)) + (idx_max_visit - idx_min_visit)`.
    9. The total heuristic value is `N_origin + N_unserved + Travel_cost`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals # Set of goal facts like '(served p1)'
        self.static_facts = task.static # Set of static facts

        # Precompute passenger destinations and floor mapping
        self.passenger_destinations = {}
        self.floor_to_idx = {}
        self.idx_to_floor = {}

        all_floors = set()
        for fact_str in self.static_facts:
            fact = parse_fact(fact_str)
            if fact[0] == 'destin':
                # fact is ('destin', passenger_name, floor_name)
                self.passenger_destinations[fact[1]] = fact[2]
            elif fact[0] == 'above':
                # fact is ('above', floor_above, floor_below)
                all_floors.add(fact[1])
                all_floors.add(fact[2])

        # Assuming floor names are f1, f2, ..., fn and correspond to levels
        # Sort floors based on the numerical part of the name
        try:
            sorted_floors = sorted(list(all_floors), key=get_floor_index)
            for idx, floor_name in enumerate(sorted_floors):
                 # Use 1-based indexing for floor levels
                self.floor_to_idx[floor_name] = idx + 1
                self.idx_to_floor[idx + 1] = floor_name
        except ValueError as e:
             # Handle cases with unexpected floor names if necessary
             print(f"Warning: Could not parse floor names. Heuristic may fail. {e}")
             self.floor_to_idx = {} # Empty mapping indicates failure


        # Extract all passenger names from goals
        self.all_passengers = set()
        for goal_fact_str in self.goals:
             goal_fact = parse_fact(goal_fact_str)
             if goal_fact[0] == 'served':
                 self.all_passengers.add(goal_fact[1])


    def __call__(self, node):
        state = node.state

        # 1. Find current lift floor
        current_floor = None
        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact[0] == 'lift-at':
                current_floor = fact[1]
                break

        # Check if floor mapping was successfully initialized
        if not self.floor_to_idx:
             # Initialization failed, cannot compute floor-based heuristic
             # Fallback to a simple heuristic or return infinity
             # Let's check if goal is reached as a fallback
             served_passengers_in_state = {parse_fact(f)[1] for f in state if parse_fact(f)[0] == 'served'}
             if self.all_passengers.issubset(served_passengers_in_state):
                 return 0 # Goal state
             else:
                 # Cannot compute meaningful heuristic without floor info
                 return math.inf # Use math.inf for infinity


        # 2. Identify unserved passengers and their status/needs
        unserved_passengers = set()
        passengers_at_origin = set() # Passengers at their origin floor
        boarded_passengers = set()   # Passengers currently boarded

        # Build sets of passengers currently served, at origin, or boarded
        served_passengers_in_state = {parse_fact(f)[1] for f in state if parse_fact(f)[0] == 'served'}
        origin_facts_in_state = {parse_fact(f) for f in state if parse_fact(f)[0] == 'origin'}
        boarded_passengers_in_state = {parse_fact(f)[1] for f in state if parse_fact(f)[0] == 'boarded'}

        for p in self.all_passengers:
            if p not in served_passengers_in_state:
                unserved_passengers.add(p)
                # Check if passenger is at their origin
                is_at_origin = False
                for fact in origin_facts_in_state:
                    if fact[1] == p:
                        passengers_at_origin.add(p)
                        is_at_origin = True
                        break # Found origin for this passenger

                # If not at origin, check if boarded
                if not is_at_origin and p in boarded_passengers_in_state:
                     boarded_passengers.add(p)


        N_unserved = len(unserved_passengers)

        # 5. If there are no unserved passengers, the heuristic is 0.
        if N_unserved == 0:
            return 0

        N_origin = len(passengers_at_origin)
        # N_boarded = len(boarded_passengers) # N_unserved = N_origin + N_boarded

        # 3f, 3g. Collect required floors
        F_pickup = set()
        F_dropoff = set()

        for p in passengers_at_origin:
             # Find origin floor from state
             f_orig_current = None
             for fact in origin_facts_in_state:
                 if fact[1] == p:
                     f_orig_current = fact[2]
                     break
             if f_orig_current: # Should always be found if in passengers_at_origin set
                 F_pickup.add(f_orig_current)
             # Find destination floor from precomputed static info
             if p in self.passenger_destinations:
                 F_dropoff.add(self.passenger_destinations[p])
             else:
                 # Unserved passenger has no destination in static facts - impossible state?
                 return math.inf


        for p in boarded_passengers:
             # Find destination floor from precomputed static info
             if p in self.passenger_destinations:
                 F_dropoff.add(self.passenger_destinations[p])
             else:
                 # Unserved passenger has no destination in static facts - impossible state?
                 return math.inf


        F_visit = F_pickup.union(F_dropoff)

        # If F_visit is empty, but N_unserved > 0, this is an impossible state
        # (e.g., passenger exists but has no origin/destin info, or is unserved
        # but not at origin or boarded). Return infinity.
        if not F_visit:
             return math.inf

        # If current_floor is None (should be caught earlier, but defensive check)
        if current_floor is None:
             return math.inf

        # 6. Map floor names to indices
        # Check if current_floor exists in the mapping (should if init succeeded)
        if current_floor not in self.floor_to_idx:
             return math.inf # Unknown current floor

        current_floor_idx = self.floor_to_idx[current_floor]

        # Check if all required floors exist in the mapping
        required_indices = set()
        for f in F_visit:
            if f in self.floor_to_idx:
                required_indices.add(self.floor_to_idx[f])
            else:
                return math.inf # Unknown required floor


        # 7. Find min/max required indices
        idx_min_visit = min(required_indices)
        idx_max_visit = max(required_indices)

        # 8. Calculate estimated travel cost
        dist_to_min = abs(current_floor_idx - idx_min_visit)
        dist_to_max = abs(current_floor_idx - idx_max_visit)
        span = idx_max_visit - idx_min_visit

        # The lift must travel from current_floor_idx to visit floors in F_visit.
        # A lower bound on travel is the distance from the current floor to the
        # closest end of the range [idx_min_visit, idx_max_visit], plus the
        # span of that range. This covers the distance to reach the 'action zone'
        # and then traverse the entire range where actions are needed.
        travel_cost = min(dist_to_min, dist_to_max) + span


        # 9. Total heuristic value
        # N_origin: Number of board actions needed (one for each passenger at origin)
        # N_unserved: Number of depart actions needed (one for each unserved passenger)
        # travel_cost: Estimated move actions needed
        h_value = N_origin + N_unserved + travel_cost

        return h_value
