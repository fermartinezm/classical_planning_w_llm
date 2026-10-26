from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (as seen in example heuristics)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like "(predicate)" or "(predicate arg)"
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return [] # Invalid fact format
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the number of 'board' actions needed (for waiting passengers),
    the number of 'depart' actions needed (for all unserved passengers),
    and adds an estimate of the lift movement cost to visit all necessary floors.

    # Assumptions
    - Floors are linearly ordered by 'above' facts, forming a single chain.
    - Passengers need to be picked up at their origin and dropped off at their destination.
    - The lift can carry multiple passengers.
    - Action costs are uniform (each action costs 1).

    # Heuristic Initialization
    - Parses 'above' facts to establish the floor order and create mappings between floor names and their integer indices.
    - Parses 'destin' facts to map each passenger to their destination floor.
    - Gathers all floor names mentioned in the problem (init and static) to handle single-floor cases or cases without 'above' facts gracefully.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify the set of passengers who are currently waiting at their origin floors ('P_wait').
    3. Identify the set of passengers who are currently boarded in the lift ('P_boarded').
    4. The set of unserved passengers ('P_unserved') is the union of 'P_wait' and 'P_boarded'.
    5. Determine the set of floors the lift *must* visit:
       - Origin floors for all passengers in 'P_wait'.
       - Destination floors for all passengers in 'P_boarded'.
       Let this set be 'RequiredFloors'.
    6. If 'RequiredFloors' is empty, all unserved passengers are already at their destination floors (or there are no unserved passengers), so the heuristic is 0 (goal state).
    7. If 'RequiredFloors' is not empty, find the minimum and maximum floor indices among these required floors using the pre-calculated floor mapping.
    8. Calculate the estimated movement cost: This is the minimum distance the lift must travel to cover the range of required floors. It's computed as the distance between the minimum and maximum required floor indices, plus the minimum distance from the current lift floor to either the minimum or maximum required floor index.
    9. Calculate the non-movement cost: This is the sum of 'board' actions needed and 'depart' actions needed.
       - Number of 'board' actions needed is the number of waiting passengers ('|P_wait|').
       - Number of 'depart' actions needed is the number of unserved passengers ('|P_unserved|').
    10. The total heuristic value is the sum of the non-movement cost and the estimated movement cost: `|P_wait| + |P_unserved| + movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        # The set of facts that must hold in goal states.
        # In miconic, this is typically (served p) for all passengers.
        self.goals = task.goals
        static_facts = task.static

        # Parse 'destin' facts to map passengers to destinations
        self.passenger_to_destin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                p, f = parts[1], parts[2]
                self.passenger_to_destin[p] = f

        # Gather all floor names mentioned in the problem (init and static)
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "above":
                    all_floors.add(parts[1])
                    all_floors.add(parts[2])
                elif parts[0] == "destin":
                     # Destination floors are relevant
                     all_floors.add(parts[2])
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "lift-at":
                    all_floors.add(parts[1])
                elif parts[0] == "origin":
                    # Origin floors are relevant
                    all_floors.add(parts[2])

        above_facts_parsed = [(parts[1], parts[2]) for fact in static_facts if match(fact, "above", "*", "*") for parts in [get_parts(fact)]]

        self.floor_to_index = {}
        self.index_to_floor = []

        if not all_floors:
             # No floors mentioned anywhere? Should not happen in valid problems.
             return

        if not above_facts_parsed:
             # Case with one or more floors but no 'above' facts.
             # Assume floors are ordered alphabetically or numerically if possible.
             try:
                 sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
                 for i, f in enumerate(sorted_floors):
                     self.floor_to_index[f] = i
                     self.index_to_floor.append(f)
             except (ValueError, TypeError): # Handle non-integer floor names
                  sorted_floors = sorted(list(all_floors))
                  for i, f in enumerate(sorted_floors):
                     self.floor_to_index[f] = i
                     self.index_to_floor.append(f)

        else:
            # Find the lowest floor: a floor that is never the 'above' floor in any 'above' fact
            upper_floors = {f_above for f_below, f_above in above_facts_parsed}
            candidate_lowest = [f for f in all_floors if f not in upper_floors]

            lowest_floor = None
            if len(candidate_lowest) == 1:
                lowest_floor = candidate_lowest[0]
            elif len(all_floors) == 1: # Single floor case (may or may not have above facts)
                 lowest_floor = list(all_floors)[0]
            else:
                 # This indicates an issue with parsing or domain structure (e.g., disconnected floors)
                 # Fallback: sort alphabetically or numerically
                 try:
                     sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
                     if sorted_floors:
                         lowest_floor = sorted_floors[0]
                 except (ValueError, TypeError): # Handle non-integer floor names
                      sorted_floors = sorted(list(all_floors))
                      if sorted_floors:
                         lowest_floor = sorted_floors[0]


            if lowest_floor:
                current = lowest_floor
                index = 0
                # Build a map for quick lookup of the floor immediately above
                above_map = {f_below: f_above for f_below, f_above in above_facts_parsed}
                # Build the ordered list and map following the 'above' chain
                while current:
                    if current in self.floor_to_index: # Prevent infinite loop from cyclic 'above' facts
                         break
                    self.floor_to_index[current] = index
                    self.index_to_floor.append(current)
                    index += 1
                    current = above_map.get(current)

                # Handle floors mentioned in the problem but not part of the main 'above' chain
                # (e.g., disconnected floors). Assign them indices outside the main range.
                # This is a fallback for potentially malformed problems.
                next_index = len(self.index_to_floor)
                for floor in all_floors:
                    if floor not in self.floor_to_index:
                        self.floor_to_index[floor] = next_index
                        self.index_to_floor.append(floor)
                        next_index += 1


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if it's a goal state first (more robust than relying on RequiredFloors)
        # A state is a goal if all passengers in the problem are served.
        all_passengers_in_problem = set(self.passenger_to_destin.keys())
        served_passengers_in_state = {parts[1] for fact in state for parts in [get_parts(fact)] if parts and parts[0] == "served"}

        if all_passengers_in_problem and all_passengers_in_problem.issubset(served_passengers_in_state):
             return 0 # It's a goal state

        # 1. Find current_floor
        current_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                current_floor = parts[1]
                break

        # If not a goal state and no lift-at, state is likely invalid.
        if current_floor is None:
             # print("Warning: State has no lift-at fact and is not a goal state.")
             return float('inf') # Should not happen in valid problems

        # Get current floor index
        if current_floor not in self.floor_to_index:
             # Should not happen in valid problems if init parsed all floors
             # print(f"Warning: Lift is at unknown floor {current_floor}")
             return float('inf')
        curr_idx = self.floor_to_index[current_floor]


        # 2. Identify P_wait, P_boarded, OriginFloorsWait, DestinFloorsBoarded
        P_wait = set()
        P_boarded = set()
        OriginFloorsWait = set()
        DestinFloorsBoarded = set()

        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "origin":
                    p, f = parts[1], parts[2]
                    P_wait.add(p)
                    OriginFloorsWait.add(f)
                elif parts[0] == "boarded":
                    p = parts[1]
                    P_boarded.add(p)

        # Add destination floors for boarded passengers
        for p in P_boarded:
            if p in self.passenger_to_destin:
                DestinFloorsBoarded.add(self.passenger_to_destin[p])
            # else: Warning handled in init if passenger has no destin

        # 3. Calculate |P_wait| and |P_boarded|
        num_wait = len(P_wait)
        num_boarded = len(P_boarded)
        num_unserved = num_wait + num_boarded # Assuming no served passengers are in P_wait or P_boarded

        # 4. Identify RequiredFloors and their min/max indices
        RequiredFloors = OriginFloorsWait | DestinFloorsBoarded

        if not RequiredFloors:
            # This implies num_unserved is 0, which is the goal state.
            # This case is already covered by the explicit goal check above.
            # Return 0 if it somehow wasn't caught, though it should be.
            return 0

        # Ensure all required floors are in our mapping (should be if init parsed all floors)
        if not all(f in self.floor_to_index for f in RequiredFloors):
             # print("Warning: Required floor not found in floor mapping.")
             return float('inf') # Should not happen in valid problems

        required_indices = [self.floor_to_index[f] for f in RequiredFloors]
        min_req_idx = min(required_indices)
        max_req_idx = max(required_indices)

        # 5. Calculate movement cost
        # Distance from current floor to the nearest required floor endpoint
        dist_to_min = abs(curr_idx - min_req_idx)
        dist_to_max = abs(curr_idx - max_req_idx)
        dist_endpoints = abs(max_req_idx - min_req_idx)

        movement_cost = min(dist_to_min, dist_to_max) + dist_endpoints

        # 6. Compute heuristic value
        # Heuristic = |P_wait| (board actions) + |P_unserved| (depart actions) + movement_cost
        heuristic_value = num_wait + num_unserved + movement_cost

        return heuristic_value
