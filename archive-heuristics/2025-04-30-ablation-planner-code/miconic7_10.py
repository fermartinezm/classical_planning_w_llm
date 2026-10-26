from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
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
    This heuristic estimates the remaining effort by summing the number of
    passengers who still need to be boarded, the number of passengers who
    are boarded but not yet served, and an estimate of the minimum lift
    movement required to visit all floors where service (pickup or dropoff)
    is needed.

    # Assumptions
    - The domain uses standard PDDL predicates for miconic: origin, destin,
      above, boarded, served, lift-at.
    - The 'above' predicates define a total order on floors, or floors can
      be ordered alphabetically if no 'above' facts are present.
    - The goal is to have specific passengers 'served'.

    # Heuristic Initialization
    - Extracts the set of passengers that need to be served from the goal conditions.
    - Extracts the destination floor for each passenger from static 'destin' facts.
    - Builds an ordered list of floors and a mapping from floor name to index
      by parsing the static 'above' facts. If no 'above' facts provide a total
      order, floors are sorted alphabetically as a fallback. This mapping
      allows calculating distances between floors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift from the `(lift-at ?f)` fact.
    2. Identify passengers who need to be served (from the goal).
    3. For each goal passenger, determine their current status:
       - Waiting at origin: `(origin p o)` in state.
       - Boarded: `(boarded p)` in state.
       - Served: `(served p)` in state (these are ignored for heuristic calculation).
    4. Count the number of waiting passengers (`N_waiting`). Each needs a 'board' action.
    5. Count the number of boarded passengers (`N_boarded`). Each needs a 'depart' action.
    6. Identify the set of floors that require a visit:
       - Origin floors of all waiting passengers.
       - Destination floors of all boarded passengers.
       Let this set be `Floors_to_visit`.
    7. Estimate the minimum lift movement required to visit all floors in `Floors_to_visit` starting from the current lift floor.
       - If `Floors_to_visit` is empty or contains no valid floors, estimated moves = 0.
       - If `Floors_to_visit` contains valid floors:
         - Find the lowest (`min_f`) and highest (`max_f`) floors among the valid floors in `Floors_to_visit` based on the floor order.
         - Get the indices: `min_idx`, `max_idx`, `current_idx`.
         - Calculate the distance between `min_f` and `max_f` (`dist_min_max = max_idx - min_idx`).
         - Calculate the distance between `current_f` and `min_idx` (`dist_cf_min = abs(current_idx - min_idx)`).
         - Calculate the distance between `current_f` and `max_idx` (`dist_cf_max = abs(current_idx - max_idx)`).
         - The minimum moves to visit all floors in the range [min_f, max_f] starting from `current_f` is estimated as:
           - If `current_idx` is below `min_idx`: `max_idx - current_idx` (go straight up to the highest needed floor).
           - If `current_idx` is above `max_idx`: `current_idx - min_idx` (go straight down to the lowest needed floor).
           - If `current_idx` is within [min_idx, max_idx]: `(max_idx - min_idx) + min(dist_cf_min, dist_cf_max)` (go to one end of the range, then sweep the full range).
         - `estimated_moves` is calculated using the appropriate case.
    8. The total heuristic value is `N_waiting + N_boarded + estimated_moves`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        passenger destinations, and floor ordering.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract passengers that need to be served from the goal
        self.goal_served_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                self.goal_served_passengers.add(parts[1])

        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = floor

        # Build floor ordering and index mapping from static 'above' facts
        above_map = {} # child_floor -> parent_floor (e.g., f2 -> f1 if (above f1 f2))
        all_floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "above":
                    floor1, floor2 = parts[1], parts[2]
                    above_map[floor2] = floor1
                    all_floors.add(floor1)
                    all_floors.add(floor2)
                # Also collect floors mentioned in other relevant predicates to ensure all floors are known
                elif parts[0] in ["origin", "destin", "lift-at"]:
                     for obj in parts[1:]:
                         # Assuming objects in these positions are floors
                         all_floors.add(obj)


        self.floors_by_index = []
        self.floor_indices = {}

        if not all_floors:
             # No floors defined in the problem
             pass # Lists remain empty

        else:
            # Find the bottom floor: a floor f such that no other floor is (above ?any_f f)
            # i.e., f is never the second argument of an 'above' predicate.
            floors_that_are_children = {parts[2] for fact in self.static if get_parts(fact) and get_parts(fact)[0] == "above"}
            bottom_floor_candidates = all_floors - floors_that_are_children

            bottom_floor = None
            if len(bottom_floor_candidates) == 1:
                 bottom_floor = list(bottom_floor_candidates)[0]
            elif len(bottom_floor_candidates) > 1:
                 # Multiple candidates for bottom floor (disconnected towers?), pick one deterministically
                 bottom_floor = sorted(list(bottom_floor_candidates))[0]
                 # print(f"Warning: Multiple bottom floor candidates found. Picking {bottom_floor}")
            # else: len == 0, no bottom floor found (e.g., circularity or no above facts)

            if bottom_floor is not None:
                # Build ordered list starting from bottom floor
                current = bottom_floor
                index = 0
                # Need reverse map: parent_floor -> child_floor
                reverse_above_map = {v: k for k, v in above_map.items()}

                while current is not None:
                    if current in self.floor_indices: # Prevent infinite loop in case of cycles (invalid PDDL)
                         # print(f"Warning: Cycle detected in floor 'above' relations involving {current}")
                         break
                    self.floors_by_index.append(current)
                    self.floor_indices[current] = index
                    index += 1
                    # Find the floor immediately above the current one
                    current = reverse_above_map.get(current)

                # Check if all floors were included in the ordered list
                if len(self.floors_by_index) != len(all_floors):
                     # This can happen if there are multiple disconnected sets of floors
                     # or floors not connected by 'above' facts.
                     # Fallback to alphabetical sort for any remaining floors.
                     remaining_floors = sorted(list(all_floors - set(self.floors_by_index)))
                     start_index = len(self.floors_by_index)
                     self.floors_by_index.extend(remaining_floors)
                     for i, floor in enumerate(remaining_floors):
                         self.floor_indices[floor] = start_index + i
                     # print(f"Warning: Not all floors connected by 'above'. Added remaining floors alphabetically: {remaining_floors}")

            else:
                # No 'above' facts providing a clear bottom floor or ordering.
                # Fallback: Sort all floors alphabetically.
                sorted_floors = sorted(list(all_floors))
                self.floors_by_index = sorted_floors
                self.floor_indices = {floor: i for i, floor in enumerate(sorted_floors)}
                # print(f"Warning: No 'above' facts to order floors. Assuming sorted order: {sorted_floors}")


    def get_floor_index(self, floor_name):
        """Get the integer index for a floor name."""
        return self.floor_indices.get(floor_name, -1) # Return -1 if floor not found

    def distance(self, floor1_name, floor2_name):
        """Calculate the distance (number of levels) between two floors."""
        idx1 = self.get_floor_index(floor1_name)
        idx2 = self.get_floor_index(floor2_name)
        if idx1 == -1 or idx2 == -1:
             # Should not happen with valid PDDL and correct parsing/initialization
             return float('inf') # Indicate impossibility or error
        return abs(idx1 - idx2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings.

        # Check if goal is reached
        all_served = True
        for passenger in self.goal_served_passengers:
            if f"(served {passenger})" not in state:
                all_served = False
                break
        if all_served:
            return 0

        current_f = None
        waiting_passengers_info = {} # {passenger: origin_floor}
        boarded_passengers = [] # [passenger]

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "lift-at":
                current_f = parts[1]
            elif predicate == "origin":
                passenger, floor = parts[1], parts[2]
                if passenger in self.goal_served_passengers:
                    waiting_passengers_info[passenger] = floor
            elif predicate == "boarded":
                passenger = parts[1]
                if passenger in self.goal_served_passengers:
                    boarded_passengers.append(passenger)
            # 'served' facts are handled by the goal check at the beginning.

        # Calculate components of the heuristic
        N_waiting = len(waiting_passengers_info)
        N_boarded = len(boarded_passengers)

        # Identify floors that need service (pickup or dropoff)
        floors_to_visit = set()
        floors_to_visit.update(waiting_passengers_info.values())
        for passenger in boarded_passengers:
             dest_floor = self.passenger_destinations.get(passenger)
             if dest_floor: # Ensure destination exists
                 floors_to_visit.add(dest_floor)

        estimated_moves = 0
        valid_floors_to_visit = {f for f in floors_to_visit if self.get_floor_index(f) != -1}

        if valid_floors_to_visit:
            # Find min and max floor indices among valid floors to visit
            min_idx_to_visit = float('inf')
            max_idx_to_visit = float('-inf')
            for floor in valid_floors_to_visit:
                idx = self.get_floor_index(floor)
                min_idx_to_visit = min(min_idx_to_visit, idx)
                max_idx_to_visit = max(max_idx_to_visit, idx)

            current_idx = self.get_floor_index(current_f)

            if current_idx != -1: # Ensure current floor is valid
                # Estimate moves based on current position and range of floors to visit
                if current_idx < min_idx_to_visit:
                    # Must go up at least to the highest required floor
                    estimated_moves = max_idx_to_visit - current_idx
                elif current_idx > max_idx_to_visit:
                    # Must go down at least to the lowest required floor
                    estimated_moves = current_idx - min_idx_to_visit
                else: # current_idx is within [min_idx_to_visit, max_idx_to_visit]
                    # Must cover the full range [min_idx, max_idx] plus travel from current to one end.
                    # Minimum moves to visit min_idx and max_idx starting from current_idx
                    dist_min_max = max_idx_to_visit - min_idx_to_visit
                    dist_cf_min = abs(current_idx - min_idx_to_visit)
                    dist_cf_max = abs(current_idx - max_idx_to_visit)
                    estimated_moves = min(dist_cf_min + dist_min_max, dist_cf_max + dist_min_max)
            # else: current_f is invalid, estimated_moves remains 0.


        # Total heuristic is sum of actions (board/depart) and estimated moves
        # Each waiting passenger needs 1 board action.
        # Each boarded passenger needs 1 depart action.
        # The estimated_moves covers the vertical travel needed to reach service floors.
        h = N_waiting + N_boarded + estimated_moves

        return h
