from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match a PDDL fact against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of args for a match
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the number of 'board' and 'depart' actions required for unserved
    passengers and adds an estimate of the necessary lift movement actions.

    # Assumptions
    - Floors are linearly ordered, defined by `(above f_lower f_upper)` facts,
      where `(above f_lower f_upper)` implies `f_upper` is immediately above `f_lower`
      if there is no intermediate floor `f_mid` such that `(above f_lower f_mid)`
      and `(above f_mid f_upper)` are also true.
    - A passenger is served when they are dropped off at their destination floor.
    - The lift can carry multiple passengers.
    - The cost of any action (move, board, depart) is 1.

    # Heuristic Initialization
    - Extracts the linear order of floors and creates a mapping from floor name to index.
    - Stores the destination floor for each passenger from the static facts.
    - Stores the set of all passengers who need to be served (from goal facts).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all unserved passengers. These are passengers listed in the goal
       who do not have the `(served p)` fact in the current state.
    2. For each unserved passenger:
       - Determine their current state: waiting at an origin floor `(origin p f)`
         or boarded in the lift `(boarded p)`.
       - Retrieve their destination floor `(destin p f_destin)` from the pre-computed map.
    3. Count the number of 'board' actions needed: This is the number of unserved
       passengers who are currently waiting at their origin floor.
    4. Count the number of 'depart' actions needed: This is the total number of
       unserved passengers (each unserved passenger will eventually need one 'depart' action).
    5. Identify the set of floors the lift *must* visit to make progress:
       - Origin floors of waiting unserved passengers.
       - Destination floors of boarded unserved passengers.
    6. Estimate the movement cost:
       - If there are no required floors, the movement cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the required floors.
       - Calculate the range span: `max_idx - min_idx`.
       - Calculate the distance from the current lift floor to the closest end of this range:
         `min(abs(current_idx - min_idx), abs(current_idx - max_idx))`.
       - The estimated movement cost is the range span plus the distance to reach the range.
         This estimates the vertical travel needed to cover the range of required floors,
         starting from the current floor.
    7. The total heuristic value is the sum of the required 'board' actions,
       required 'depart' actions, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, floor map,
        passenger destinations, and the set of passengers to be served.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract floor order and create floor map
        self.floor_map = self._build_floor_map(static_facts)

        # 2. Store destination floor for each passenger
        self.passenger_destin = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destin[passenger] = floor

        # 3. Store the set of all passengers who need to be served
        self.passengers_to_serve = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                _, passenger = get_parts(goal)
                self.passengers_to_serve.add(passenger)

    def _build_floor_map(self, static_facts):
        """
        Builds a map from floor name to its index based on 'above' facts.
        Assumes 'above' facts define a linear order and identifies immediate adjacency.
        """
        all_floors = set()
        above_relations = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (above f1 f2)
                    _, f_lower, f_upper = parts
                    above_relations.add((f_lower, f_upper))
                    all_floors.add(f_lower)
                    all_floors.add(f_upper)

        if not all_floors:
             return {} # Handle case with no floors

        # Find immediate 'above' relations: (f1, f2) is immediate if (above f1 f2) is true
        # and there is no f_mid such that (above f1 f_mid) and (above f_mid f2) are true.
        immediate_above = {}
        for f1 in all_floors:
            potential_next_floors = {f2 for f_lower, f2 in above_relations if f_lower == f1}
            for f2_candidate in potential_next_floors:
                is_intermediate_present = False
                for f_mid in all_floors:
                    if f_mid != f1 and f_mid != f2_candidate and \
                       (f1, f_mid) in above_relations and (f_mid, f2_candidate) in above_relations:
                        is_intermediate_present = True
                        break
                if not is_intermediate_present:
                    # Found the floor immediately above f1
                    immediate_above[f1] = f2_candidate
                    # Assuming a linear structure, there should be only one immediate_above for each floor
                    break # Move to the next f1

        # Find the lowest floor: a floor that is not the upper floor in any immediate relation.
        upper_floors_in_immediate_rel = set(immediate_above.values())
        potential_lowest = [f for f in all_floors if f not in upper_floors_in_immediate_rel]

        lowest_floor = None
        if len(potential_lowest) == 1:
             lowest_floor = potential_lowest[0]
        elif len(all_floors) == 1:
             lowest_floor = list(all_floors)[0]
        else:
             # Fallback: If immediate relations don't form a clear chain start,
             # try finding the floor that is not an upper floor in *any* above relation.
             all_upper_floors = {f_upper for _, f_upper in above_relations}
             potential_lowest = [f for f in all_floors if f not in all_upper_floors]
             if len(potential_lowest) == 1:
                  lowest_floor = potential_lowest[0]
             else:
                  # If still ambiguous, sort alphabetically as a last resort.
                  # This is a guess but works for standard f1, f2, ... naming.
                  print("Warning: Could not determine unique lowest floor from 'above' facts. Sorting alphabetically.")
                  sorted_floors = sorted(list(all_floors))
                  return {f: i for i, f in enumerate(sorted_floors)}


        if lowest_floor is None and len(all_floors) > 0:
             # Should have found a lowest floor if all_floors > 0 and logic is correct for linear.
             # This might indicate a non-linear structure or error in facts.
             print("Error: Failed to determine lowest floor from 'above' facts.")
             return {} # Return empty map

        if lowest_floor is None and len(all_floors) == 0:
             return {} # No floors found

        # Build the ordered list starting from the lowest floor
        floor_map = {}
        current_floor = lowest_floor
        index = 0
        visited_floors = set()
        while current_floor is not None and current_floor not in visited_floors:
            floor_map[current_floor] = index
            visited_floors.add(current_floor)
            index += 1
            current_floor = immediate_above.get(current_floor)

        # Check if all floors were included (might miss floors not in the main chain)
        if len(floor_map) != len(all_floors):
             print("Warning: Not all floors included in the linear order derived from 'above' facts.")
             # Continue with the partial map, assuming required floors are in the main chain.

        return floor_map


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Get current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2: # Ensure it's (lift-at f)
                    _, current_lift_floor = parts
                    break

        if current_lift_floor is None or current_lift_floor not in self.floor_map:
             # Should not happen in a valid miconic state with a correctly built map
             print(f"Error: Lift location {current_lift_floor} not found in floor map.")
             return float('inf') # Cannot make progress

        current_lift_idx = self.floor_map[current_lift_floor]

        # Identify unserved passengers and their state/required floors
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.passengers_to_serve - served_passengers

        if not unserved_passengers:
             # Should be caught by the goal check above, but double check
             return 0

        pickup_floors = set()
        dropoff_floors = set()
        num_waiting = 0
        num_boarded = 0

        for passenger in unserved_passengers:
            destin_floor = self.passenger_destin.get(passenger)

            if destin_floor is None or destin_floor not in self.floor_map:
                 # Destination not found or not in floor map - problem definition issue?
                 print(f"Warning: Destination floor for passenger {passenger} not found or not in floor map.")
                 return float('inf') # Treat as unsolvable

            is_waiting = False
            is_boarded = False

            # Find passenger's current state (origin or boarded)
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    parts = get_parts(fact)
                    if len(parts) == 3: # Ensure it's (origin p f)
                        _, _, origin_floor = parts
                        if origin_floor not in self.floor_map:
                             print(f"Warning: Origin floor {origin_floor} for passenger {passenger} not in floor map.")
                             return float('inf') # Treat as unsolvable
                        is_waiting = True
                        pickup_floors.add(origin_floor)
                        num_waiting += 1
                        break # Found origin, no need to check for boarded
                if match(fact, "boarded", passenger):
                    parts = get_parts(fact)
                    if len(parts) == 2: # Ensure it's (boarded p)
                        is_boarded = True
                        num_boarded += 1
                        break # Found boarded

            if is_waiting:
                 # Waiting passengers need to be dropped off at their destination
                 dropoff_floors.add(destin_floor)
            elif is_boarded:
                 # Boarded passengers need to be dropped off at their destination
                 dropoff_floors.add(destin_floor)
            else:
                 # Unserved passenger is neither waiting nor boarded.
                 # This implies they are not at their origin and not in the lift.
                 # This state should not be reachable in a valid miconic problem
                 # unless they were dropped off at the wrong floor, which is not an action effect.
                 # Treat as unsolvable.
                 print(f"Warning: Unserved passenger {passenger} is neither waiting nor boarded.")
                 return float('inf')


        # Heuristic components:
        # 1. Board actions needed: 1 for each waiting passenger.
        # 2. Depart actions needed: 1 for each unserved passenger (waiting or boarded).
        # 3. Movement actions needed.

        # Count board/depart actions
        board_actions = num_waiting
        depart_actions = num_waiting + num_boarded # Total unserved passengers

        # Calculate movement cost
        all_required_floors = pickup_floors.union(dropoff_floors)

        movement_cost = 0
        if all_required_floors:
            required_indices = [self.floor_map[f] for f in all_required_floors]
            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)

            # Estimate moves to cover the range [min_req_idx, max_req_idx] starting from current_lift_idx
            range_span = max_req_idx - min_req_idx
            dist_to_min = abs(current_lift_idx - min_req_idx)
            dist_to_max = abs(current_lift_idx - max_req_idx)

            # Movement cost is the span of the required floors plus the distance
            # from the current floor to the closest end of that span.
            movement_cost = range_span + min(dist_to_min, dist_to_max)

        # Total heuristic value
        # Sum of required board actions, required depart actions, and estimated moves.
        total_heuristic = board_actions + depart_actions + movement_cost

        return total_heuristic
