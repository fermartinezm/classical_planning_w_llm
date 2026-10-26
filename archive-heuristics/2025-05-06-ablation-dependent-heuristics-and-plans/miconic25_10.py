from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # This case should ideally not happen for PDDL facts we process
         # but as a fallback, split might work for simple object names
         return fact_str.split()
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining cost by summing the estimated cost
    for each unserved passenger independently. The cost for a passenger depends
    on whether they are waiting at their origin or already boarded, and includes
    estimated movement cost and the cost of board/depart actions.

    # Assumptions
    - The heuristic calculates the cost for each passenger as if they were the
      only one being transported, summing these individual costs. This
      overestimates the cost when passengers can share lift travel, making
      the heuristic non-admissible but potentially good for greedy search.
    - Floor ordering is strictly linear and defined by the `above` predicate.
    - Passengers' origin and destination floors are different for unserved passengers.

    # Heuristic Initialization
    - Parses static facts (`task.static`) and initial state facts (`task.initial_state`)
      to determine the linear ordering of floors and create a mapping from floor
      names to numerical indices.
    - Parses static facts (`task.static`) to determine the destination floor
      for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Identify all passengers and their states: waiting at origin, boarded, or served.
    3. For each passenger who is *not* served:
       a. Retrieve their destination floor (pre-calculated during initialization).
       b. If the passenger is waiting at their origin floor `f_origin`:
          - Estimate cost as:
            - Distance from the current lift floor to `f_origin` (`abs(current_floor_idx - origin_floor_idx)`).
            - Cost of boarding (1 action).
            - Distance from `f_origin` to their destination `f_destin` (`abs(origin_floor_idx - destin_floor_idx)`).
            - Cost of departing (1 action).
          - Total for waiting passenger: `abs(current_floor_idx - origin_floor_idx) + abs(origin_floor_idx - destin_floor_idx) + 2`.
       c. If the passenger is boarded:
          - Estimate cost as:
            - Distance from the current lift floor to their destination `f_destin` (`abs(current_floor_idx - destin_floor_idx)`).
            - Cost of departing (1 action).
          - Total for boarded passenger: `abs(current_lift_floor_idx - destin_floor_idx) + 1`.
    4. The total heuristic value is the sum of the estimated costs for all unserved passengers.
    5. If all passengers are served, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        super().__init__(task)

        # Combine initial state and static facts to find all relevant information
        all_facts = set(task.initial_state) | set(task.static)

        # 1. Parse floor ordering and create floor_to_index map
        floor_names = set()
        above_map = {} # Maps f_low -> f_high

        # Collect floor names and build above_map from (above f_low f_high) facts
        f_highs_in_above = set()
        f_lows_in_above = set()
        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_low, f_high = parts[1], parts[2]
                floor_names.add(f_low)
                floor_names.add(f_high)
                above_map[f_low] = f_high
                f_lows_in_above.add(f_low)
                f_highs_in_above.add(f_high)
            # Also collect floor names from other relevant facts just in case
            elif parts[0] in ["lift-at", "origin", "destin"]:
                 if len(parts) > 1: # Ensure there's an argument
                     floor_names.add(parts[-1]) # Floor is typically the last argument

        self.ordered_floors = []
        self.floor_to_index = {}

        if not floor_names:
             # Should not happen in valid problems, but handle defensively
             print("Warning: No floors found in problem definition.")
             return

        # Find the bottom floor: a floor that is in floor_names but never a f_high in an (above f_low f_high) fact
        bottom_floor = None
        potential_bottom_floors = floor_names - f_highs_in_above
        if len(potential_bottom_floors) == 1:
             bottom_floor = list(potential_bottom_floors)[0]
        elif len(floor_names) == 1:
             # Single floor case
             bottom_floor = list(floor_names)[0]
        else:
             # Fallback: Could not determine bottom floor uniquely.
             # This happens if there are no 'above' facts (e.g., single floor)
             # or if the 'above' facts don't form a single chain.
             # For robustness, if we can't find a unique bottom, just sort floors alphabetically.
             print("Warning: Could not uniquely determine bottom floor from 'above' facts. Sorting floors alphabetically.")
             self.ordered_floors = sorted(list(floor_names))
             self.floor_to_index = {floor: index for index, floor in enumerate(self.ordered_floors)}
             bottom_floor = None # Indicate we used fallback and won't build from above_map


        if bottom_floor:
            # Build ordered list (bottom to top) by following above_map
            current_floor = bottom_floor
            while current_floor and current_floor in floor_names: # Ensure floor exists
                self.ordered_floors.append(current_floor)
                current_floor = above_map.get(current_floor)
                # Add a safety break for cycles or malformed maps
                if current_floor in self.ordered_floors:
                     print("Warning: Detected cycle or issue in floor ordering.")
                     # Clear partial order and use fallback sorted list
                     self.ordered_floors = sorted(list(floor_names))
                     self.floor_to_index = {floor: index for index, floor in enumerate(self.ordered_floors)}
                     bottom_floor = None # Stop building from map
                     break

            # If we successfully built the ordered list from bottom_floor
            if bottom_floor is not None:
                 self.floor_to_index = {floor: index for index, floor in enumerate(self.ordered_floors)}

        # 2. Parse passenger destinations
        self.passenger_destin = {}
        # Destinations are static
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.passenger_destin[passenger] = floor

        # Collect all passenger names mentioned in initial state (origin, boarded)
        # and goals (served) to ensure we consider all relevant passengers.
        all_passengers_in_problem = set(self.passenger_destin.keys())
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] in ["origin", "boarded", "served"]:
                  if len(parts) > 1:
                       all_passengers_in_problem.add(parts[1])

        # Ensure all passengers found have a destination in self.passenger_destin
        # (as per PDDL structure, they should).
        for p in all_passengers_in_problem:
             if p not in self.passenger_destin:
                  # This case indicates a problem definition issue.
                  print(f"Warning: Passenger {p} found in state/goals but has no destin fact.")
                  # Add a placeholder to avoid errors later, though this problem might be unsolvable
                  self.passenger_destin[p] = None


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Identify current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # If lift location is missing, state is invalid or unsolvable
        if current_lift_floor is None:
             # This should not happen in a valid miconic state reachable from initial state
             print("Error: Lift location not found in state.")
             return float('inf')

        current_lift_floor_idx = self.floor_to_index.get(current_lift_floor)
        # If lift floor is not in our parsed floors, something is wrong
        if current_lift_floor_idx is None:
             print(f"Error: Unknown floor '{current_lift_floor}' found for lift location.")
             return float('inf')


        # Identify passenger states
        served_passengers = set()
        boarded_passengers = set()
        waiting_passengers = {} # passenger -> origin_floor

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "served":
                served_passengers.add(parts[1])
            elif parts[0] == "boarded":
                boarded_passengers.add(parts[1])
            elif parts[0] == "origin":
                passenger, floor = parts[1], parts[2]
                waiting_passengers[passenger] = floor

        total_heuristic = 0

        # Iterate through all passengers we know about
        for passenger in self.passenger_destin.keys():
            if passenger in served_passengers:
                continue # This passenger is already served

            destin_floor = self.passenger_destin.get(passenger)
            if destin_floor is None:
                 # Passenger had no destin fact, warned during init. Skip.
                 continue

            destin_floor_idx = self.floor_to_index.get(destin_floor)
            if destin_floor_idx is None:
                 # Destination floor not parsed, warned during init. Skip or error.
                 print(f"Error: Destination floor '{destin_floor}' for passenger '{passenger}' not found in floor map.")
                 total_heuristic += float('inf') # Indicate unsolvable part
                 continue


            if passenger in boarded_passengers:
                # Passenger is boarded, needs to reach destination and depart
                # Cost = movement from current lift floor to destination + depart action
                movement_cost = abs(current_lift_floor_idx - destin_floor_idx)
                total_heuristic += movement_cost + 1 # 1 for depart action

            elif passenger in waiting_passengers:
                # Passenger is waiting at origin, needs pickup, travel, and dropoff
                origin_floor = waiting_passengers[passenger]
                origin_floor_idx = self.floor_to_index.get(origin_floor)

                if origin_floor_idx is None:
                     print(f"Error: Origin floor '{origin_floor}' for passenger '{passenger}' not found in floor map.")
                     total_heuristic += float('inf') # Indicate unsolvable part
                     continue

                # Cost = movement from current lift floor to origin + board action
                #      + movement from origin to destination + depart action
                movement_to_origin = abs(current_lift_floor_idx - origin_floor_idx)
                movement_origin_to_destin = abs(origin_floor_idx - destin_floor_idx)
                total_heuristic += movement_to_origin + 1 + movement_origin_to_destin + 1 # 1 for board, 1 for depart

            # Else: Passenger is unserved, not boarded, and not waiting at origin.
            # This implies an invalid state according to typical miconic problem generation.
            # We assume this case doesn't contribute to the heuristic in a solvable problem.
            # If it happens, the passenger is effectively stuck and the problem might be unsolvable.
            # We could add a large penalty here if needed, but the current sum approach handles it by adding 0.


        return total_heuristic
