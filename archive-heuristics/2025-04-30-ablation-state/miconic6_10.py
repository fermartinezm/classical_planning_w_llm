from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

# Helper functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Return empty list for malformed facts, heuristic should handle this
         return []
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It calculates the cost for each unserved passenger independently
    as the sum of:
    1. Lift movement cost from the current floor to the passenger's origin floor (if waiting).
    2. Boarding action cost (1) (if waiting).
    3. Lift movement cost from the origin floor to the destination floor.
    4. Departing action cost (1).
    If the passenger is already boarded, steps 1 and 2 are skipped.
    The total heuristic value is the sum of these individual passenger costs.
    Lift movement cost between floors is the absolute difference in their indices.
    This heuristic is non-admissible as it double-counts lift movements if
    multiple passengers require the lift to visit the same floor.

    # Assumptions
    - All floors are totally ordered. The `above` predicates define this order,
      where `(above f_higher f_lower)` means `f_higher` is directly above `f_lower`.
      A fallback numerical/alphabetical sort is used if the `above` facts don't
      form a clear linear chain.
    - Each action (board, depart, up, down) has a cost of 1.
    - The lift has unlimited capacity.
    - Passengers do not change their origin or destination.
    - Unserved passengers are either waiting at their origin or are boarded.

    # Heuristic Initialization
    - Determine the total order of floors based on the `above` predicates.
    - Create a mapping from floor names to their numerical index in the order.
    - Store the origin and destination floors for each passenger based on static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state.
    2. Get the numerical index for the current lift floor using the pre-calculated floor mapping.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through all passengers whose origin and destination are known (from static facts).
    5. For each passenger, check if the fact `(served passenger)` is present in the current state. If it is, this passenger is served, and their contribution to the heuristic is 0; continue to the next passenger.
    6. If the passenger is not served, determine their current status by checking the state:
       - Is the fact `(origin passenger origin_floor)` present? (Passenger is waiting).
       - Is the fact `(boarded passenger)` present? (Passenger in the lift).
    7. Calculate the estimated cost for this unserved passenger:
       - If the passenger is waiting at their origin floor `origin_f`:
         - Get the numerical indices for `origin_f` and `destin_f`.
         - The cost is the absolute difference between the current lift index and the origin index (move to origin) + 1 (board action) + the absolute difference between the origin index and the destination index (move to destination) + 1 (depart action).
       - If the passenger is boarded:
         - Get the numerical index for `destin_f`.
         - The cost is the absolute difference between the current lift index and the destination index (move to destination) + 1 (depart action).
    8. Add the calculated cost for the current passenger to the total heuristic cost.
    9. After iterating through all passengers, return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger info.
        """
        self.goals = task.goals # Goal conditions (used to check if a passenger is served)
        static_facts = task.static # Facts that are not affected by actions.

        # 1. Determine floor order and create floor_to_index map
        # Build a map: floor_higher -> floor_lower based on (above f_higher f_lower)
        floor_directly_below = {}
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_higher, f_lower = parts[1], parts[2]
                floor_directly_below[f_higher] = f_lower
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        ordered_floors = []
        # Find the lowest floor: a floor that is a value in floor_directly_below but not a key.
        # This floor has a floor above it, but nothing below it in the direct relation.
        potential_lowest = set(floor_directly_below.values()) - set(floor_directly_below.keys())

        if len(potential_lowest) == 1:
            lowest_floor = potential_lowest.pop()
            # Build ordered list by traversing up
            ordered_floors.append(lowest_floor)
            current = lowest_floor
            # Find the floor above 'current' by searching in floor_directly_below values
            while True:
                next_floor = None
                # Iterate through the map to find the floor directly above 'current'
                for f_higher, f_lower in floor_directly_below.items():
                    if f_lower == current:
                        next_floor = f_higher
                        break # Found the next floor up
                if next_floor is None:
                    break # Reached the top floor
                ordered_floors.append(next_floor)
                current = next_floor
        else:
             # Fallback: If the above facts don't define a clear single linear chain
             # or if the naming convention is reliable (f1, f2, ...), sort by number.
             # Ensure all_floors is not empty before sorting
             if all_floors:
                 try:
                     # Attempt numerical sort based on f<number> format
                     ordered_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
                 except (ValueError, IndexError):
                     # If sorting by number fails, just sort alphabetically
                     ordered_floors = sorted(list(all_floors))


        self.floor_to_index = {f: i for i, f in enumerate(ordered_floors)}


        # 2. Store passenger origin and destination floors
        self.passenger_info = {}
        # Need to find all passengers and their origin/destin facts
        passenger_origins = {}
        passenger_destins = {}

        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "origin" and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 passenger_origins[p] = f
             elif parts and parts[0] == "destin" and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 passenger_destins[p] = f

        # Combine origins and destinations for passengers who have both defined
        for p in passenger_origins:
            if p in passenger_destins:
                 self.passenger_info[p] = {'origin': passenger_origins[p], 'destin': passenger_destins[p]}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Check if goal is reached (all goals are served predicates)
        all_served = True
        for passenger in self.passenger_info:
             if f"(served {passenger})" not in state:
                  all_served = False
                  break
        if all_served:
             return 0

        # 1. Find current lift floor
        current_lift_f = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_f = parts[1]
                break

        if current_lift_f is None or current_lift_f not in self.floor_to_index:
             # This indicates an unexpected state (lift not at a known floor)
             # or a problem definition issue. Return infinity as it's likely unsolvable
             # from this state under normal circumstances.
             return float('inf')

        current_lift_idx = self.floor_to_index[current_lift_f]

        total_cost = 0  # Initialize action cost counter.

        # 2. Iterate through passengers and calculate cost if not served
        for passenger, info in self.passenger_info.items():
            # Check if passenger is served
            if f"(served {passenger})" in state:
                continue # Passenger is already served, cost is 0 for this passenger

            origin_f = info['origin']
            destin_f = info['destin']

            # Ensure origin and destination floors are known
            if origin_f not in self.floor_to_index or destin_f not in self.floor_to_index:
                 # This indicates a problem definition issue (passenger origin/destin on unknown floor)
                 # Return infinity as this passenger cannot be served.
                 return float('inf')


            origin_idx = self.floor_to_index[origin_f]
            destin_idx = self.floor_to_index[destin_f]

            # Check passenger state: waiting at origin or boarded
            is_waiting = f"(origin {passenger} {origin_f})" in state
            is_boarded = f"(boarded {passenger})" in state

            # Calculate cost based on state
            if is_waiting:
                # Cost to go from current lift floor to origin floor
                move_to_origin_cost = abs(current_lift_idx - origin_idx)
                # Cost to board
                board_cost = 1
                # Cost to go from origin floor to destination floor
                move_to_destin_cost = abs(origin_idx - destin_idx)
                # Cost to depart
                depart_cost = 1
                cost_for_passenger = move_to_origin_cost + board_cost + move_to_destin_cost + depart_cost
                total_cost += cost_for_passenger

            elif is_boarded:
                # Cost to go from current lift floor to destination floor
                move_to_destin_cost = abs(current_lift_idx - destin_idx)
                # Cost to depart
                depart_cost = 1
                cost_for_passenger = move_to_destin_cost + depart_cost
                total_cost += cost_for_passenger

            # If neither waiting nor boarded, and not served, this is an invalid state
            # for this heuristic's assumptions. It's safer to return infinity.
            # This case shouldn't happen in valid states if the problem is well-formed
            # and actions maintain the invariant that a passenger is either at origin, boarded, or served.
            elif f"(served {passenger})" not in state:
                 # Passenger is unserved but not waiting or boarded. Invalid state.
                 return float('inf')


        return total_cost
