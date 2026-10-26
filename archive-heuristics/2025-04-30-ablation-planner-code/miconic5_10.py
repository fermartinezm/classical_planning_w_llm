from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # Assuming task object also contains the initial state for convenience
        # to get all passenger names, although static facts (destin) are sufficient.
        # self.init = task.init

    def __call__(self, node):
        raise NotImplementedError

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining cost by summing up an estimated cost
    for each unserved passenger independently. The estimated cost for a passenger
    considers the actions (board/depart) and the required lift movement segments
    to get that specific passenger from their current location (origin or boarded)
    to their destination. This heuristic is non-admissible as it sums costs
    independently for each passenger, overcounting shared lift movements and actions.

    # Assumptions
    - Floors are totally ordered by the `above` predicate, forming a single chain.
    - All passengers must be served to reach the goal.
    - Each action (board, depart, up, down) has a cost of 1.
    - Passengers not mentioned in `destin` predicates in static facts are not part of the goal.
    - Passengers not in `origin`, `boarded`, or `served` predicates in the state
      are in an invalid state or not relevant to the current subproblem.

    # Heuristic Initialization
    - Extracts the floor ordering from `above` predicates to create a mapping
      from floor names to integer levels.
    - Extracts the destination floor for each passenger from `destin` predicates
      in static facts. The set of passengers is defined by those appearing in `destin`.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Current Lift Location:** Find the floor where the lift is currently located using the `(lift-at ?f)` predicate in the state.
    2. **Map Floors to Levels:** Use the precomputed mapping (`self.floor_levels`) to get the integer level for the current lift floor. If the floor is not in the mapping, the heuristic cannot be computed reliably (return infinity).
    3. **Pre-process State:** Create quick lookup structures (dictionaries/sets) for `origin`, `boarded`, and `served` predicates in the current state.
    4. **Iterate Through Passengers:** For each passenger identified during initialization (those with a destination defined in static facts):
       - **Check if Served:** If the passenger is in the set of served passengers from the state, their contribution to the heuristic is 0.
       - **If Not Served:**
         - Find the passenger's destination floor using the precomputed mapping (`self.passenger_destinations`). Map the destination floor to its corresponding level. If the destination floor is not in the floor levels mapping, the heuristic cannot be computed reliably (return infinity).
         - **Check if Boarded:** If the passenger is in the set of boarded passengers from the state:
           - This passenger needs to be dropped off.
           - Estimated cost for this passenger = 1 (for the `depart` action) + the absolute difference in floor levels between the current lift location and the passenger's destination floor (estimated movement cost for this passenger's dropoff trip).
         - **If Not Boarded (must be waiting at origin):**
           - Find the passenger's origin floor from the state's origin mapping. Map the origin floor to its corresponding level. If the origin floor is not in the floor levels mapping, the heuristic cannot be computed reliably (return infinity).
           - This passenger needs to be picked up and dropped off.
           - Estimated cost for this passenger = 2 (for `board` and `depart` actions) + the absolute difference in floor levels between the current lift location and the passenger's origin floor (estimated movement cost to pick up) + the absolute difference in floor levels between the passenger's origin floor and their destination floor (estimated movement cost to drop off).
         - **Handle Invalid State:** If an unserved passenger is neither waiting nor boarded (and not served), this indicates an unexpected or invalid state. This passenger cannot be served. Assign a large penalty (infinity) as the problem is likely unsolvable from this state.
    5. **Sum Contributions:** The total heuristic value is the sum of the estimated costs for all unserved passengers.
    6. **Goal State:** If the sum is 0, it implies no unserved passengers were found, meaning the state is a goal state. The heuristic correctly returns 0. If the sum is non-zero (or infinity), it represents the estimated remaining cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Extract floor ordering and create level mapping
        self.floor_levels = self._get_floor_levels(self.static)

        # 2. Extract passenger destinations and collect all passenger names
        self.passenger_destinations = {}
        self.all_passengers = set()
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destinations[passenger] = floor
                self.all_passengers.add(passenger)

        # Note: This heuristic considers only passengers listed in 'destin' facts.
        # Passengers existing in the state but without a 'destin' fact are ignored.
        # Passengers in 'destin' but not in the state (origin/boarded/served)
        # will trigger the "Invalid State" check in __call__.


    def _get_floor_levels(self, static_facts):
        """
        Determines the order of floors based on 'above' predicates and assigns
        an integer level to each floor. Assumes 'above' predicates define a
        single total order for all relevant floors.
        Returns a dictionary mapping floor name strings to integer levels (1-based).
        Returns an empty dictionary if floor order cannot be determined.
        """
        above_relations = [] # Stores (lower_floor, higher_floor) tuples
        all_floors_in_above = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_high, f_low = get_parts(fact)
                above_relations.append((f_low, f_high))
                all_floors_in_above.add(f_low)
                all_floors_in_above.add(f_high)

        if not all_floors_in_above:
             # No 'above' facts found. Cannot determine order.
             # This might happen in trivial problems or if floors are defined differently.
             # If there's only one floor object, this is fine. If multiple, heuristic will be bad.
             # We cannot reliably assign levels without 'above'.
             # print("Warning: No 'above' relations found. Cannot determine floor order.")
             return {} # Return empty mapping

        # Find the lowest floor: a floor that is a 'lower' floor but never a 'higher' floor
        higher_arg_floors = {f_high for f_low, f_high in above_relations}
        lower_arg_floors = {f_low for f_low, f_high in above_relations}
        potential_lowest_floors = lower_arg_floors - higher_arg_floors

        lowest_floor = None
        if potential_lowest_floors:
             # Assuming a single chain, there should be exactly one such floor.
             lowest_floor = list(potential_lowest_floors)[0]
        else:
             # This case might occur if the lowest floor is only ever the second argument
             # but is also the first argument in a different relation (e.g., f1 < f2, f0 < f1).
             # Or if the graph isn't a simple chain.
             # Let's find any floor that is only ever the second argument.
             only_lower_arg = lower_arg_floors - higher_arg_floors
             if only_lower_arg:
                  lowest_floor = list(only_lower_arg)[0]
             else:
                  # Fallback: If we still can't find a unique lowest floor,
                  # the 'above' facts might not form a simple chain covering all floors.
                  # Or it's a single floor case already handled.
                  # Or the PDDL is structured unexpectedly.
                  print("Warning: Could not definitively find the lowest floor from 'above' relations.")
                  # Cannot build ordered chain reliably.
                  return {}


        # Build the ordered list of floors starting from the lowest
        ordered_floors = [lowest_floor]
        current_floor = lowest_floor
        while True:
            next_floor = None
            # Find the floor directly above the current floor
            for f_low, f_high in above_relations:
                if f_low == current_floor:
                    next_floor = f_high
                    break # Assuming only one floor directly above

            if next_floor and next_floor not in ordered_floors:
                ordered_floors.append(next_floor)
                current_floor = next_floor
            else:
                break # Reached the top or a discontinuity

        # Create the floor_levels mapping
        floor_levels = {floor: i + 1 for i, floor in enumerate(ordered_floors)}

        # Optional: Check if all floors mentioned in 'above' are in the mapping
        if len(floor_levels) != len(all_floors_in_above):
             print(f"Warning: Not all floors mentioned in 'above' facts ({all_floors_in_above}) are included in the derived ordered chain ({ordered_floors}). Heuristic might be inaccurate.")
             # The mapping only contains floors in the main chain found.

        return floor_levels


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Check if floor levels mapping was successfully created
        if not self.floor_levels and self.all_passengers:
             # Cannot compute heuristic involving movement without floor order,
             # unless there are no passengers needing transport.
             # If there are passengers, but no floor levels, it's likely unsolvable or misconfigured.
             print("Error: Floor levels mapping is empty. Cannot compute heuristic for problem with passengers.")
             return float('inf')

        # Find current lift location
        current_lift_f = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_f = get_parts(fact)
                break

        if current_lift_f is None:
             # This should not happen in a valid miconic state
             print("Error: Lift location not found in state.")
             return float('inf')

        current_level = self.floor_levels.get(current_lift_f)
        if current_level is None and self.floor_levels:
             # Lift is at a floor not in our mapping (not in 'above' chain)
             print(f"Warning: Lift at floor {current_lift_f} not found in floor levels mapping.")
             return float('inf') # Cannot compute distance reliably
        elif current_level is None and not self.floor_levels:
             # No floor levels mapping exists (e.g., no 'above' facts).
             # Assume all floors are the same level or movement cost is 0.
             # This is a fallback for simple cases, might be inaccurate.
             current_level = 0 # Assign a dummy level if no mapping


        total_heuristic = 0
        unserved_passengers_found = False # Flag to check if any unserved passenger exists

        # Pre-process state for quick lookups
        state_origins = {}
        state_boarded = set()
        state_served = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                state_origins[parts[1]] = parts[2]
            elif parts[0] == "boarded":
                state_boarded.add(parts[1])
            elif parts[0] == "served":
                state_served.add(parts[1])


        # Calculate cost for each unserved passenger
        for passenger in self.all_passengers:
            if passenger in state_served:
                continue # Passenger is already served

            unserved_passengers_found = True

            dest_f = self.passenger_destinations.get(passenger)
            if dest_f is None:
                 # Should not happen based on how self.all_passengers is populated
                 print(f"Error: Destination not found for passenger {passenger}.")
                 total_heuristic += float('inf')
                 continue

            dest_level = self.floor_levels.get(dest_f)
            if dest_level is None and self.floor_levels:
                 print(f"Warning: Destination floor {dest_f} for passenger {passenger} not found in floor levels mapping.")
                 total_heuristic += float('inf') # Cannot compute distance
                 continue
            elif dest_level is None and not self.floor_levels:
                 # No floor levels mapping exists. Assume 0 distance for movement.
                 dest_level = 0 # Assign a dummy level if no mapping


            if passenger in state_boarded:
                # Passenger is boarded, needs to depart at destination
                # Cost = 1 (depart) + movement from current lift floor to destination floor
                move_cost = abs(current_level - dest_level)
                total_heuristic += 1 + move_cost
            elif passenger in state_origins:
                # Passenger is waiting at origin, needs to board and depart
                orig_f = state_origins[passenger]
                orig_level = self.floor_levels.get(orig_f)

                if orig_level is None and self.floor_levels:
                     print(f"Warning: Origin floor {orig_f} for passenger {passenger} not found in floor levels mapping.")
                     total_heuristic += float('inf') # Cannot compute distance
                     continue
                elif orig_level is None and not self.floor_levels:
                     # No floor levels mapping exists. Assume 0 distance for movement.
                     orig_level = 0 # Assign a dummy level if no mapping


                # Cost = 2 (board + depart) + movement from current lift floor to origin floor
                # + movement from origin floor to destination floor
                move_cost_to_orig = abs(current_level - orig_level)
                move_cost_orig_to_dest = abs(orig_level - dest_level)
                total_heuristic += 2 + move_cost_to_orig + move_cost_orig_to_dest
            else:
                # Unserved passenger is neither waiting nor boarded. Invalid state.
                # This passenger cannot be served from this state.
                print(f"Error: Unserved passenger {passenger} is neither waiting nor boarded.")
                return float('inf') # Problem is unsolvable from this state

        # If no unserved passengers were found, total_heuristic will be 0.
        # If unserved passengers were found but total_heuristic is 0 (e.g., due to inf),
        # the check above would have returned inf.
        # So, if we reach here and total_heuristic is 0, it means unserved_passengers_found was False.
        # If unserved_passengers_found is True but total_heuristic is 0, something is wrong.
        # The calculation should always yield > 0 for unserved passengers unless floors are the same.
        # E.g., waiting at f1, destin f1, lift at f1: 2 + abs(1-1) + abs(1-1) = 2.
        # Boarded, destin f1, lift at f1: 1 + abs(1-1) = 1.
        # So, total_heuristic > 0 if unserved_passengers_found is True, unless infinity was added.

        return total_heuristic

