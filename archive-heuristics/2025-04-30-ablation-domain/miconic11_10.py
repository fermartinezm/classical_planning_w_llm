from fnmatch import fnmatch
# Assuming heuristic_base is available in the execution environment
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# The match function is not strictly needed in the refactored __call__
# but keeping it as it was in the example and might be useful elsewhere.
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
    This heuristic estimates the remaining effort to serve all passengers.
    It sums the number of "event" actions needed per unserved passenger
    (board and/or depart) and an estimate of the lift movement actions required
    to visit all necessary floors.

    # Assumptions
    - Each unserved passenger is either waiting at their origin or boarded.
    - The cost of boarding is 1 action.
    - The cost of departing is 1 action.
    - The cost of moving the lift between adjacent floors is 1 action.
    - The estimated lift movement cost is the maximum distance from the current
      floor to the lowest or highest floor that needs to be visited.

    # Heuristic Initialization
    - Extracts passenger destinations from static facts.
    - Determines the strict ordering of floors based on `(above f_higher f_lower)`
      static facts and creates a mapping from floor name to a numerical index.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers and their destinations from static facts.
    2. Determine the numerical index for each floor based on the `above` relation.
       The floor with the most floors below it is the highest, assigned the largest index.
       The floor with no floors below it is the lowest, assigned index 0.
    3. For a given state:
       a. Identify which passengers are already served, which are waiting at an origin, and which are boarded. Also find the lift's current floor.
       b. Initialize the heuristic value to 0.
       c. Initialize sets for floors that need pickup (`F_pickup`) and floors that need dropoff (`F_dropoff`).
       d. Iterate through all passengers who are *not* served:
          - If the passenger is waiting at their origin floor `f_origin`:
            - Add 2 to the heuristic (for the future board and depart actions).
            - Add `f_origin` to `F_pickup`.
          - If the passenger is boarded:
            - Add 1 to the heuristic (for the future depart action).
            - Find their destination floor `f_destin` (from initialized destinations).
            - Add `f_destin` to `F_dropoff`.
       e. Combine `F_pickup` and `F_dropoff` into `F_needed`.
       f. If `F_needed` is not empty:
          - Find the numerical index for the current floor (`f_current_idx`).
          - Find the minimum and maximum numerical indices among floors in `F_needed` (`min_needed_idx`, `max_needed_idx`).
          - Estimate lift movement cost as `max(abs(f_current_idx - min_needed_idx), abs(f_current_idx - max_needed_idx))`.
          - Add this estimated movement cost to the heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and floor ordering.
        """
        self.goals = task.goals # Goal conditions are used to identify all passengers
        static_facts = task.static # Facts that are not affected by actions.

        # 1. Extract passenger destinations
        self.destinations = {}
        # We can get all passengers from the goal facts (served ?p) or destin facts
        # Let's use destin facts as they provide the destination directly
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.destinations[p] = f

        # Get the list of all passengers from destinations
        self.all_passengers = list(self.destinations.keys())

        # 2. Determine floor ordering and create floor_to_index map
        # Collect all floors mentioned in above facts
        all_floors = set()
        above_relations = []
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_higher, f_lower = parts[1], parts[2]
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                above_relations.append((f_higher, f_lower))

        # Count how many floors are strictly below each floor
        floors_below_count = {}
        for f in all_floors:
            count = 0
            for rel_f_higher, rel_f_lower in above_relations:
                if rel_f_higher == f: # f is above rel_f_lower
                    count += 1
            floors_below_count[f] = count

        # Sort floors by the count of floors below them, descending (highest count = highest floor)
        # This gives the order from highest floor to lowest floor
        sorted_floors_desc = sorted(all_floors, key=lambda f: floors_below_count.get(f, 0), reverse=True)

        # Create the floor_to_index map (lowest floor gets index 0)
        # Reverse the sorted list to get order from lowest to highest
        self.floor_to_index = {f: i for i, f in enumerate(reversed(sorted_floors_desc))}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        heuristic_value = 0

        # --- Step 3a: Identify served/waiting/boarded passengers and current lift floor ---
        served_passengers = set()
        current_lift_floor = None
        waiting_passengers_info = {} # {passenger: origin_floor}
        boarded_passengers = set() # {passenger}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "served" and len(parts) == 2:
                served_passengers.add(parts[1])
            elif predicate == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
            elif predicate == "origin" and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 waiting_passengers_info[p] = f
            elif predicate == "boarded" and len(parts) == 2:
                 p = parts[1]
                 boarded_passengers.add(p)

        # --- Step 3d: Calculate passenger-specific costs and identify needed floors ---
        F_pickup = set()
        F_dropoff = set()

        for p in self.all_passengers:
            if p not in served_passengers:
                if p in waiting_passengers_info:
                    # Passenger needs board + depart actions
                    heuristic_value += 2
                    F_pickup.add(waiting_passengers_info[p])
                elif p in boarded_passengers:
                    # Passenger needs depart action
                    heuristic_value += 1
                    # Add destination floor to dropoff locations
                    # Destination must exist if passenger is in self.all_passengers
                    F_dropoff.add(self.destinations[p])
                # else: Passenger is not served, not waiting, not boarded? (Shouldn't happen in valid states)


        # --- Step 3e, f, g: Estimate lift movement cost ---
        F_needed = F_pickup | F_dropoff

        # Only calculate movement cost if there are floors that need visiting
        if F_needed:
            # Ensure the current lift floor is known and is a valid floor
            # If not, it's an invalid state, return a high value? Or assume valid?
            # Assuming valid states for now.
            if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
                 # This case should ideally not be reached with valid problem generation
                 # and state transitions. Return a large value to penalize such states.
                 # Returning float('inf') might cause issues with integer arithmetic if heuristic_value is int.
                 # Let's assume valid state for competition context.
                 pass # Assume valid state

            # Ensure all needed floors are valid floors (should be if problem is valid)
            # This check is technically redundant if problem is valid and floors in origin/destin
            # are guaranteed to be in above relations, but kept for robustness.
            valid_needed_floors = [f for f in F_needed if f in self.floor_to_index]

            if valid_needed_floors: # Only calculate moves if there are valid floors to visit
                current_lift_idx = self.floor_to_index[current_lift_floor]
                needed_indices = {self.floor_to_index[f] for f in valid_needed_floors}
                min_needed_idx = min(needed_indices)
                max_needed_idx = max(needed_indices)

                # Estimated moves = distance to the furthest needed floor
                moves = max(abs(current_lift_idx - min_needed_idx), abs(current_lift_idx - max_needed_idx))
                heuristic_value += moves
            # else: F_needed was non-empty but contained only floors not in the index map.
            # In this case, moves = 0 is implicitly handled as no moves are added.


        return heuristic_value
