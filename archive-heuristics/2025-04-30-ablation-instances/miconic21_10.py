from fnmatch import fnmatch
# Assuming heuristic_base is available in the specified path
from heuristics.heuristic_base import Heuristic
import re # Although split() is generally sufficient for typical PDDL facts

# Define helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or invalid fact format defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return an empty list for invalid facts
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

# Define the heuristic class
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the number of 'board' actions needed (for waiting passengers),
    the number of 'depart' actions needed (for boarded passengers), and
    estimates the minimum number of 'up'/'down' actions required to visit
    all necessary pickup and dropoff floors.

    # Assumptions
    - Passengers must be picked up at their origin floor and dropped off at their destination floor.
    - The lift can carry multiple passengers.
    - The 'above' predicate defines a linear ordering of floors.
    - The cost of 'board', 'depart', 'up', and 'down' actions is 1.
    - All passenger destinations are provided in the static facts via the `(destin ?p ?f)` predicate.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.
    - Determine the linear order of floors based on the 'above' predicate and create a mapping from floor names to integer levels. The lowest floor is assigned level 1.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the lift from the state fact `(lift-at ?f)`.
    2. Initialize counters for actions needed: `n_waiting = 0`, `n_boarded = 0`.
    3. Initialize sets for required floors: `pickup_floors = set()`, `dropoff_floors = set()`.
    4. Iterate through all passengers whose destination is known (from the static facts `self.passenger_destinations`).
    5. For each passenger:
       - Check if the passenger is already served using the state fact `(served ?p)`. If yes, ignore this passenger.
       - If the passenger is unserved:
         - Check if the passenger is waiting at their origin floor using the state fact `(origin ?p ?o)`. If yes:
           - Increment `n_waiting` (counts the required 'board' action).
           - Add the origin floor `o` to the `pickup_floors` set.
         - Else, check if the passenger is currently boarded in the lift using the state fact `(boarded ?p)`. If yes:
           - Increment `n_boarded` (counts the required 'depart' action).
           - Add the passenger's destination floor (looked up from `self.passenger_destinations`) to the `dropoff_floors` set.
         - (Passengers should be either served, waiting at origin, or boarded if unserved in a valid state).
    6. Calculate the estimated minimum number of 'move' actions ('up'/'down'):
       - Combine `pickup_floors` and `dropoff_floors` into `all_target_floors`.
       - If `all_target_floors` is empty, the move cost is 0.
       - Otherwise, map the current lift floor and all target floors to their integer levels using the pre-calculated `self.floor_map`.
       - Filter out any target floors not found in the floor map.
       - If there are still target floors:
         - Find the minimum (`min_target_level`) and maximum (`max_target_level`) levels among the valid target floors.
         - Get the level of the current lift floor. If the current floor is not in the map, return infinity.
         - The minimum move cost is calculated as:
           - `(max_target_level - min_target_level)` (cost to traverse the full range of target floors)
           - PLUS the cost to reach the nearest extreme of the target range from the current floor:
             - If `current_level` is below `min_target_level`, add `(min_target_level - current_level)`.
             - If `current_level` is above `max_target_level`, add `(current_level - max_target_level)`.
             - If `current_level` is within the range, add `min(current_level - min_target_level, max_target_level - current_level)`.
    7. The total heuristic value is the sum of `n_waiting` (board actions), `n_boarded` (depart actions), and the estimated `move_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination = get_parts(fact)
                self.passenger_destinations[passenger] = destination

        # Extract floor order and create floor-to-level map
        # Assuming (above f_higher f_lower) means f_higher is directly above f_lower
        # This implies f_lower < f_higher in terms of level.
        # Example: (above f2 f1), (above f3 f2) -> f1 < f2 < f3
        # We need to find the lowest floor and traverse upwards.
        # The lowest floor is one that is NOT the f_lower in any (above f_higher f_lower) fact.

        floors_that_are_f_higher = set()
        floors_that_are_f_lower = set()
        floor_above_of = {} # f_lower -> f_higher
        for fact in static_facts:
             if match(fact, "above", "*", "*"):
                 _, f_higher, f_lower = get_parts(fact)
                 floors_that_are_f_higher.add(f_higher)
                 floors_that_are_f_lower.add(f_lower)
                 floor_above_of[f_lower] = f_higher

        all_floors = floors_that_are_f_higher | floors_that_are_f_lower

        # Find the lowest floor: a floor that is not the f_lower in any (above f_higher f_lower) fact.
        lowest_floor = None
        for floor in all_floors:
            if floor not in floors_that_are_f_lower:
                 lowest_floor = floor
                 break

        # Handle cases where a clear lowest floor isn't found by the simple rule
        if lowest_floor is None and all_floors:
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # Fallback: Find the highest floor (not f_higher in any relation) and traverse down
                 highest_floor_candidate = None
                 for floor in all_floors:
                     if floor not in floors_that_are_f_higher:
                         highest_floor_candidate = floor
                         break

                 if highest_floor_candidate:
                     # Traverse down from the highest to find the lowest
                     current = highest_floor_candidate
                     floor_lower_of = {f_higher: f_lower for f_lower, f_higher in floor_above_of.items()}
                     while True:
                         next_lower = floor_lower_of.get(current)
                         if next_lower is None:
                             lowest_floor = current
                             break
                         current = next_lower
                 else:
                     # Still couldn't find a linear order structure
                     print("Warning: Could not determine floor order from static facts.")
                     self.floor_map = {}
                     self.floors_ordered = []
                     return


        if lowest_floor is None:
             # No floors found or unable to determine lowest. Handle gracefully.
             self.floor_map = {}
             self.floors_ordered = []
             # print("Warning: No floors found in static facts.") # Avoid excessive printing
             return

        # Build ordered list (lowest to highest) and map (lowest gets level 1)
        self.floors_ordered = [] # Will store lowest -> highest
        current_floor = lowest_floor
        level = 1
        while current_floor is not None:
            self.floors_ordered.append(current_floor)
            self.floor_map[current_floor] = level
            level += 1
            current_floor = floor_above_of.get(current_floor) # Get the floor directly above current_floor

        if not self.floor_map:
             print("Warning: Floor map is empty after processing.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If floor map is empty but there are passengers, heuristic is infinity
        if not self.floor_map and self.passenger_destinations:
             return float('inf')

        # Check if goal is reached
        if self.goals <= state:
            return 0

        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        if current_lift_floor is None:
             # This should not happen in a valid state, but handle defensively
             print("Error: Lift location not found in state.")
             return float('inf')

        n_waiting = 0
        n_boarded = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Iterate through all passengers whose destination is known (assumed to be all passengers needing service)
        for passenger, destination_floor in self.passenger_destinations.items():
            # Check if passenger is already served
            if f"(served {passenger})" in state:
                continue # Passenger is served, ignore

            # Passenger is unserved. Find their current status.
            is_boarded = f"(boarded {passenger})" in state
            current_origin_floor_fact = None
            for fact in state:
                 if match(fact, "origin", passenger, "*"):
                     current_origin_floor_fact = fact
                     break

            if current_origin_floor_fact is not None: # Passenger is waiting at origin
                 n_waiting += 1
                 _, _, origin_floor = get_parts(current_origin_floor_fact)
                 pickup_floors.add(origin_floor)
            elif is_boarded: # Passenger is boarded (and not served)
                 n_boarded += 1
                 dropoff_floors.add(destination_floor)
            # Else: Passenger is unserved but not at origin and not boarded.
            # This state implies an issue (e.g., passenger disappeared).
            # We ignore such passengers for the heuristic calculation, assuming they can't be served.


        # Calculate move cost
        all_target_floors = pickup_floors | dropoff_floors
        move_cost = 0

        if all_target_floors:
            current_level = self.floor_map.get(current_lift_floor)
            if current_level is None:
                 # Current lift floor is not in the floor map - indicates problem setup issue
                 print(f"Error: Current lift floor '{current_lift_floor}' not found in floor map.")
                 return float('inf')

            # Filter target floors to only include those present in our floor map
            valid_target_floors = {f for f in all_target_floors if f in self.floor_map}

            if not valid_target_floors:
                 # This happens if all target floors were not in the floor map (e.g., not in 'above' facts)
                 # Cannot estimate moves meaningfully for these floors.
                 # We still count board/depart actions for passengers on known floors.
                 # If all target floors are unknown, move_cost remains 0.
                 pass # move_cost is already 0

            else:
                target_levels = {self.floor_map[f] for f in valid_target_floors}
                min_target_level = min(target_levels)
                max_target_level = max(target_levels)

                # Cost to traverse the full range of target floors
                move_cost = (max_target_level - min_target_level)

                # Cost to reach the nearest extreme of the target range from the current floor
                if current_level < min_target_level:
                    move_cost += (min_target_level - current_level)
                elif current_level > max_target_level:
                    move_cost += (current_level - max_target_level)
                else: # current_level is within [min_target_level, max_target_level]
                    move_cost += min(current_level - min_target_level, max_target_level - current_level)

        # Total heuristic is sum of actions
        total_cost = n_waiting + n_boarded + move_cost

        return total_cost
