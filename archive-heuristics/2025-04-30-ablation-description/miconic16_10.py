from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have enough parts to match the args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining cost by summing:
    1. The number of passengers waiting at their origin floors (each needs boarding).
    2. The number of passengers currently boarded in the lift (each needs departing).
    3. An estimate of the lift movement cost required to visit all necessary floors.

    # Assumptions
    - Floors are arranged linearly and ordered by the `above` predicate.
    - The cost of moving the lift one floor is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The estimated lift movement cost is a lower bound based on the range of floors that must be visited.

    # Heuristic Initialization
    - Extracts the floor ordering from the static `above` facts to create a floor-to-index mapping.
    - Extracts the destination floor for each passenger from the static `destin` facts.
    - Identifies all relevant passengers from static `origin` and `destin` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. Find the current floor of the lift. If the lift location is unknown in a non-goal state, return infinity.
    3. Identify all passengers who are not yet served based on the goal conditions and current state.
    4. If there are no unserved passengers, return 0 (already handled by goal check, but for safety).
    5. Separate unserved passengers into two groups based on the current state: those waiting at their origin (`origin` predicate) and those currently boarded (`boarded` predicate).
    6. Count the number of waiting passengers. This contributes directly to the heuristic (each needs a 'board' action).
    7. Count the number of boarded passengers. This contributes directly to the heuristic (each needs a 'depart' action).
    8. Determine the set of floors the lift *must* visit:
       - The origin floors of all waiting passengers (found in the current state).
       - The destination floors of all unserved passengers (found in static facts).
    9. Include the current lift floor in the set of required floors for movement calculation.
    10. If the set of required floors (excluding the current lift floor if it's the only one) is effectively empty (i.e., only contains the current floor, or is empty), the movement cost is 0.
    11. If the set is not empty, get the floor indices for the current floor and all required floors using the pre-calculated mapping.
    12. Calculate the minimum and maximum floor indices among the required floors.
    13. Estimate the lift movement cost as the distance from the current floor index to the nearest extreme required floor index (min or max), plus the total range between the minimum and maximum required floor indices. This is `min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)`.
    14. The total heuristic value is the sum of the counts from steps 6 and 7, and the estimated movement cost from step 13.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions (used to check if a passenger is served)
        self.static = task.static  # Static facts (used for above and destin)

        # Build floor index mapping from (above f_a f_b) facts
        # (above f_a f_b) means f_a is directly above f_b
        above_map = {} # Maps f_below -> f_above
        all_floors = set()
        floors_that_are_below = set() # Floors that appear as f_below

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)
                floors_that_are_below.add(f_below)

        # Find the lowest floor: a floor that is in all_floors but is not a value in above_map
        # (i.e., no floor is below it in the defined above relationships)
        lowest_floor = None
        floors_that_are_above_values = set(above_map.values())
        for floor in all_floors:
            if floor not in floors_that_are_above_values:
                 lowest_floor = floor
                 break

        self.floor_indices = {}
        if lowest_floor:
            current = lowest_floor
            index = 0
            # Iterate upwards from the lowest floor using the above_map
            while current in all_floors: # Or while current is not None
                 self.floor_indices[current] = index
                 if current in above_map:
                     current = above_map[current]
                     index += 1
                 else:
                     # Reached the top floor (a floor that is never a key in above_map)
                     break
        # If no floors or no above facts, floor_indices remains empty.
        # This might happen in trivial problems or if the domain/instance is malformed.
        # The heuristic should handle this gracefully (e.g., return 0 if no floors).


        # Store destination floors for each passenger
        self.passenger_destinations = {}
        # Identify all relevant passengers from static facts (either origin or destin)
        self.all_passengers = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "destin" and len(parts) == 3:
                passenger, destination_floor = parts[1:]
                self.passenger_destinations[passenger] = destination_floor
                self.all_passengers.add(passenger)
            elif parts[0] == "origin" and len(parts) == 3:
                 passenger, _ = parts[1:]
                 self.all_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is reached (all passengers served)
        # This is important for greedy best-first search to terminate correctly
        if self.goals <= state:
             return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown in a non-goal state, it's likely an invalid state
        if current_lift_floor is None:
             return math.inf # Cannot proceed without lift location


        # Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = {p for p in self.all_passengers if p not in served_passengers}

        # If no unserved passengers, heuristic is 0 (already handled by goals <= state, but double check)
        if not unserved_passengers:
             return 0

        # Identify waiting and boarded passengers among the unserved ones
        waiting_passengers = set()
        boarded_passengers = set()
        origins_waiting = set() # Floors where passengers are waiting

        for passenger in unserved_passengers:
            if f'(boarded {passenger})' in state:
                boarded_passengers.add(passenger)
            else: # Assume they are at their origin if not served and not boarded
                 # Find their origin floor
                 origin_floor = None
                 for fact in state:
                     if match(fact, "origin", passenger, "*"):
                         origin_floor = get_parts(fact)[2]
                         break
                 # Add to waiting only if origin is found in state
                 if origin_floor:
                     waiting_passengers.add(passenger)
                     origins_waiting.add(origin_floor)


        # Heuristic component 1: Number of board actions needed
        h_board = len(waiting_passengers)

        # Heuristic component 2: Number of depart actions needed
        # Only count passengers currently boarded, as waiting passengers will be boarded later
        h_depart = len(boarded_passengers)

        # Heuristic component 3: Estimated lift movement cost
        # Floors the lift must visit: origins of waiting passengers + destinations of all unserved passengers
        destinations_needed = {self.passenger_destinations[p] for p in unserved_passengers if p in self.passenger_destinations}

        all_required_floors = origins_waiting.union(destinations_needed)

        h_movement = 0
        # Only calculate movement if there are floors to visit other than potentially the current one
        if all_required_floors:
            # Get indices for current floor and required floors
            current_idx = self.floor_indices.get(current_lift_floor)
            required_indices = {self.floor_indices.get(f) for f in all_required_floors if f in self.floor_indices}

            # Ensure current floor index is valid and required floors have valid indices
            # And ensure there's at least one required floor with a valid index
            if current_idx is not None and required_indices:
                min_req_idx = min(required_indices)
                max_req_idx = max(required_indices)

                # Estimate movement cost: distance to nearest extreme + full range traversal
                h_movement = min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)
            # else: # This case means all_required_floors was not empty, but some floors didn't have indices
                  # or the current floor didn't have an index. This suggests a problem
                  # with the floor mapping setup or the state/static facts.
                  # In a well-formed problem, this branch shouldn't be needed if all_required_floors is not empty.
                  # If it happens, the heuristic might be inaccurate, but returning 0 is better than crashing.
                  # The current_idx is guaranteed to be not None here due to the check at the start.
                  # So the only issue would be if required_indices is empty despite all_required_floors not being empty,
                  # which implies required floors are not in self.floor_indices.
                  # Let's assume valid inputs where all floors mentioned have indices.
                  # If required_indices is empty, it means all_required_floors contained floors not in self.floor_indices.
                  # This should ideally not happen. If it does, movement cost is effectively unknown, but 0 is a safe lower bound.


        # Total heuristic is the sum of components
        total_heuristic = h_board + h_depart + h_movement

        return total_heuristic
