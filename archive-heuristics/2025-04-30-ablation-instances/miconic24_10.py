from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # This case should ideally not happen with valid PDDL fact strings
         # print(f"Warning: Unexpected fact format: {fact}") # Optional warning
         return [] # Return empty list for safety
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the minimum required board/depart actions for unserved passengers
    and an estimate of the lift movement cost to visit all necessary floors.

    # Assumptions
    - There is a single lift.
    - Floors are totally ordered by the `above` predicate, forming a single linear sequence.
    - Floor names are consistently structured (e.g., f1, f2, ...).
    - The cost of each action (board, depart, up, down) is 1.
    - All passengers have a defined destination in the static facts.
    - All floors mentioned in the problem are ordered by the `above` predicate chain.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts, storing them in `self.passenger_to_destin`.
    - Builds a mapping from floor names (strings) to integer indices (`self.floor_to_int`) based on the `above` predicate. It identifies the lowest floor and traverses the `above` chain to assign sequential integer indices.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the lift's current floor by finding the `(lift-at ?f)` fact in the state. Convert its name `?f` to an integer index using the `self.floor_to_int` mapping. Store this as `current_lift_floor_int`.
    2. Initialize `num_unboarded = 0` and `num_boarded = 0`. These will count unserved passengers who are either at their origin or boarded.
    3. Initialize an empty set `required_floors_int_set`. This set will store the integer indices of all floors the lift must visit to make progress towards serving passengers.
    4. Iterate through the facts in the current state:
        - If a fact matches the pattern `(origin ?p ?f)`: This passenger `?p` is waiting at floor `?f`. If `?p` is not yet served (check if `(served ?p)` is *not* in the state), increment `num_unboarded` and add the integer index of floor `?f` (obtained from `self.floor_to_int`) to `required_floors_int_set`.
        - If a fact matches the pattern `(boarded ?p)`: This passenger `?p` is inside the lift. If `?p` is not yet served, increment `num_boarded`. Look up the destination floor `?d` for passenger `?p` using `self.passenger_to_destin`. Add the integer index of floor `?d` to `required_floors_int_set`.
        - Facts like `(served ?p)` and `(lift-at ?f)` (after initial identification) are ignored as they are processed differently or don't directly contribute to the remaining actions in this counting method. Static facts are also ignored here.
    5. Calculate the base cost: This is the sum `num_unboarded + num_boarded`. This represents a lower bound on the number of `board` and `depart` actions required.
    6. Calculate the estimated move cost:
        - If `required_floors_int_set` is empty, it means all unserved passengers are either served or are boarded and need to depart at the current floor. No further vertical movement is estimated for these passengers, so `move_cost = 0`.
        - If `required_floors_int_set` is not empty, find the minimum (`min_req_floor_int`) and maximum (`max_req_floor_int`) integer indices within the set.
        - The estimated move cost is the total vertical span from the current lift floor to encompass all required floors: `max(current_lift_floor_int, max_req_floor_int) - min(current_lift_floor_int, min_req_floor_int)`. This counts the distance from the current floor to the furthest required floor in either direction.
    7. The total heuristic value is the sum of the `base_cost` and the `move_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and
        building the floor-to-integer mapping from static facts.
        """
        self.goals = task.goals # Goals are used to check if a state is a goal state (h=0)
        self.static = task.static

        # Build passenger destination mapping from static facts
        self.passenger_to_destin = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_to_destin[passenger] = floor

        # Build floor-to-integer mapping based on 'above' predicate
        # (above f_lower f_higher) means f_higher is directly above f_lower
        above_map = {} # lower_floor_str -> higher_floor_str
        is_lower_floor = set()
        is_higher_floor = set()
        all_floors_in_above = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                above_map[f_lower] = f_higher
                is_lower_floor.add(f_lower)
                is_higher_floor.add(f_higher)
                all_floors_in_above.add(f_higher)
                all_floors_in_above.add(f_lower)

        self.floor_to_int = {}
        if not all_floors_in_above:
             # This case implies no 'above' facts, likely only one floor or invalid domain structure.
             # Try to find any floor mentioned in initial state or goals/destinations.
             potential_floors = set()
             # Look in initial state for lift location or passenger origins
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if parts and (parts[0] == "lift-at" or parts[0] == "origin"):
                     if len(parts) > 1: # Ensure there's a floor argument
                         potential_floors.add(parts[-1]) # Last part is the floor
             # Look in passenger destinations
             for floor in self.passenger_to_destin.values():
                  potential_floors.add(floor)

             if potential_floors:
                 # Just map the first found floor to 1
                 self.floor_to_int[list(potential_floors)[0]] = 1
             # else: self.floor_to_int remains empty, will cause KeyError later if floors are used.

        else:
            # Find the lowest floor (is_lower_floor but not is_higher_floor within the 'above' facts)
            lowest_floor_str = None
            for f in is_lower_floor:
                if f not in is_higher_floor:
                    lowest_floor_str = f
                    break

            # If a clear lowest floor isn't found among 'above' facts, pick one from the set
            # This might happen if 'above' facts don't form a single chain or miss floors.
            # Standard miconic should have a clear lowest floor.
            if lowest_floor_str is None:
                 # Fallback: pick an arbitrary floor from those involved in 'above'
                 if all_floors_in_above:
                     lowest_floor_str = list(all_floors_in_above)[0]

            if lowest_floor_str:
                floor_int = 1
                current_floor_str = lowest_floor_str
                self.floor_to_int[current_floor_str] = floor_int

                # Traverse upwards using the above_map to build the mapping
                while current_floor_str in above_map:
                    next_floor_str = above_map[current_floor_str]
                    floor_int += 1
                    self.floor_to_int[next_floor_str] = floor_int
                    current_floor_str = next_floor_str

        # Note: This floor mapping assumes all relevant floors in the problem
        # are part of the single chain defined by 'above' facts starting from the lowest floor.
        # If a floor appears in state/goals but not in this chain, lookup will fail.
        # Standard miconic problems usually satisfy this assumption.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached - heuristic is 0 at goal
        if self.goals <= state:
            return 0

        current_lift_floor_str = None
        num_unboarded = 0
        num_boarded = 0
        required_floors_int_set = set()

        # Iterate through the state to find relevant facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "lift-at":
                # (lift-at ?f)
                if len(parts) == 2:
                    current_lift_floor_str = parts[1]
            elif predicate == "origin":
                # (origin ?p ?f)
                if len(parts) == 3:
                    passenger = parts[1]
                    origin_floor_str = parts[2]
                    # Only count if the passenger is not yet served
                    if f'(served {passenger})' not in state:
                        num_unboarded += 1
                        if origin_floor_str in self.floor_to_int:
                            required_floors_int_set.add(self.floor_to_int[origin_floor_str])
                        # else: Floor not mapped, potential issue with domain/instance definition

            elif predicate == "boarded":
                # (boarded ?p)
                if len(parts) == 2:
                    passenger = parts[1]
                    # Only count if the passenger is not yet served
                    if f'(served {passenger})' not in state:
                        num_boarded += 1
                        # Get destination floor for this boarded passenger
                        if passenger in self.passenger_to_destin:
                            destin_floor_str = self.passenger_to_destin[passenger]
                            if destin_floor_str in self.floor_to_int:
                                 required_floors_int_set.add(self.floor_to_int[destin_floor_str])
                            # else: Destination floor not mapped or passenger has no destination (issue with domain/instance)
                        # else: Passenger has no destination defined (issue with domain/instance)

            # served facts are implicitly handled by checking 'not in state' above
            # static facts like 'above' and 'destin' are not in the state, they are in self.static

        # If lift-at fact wasn't found, state is likely invalid or initial parsing failed.
        # Returning a large value or raising error might be appropriate in a robust system.
        # Assuming valid states always have (lift-at ?f).
        if current_lift_floor_str is None:
             # This indicates an unexpected state format. Return a high value.
             # print("Warning: lift-at fact not found in state.") # Optional warning
             return float('inf') # Cannot compute heuristic without lift location

        current_lift_floor_int = self.floor_to_int.get(current_lift_floor_str)
        # If current lift floor is not in mapping, something is wrong.
        if current_lift_floor_int is None:
             # print(f"Warning: Current lift floor '{current_lift_floor_str}' not found in floor mapping.") # Optional warning
             return float('inf') # Cannot compute heuristic

        # Calculate base cost (minimum board/depart actions)
        base_cost = num_unboarded + num_boarded

        # Calculate move cost
        move_cost = 0
        if required_floors_int_set:
            min_req_floor_int = min(required_floors_int_set)
            max_req_floor_int = max(required_floors_int_set)

            # Estimated moves = total vertical span from current floor to cover all required floors
            # This is max(current, max_req) - min(current, min_req)
            move_cost = max(current_lift_floor_int, max_req_floor_int) - min(current_lift_floor_int, min_req_floor_int)

        total_cost = base_cost + move_cost

        return total_cost
