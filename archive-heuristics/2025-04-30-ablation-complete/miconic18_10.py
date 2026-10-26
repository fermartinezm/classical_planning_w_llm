from fnmatch import fnmatch
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False).
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic: # Inherit from Heuristic if available, e.g., class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort by summing:
    1. Twice the number of passengers waiting at their origin floors (each needs board + depart).
    2. The number of passengers currently boarded (each needs depart).
    3. The number of distinct floors that require a stop (either for pickup or dropoff).
    4. The vertical distance from the lift's current floor to the nearest floor requiring a stop.

    # Assumptions
    - The lift has unlimited capacity.
    - Floors are ordered linearly based on `above` predicates.

    # Heuristic Initialization
    - Parse `above` facts to build a mapping from floor name to its index (0-based from the lowest floor).
    - Parse `destin` facts to build a mapping from passenger name to their destination floor.
    - Store the set of goal facts (`(served ?p)` for all passengers).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value to 0.
    2. Find the lift's current floor from the state fact `(lift-at ?f)`. Convert this to a floor index using the precomputed map.
    3. Identify all passengers who are not yet served. These are passengers `p` for whom `(served ?p)` is in the goal set but not in the current state.
    4. Iterate through the state facts to find:
       - `(origin ?p ?f)` facts: For passengers `p` who are unserved, add 2 to the heuristic (board + depart). Collect floor `f` into `pickup_floors`.
       - `(boarded ?p)` facts: For passengers `p` who are unserved, add 1 to the heuristic (depart). Look up destination `d` and collect into `dropoff_floors`.
    5. Combine `pickup_floors` and `dropoff_floors` into `required_floors`.
    6. If `required_floors` is not empty:
       - Add the number of distinct floors in `required_floors` to the heuristic.
       - Convert `required_floors` to `required_indices`.
       - Calculate the minimum distance from the lift's current floor index to any index in `required_indices`. Add this distance to the heuristic.
    7. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        # Assuming task object has attributes: goals, static
        self.goals = task.goals
        self.static = task.static

        # Build floor order and index map from 'above' facts
        above_map = {} # lower -> higher
        all_floors = set()
        higher_floors_in_above = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                higher, lower = parts[1], parts[2]
                above_map[lower] = higher
                all_floors.add(higher)
                all_floors.add(lower)
                higher_floors_in_above.add(higher)

        self.floor_to_index = {}
        self.index_to_floor = {}

        if all_floors:
            # Find the lowest floor (a floor in all_floors but not a 'higher' floor in any 'above' fact)
            lowest_floor = None
            for floor in all_floors:
                if floor not in higher_floors_in_above:
                    lowest_floor = floor
                    break

            # If lowest_floor is still None, it means there's a cycle or no above facts for >1 floor.
            # Or it's a single floor problem with no above facts.
            if lowest_floor is None and len(all_floors) > 0:
                 # If no above facts, assume the single floor is the lowest/only floor
                 if not above_map:
                      lowest_floor = list(all_floors)[0]
                 else:
                      # This case indicates a problem structure not expected (e.g., disconnected floors, cycles)
                      # For robustness, we could sort floors alphabetically or handle as error.
                      # Assuming well-formed input where 'above' forms a chain.
                      pass # lowest_floor remains None, will cause issue if not handled

            if lowest_floor is not None:
                floor_order = []
                current = lowest_floor
                index = 0
                while current is not None:
                    floor_order.append(current)
                    self.floor_to_index[current] = index
                    self.index_to_floor[index] = current
                    index += 1
                    current = above_map.get(current)
            # else: all_floors was empty or lowest_floor couldn't be determined, maps remain empty

        # Build passenger destination map from 'destin' facts
        self.destin_map = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.destin_map[passenger] = floor

        # Store the set of passengers who *need* to be served (those mentioned in goal served facts).
        self.passengers_to_serve = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Assuming state is a frozenset of fact strings

        h_value = 0

        current_floor = None
        pickup_floors = set()
        boarded_passengers = set()
        
        # Keep track of passengers who are currently served in this state
        served_passengers_in_state = set()

        # First pass to find lift location and served passengers
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'lift-at':
                current_floor = parts[1]
            elif parts[0] == 'served':
                 passenger = parts[1]
                 served_passengers_in_state.add(passenger)

        # If current_floor is not found or not in floor map, return infinity
        if current_floor is None or current_floor not in self.floor_to_index:
             # This state is likely unreachable or invalid
             return float('inf')

        current_idx = self.floor_to_index[current_floor]

        # Second pass to identify unserved passengers (origin/boarded) and calculate costs
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue

             if parts[0] == 'origin':
                 passenger, floor = parts[1], parts[2]
                 # If this passenger needs serving and is waiting at origin (i.e., not served yet)
                 if passenger in self.passengers_to_serve and passenger not in served_passengers_in_state:
                     h_value += 2 # Needs board (1) + depart (1)
                     pickup_floors.add(floor)
             elif parts[0] == 'boarded':
                 passenger = parts[1]
                 # If this passenger needs serving and is boarded (i.e., not served yet)
                 if passenger in self.passengers_to_serve and passenger not in served_passengers_in_state:
                     h_value += 1 # Needs depart (1)
                     boarded_passengers.add(passenger)


        # Determine dropoff floors for currently boarded passengers who need serving
        dropoff_floors = set()
        for passenger in boarded_passengers:
            # boarded_passengers set already filtered for unserved passengers who need serving
            dest_floor = self.destin_map.get(passenger)
            if dest_floor: # Should always exist for valid problems
                dropoff_floors.add(dest_floor)
            # else: passenger has no destination? Invalid problem? Assume valid.

        # 5. Combine required floors
        required_floors = pickup_floors | dropoff_floors

        # 6. Calculate movement cost
        if required_floors:
            # Add number of distinct required floors
            h_value += len(required_floors)

            # Calculate distance from current floor to the nearest required floor
            required_indices = {self.floor_to_index[f] for f in required_floors if f in self.floor_to_index}
            
            if required_indices: # Ensure there are valid indices before calculating min
                min_dist_to_required = float('inf')
                for req_idx in required_indices:
                    min_dist_to_required = min(min_dist_to_required, abs(current_idx - req_idx))
                h_value += min_dist_to_required
            # else: required_floors had invalid floor names? Or was empty after filtering?
            # If required_floors was not empty but required_indices is empty, something is wrong with floor mapping.
            # If required_floors was empty, this block is skipped.

        return h_value
