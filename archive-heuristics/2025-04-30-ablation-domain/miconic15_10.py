from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has the expected structure
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed facts, or handle as an error
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions (floor movements, boarding, departing)
    required to serve all passengers. It calculates the minimum floor travel distance
    needed to visit all floors where passengers are waiting or need to be dropped off,
    and adds the number of individual board and depart actions required.

    # Assumptions
    - The domain follows standard Miconic rules: passengers wait at an origin floor,
      need to be boarded, transported by the lift, and depart at their destination floor.
    - The lift can carry multiple passengers.
    - Floor levels are linearly ordered by the `above` predicate, forming a single chain.
    - Actions are `move-up`, `move-down`, `board`, `depart`. Each counts as 1 action.

    # Heuristic Initialization
    - Parses static facts to determine the linear order of floors and create a mapping
      from floor names to numerical indices (lowest floor is index 0).
    - Parses static facts to store the destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift from the state facts.
    2. Convert the current lift floor name to its numerical index using the mapping created during initialization.
    3. Identify all passengers who have not yet been served by checking for the absence of the `(served ?p)` fact.
    4. If the set of unserved passengers is empty, the goal state is reached, and the heuristic value is 0.
    5. For each unserved passenger, determine their current status:
       - If a `(origin ?p ?f)` fact exists for the passenger `?p`, they are waiting at floor `?f`.
       - If a `(boarded ?p)` fact exists for the passenger `?p`, they are inside the lift.
       (An unserved passenger must be in one of these two states).
    6. Collect the set of unique floors that the lift *must* visit to serve the unserved passengers:
       - Add the origin floor for every unserved passenger who is waiting.
       - Add the destination floor (looked up from initialization data) for every unserved passenger who is boarded.
    7. Convert this set of needed floor names into a set of their corresponding numerical indices.
    8. If there are no needed floor indices (this should only happen if there are no unserved passengers, handled in step 4), the floor movements cost is 0.
    9. If there are needed floor indices, find the minimum (`min_idx`) and maximum (`max_idx`) index among them.
    10. Calculate the minimum number of floor movements required. This is the distance needed to travel from the current lift floor index (`current_idx`) to cover the range [`min_idx`, `max_idx`]. The minimum moves required is the span of the needed floors (`max_idx - min_idx`) plus the minimum distance from the current floor to either end of the span (`min(abs(current_idx - min_idx), abs(current_idx - max_idx))`). If only one floor is needed (`min_idx == max_idx`), this simplifies to `abs(current_idx - min_idx)`.
    11. Count the number of individual `board` actions needed: Each unserved passenger currently waiting at their origin requires one `board` action. This count is the number of unserved passengers for whom an `(origin ?p ?f)` fact exists.
    12. Count the number of individual `depart` actions needed: Each unserved passenger (whether waiting or boarded) will eventually require one `depart` action at their destination. This count is the total number of unserved passengers.
    13. The total heuristic value is the sum of the calculated floor movements, the number of board actions, and the number of depart actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, floor indexing,
        and passenger destinations from static facts.
        """
        self.goals = task.goals  # Goals are (served p) facts
        static_facts = task.static

        # 1. Build floor order and index mapping
        all_floors = set()
        floors_with_something_above = set()
        floor_immediately_above_map = {} # Map floor -> floor immediately above it

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                all_floors.add(f_above)
                all_floors.add(f_below)
                floors_with_something_above.add(f_below)
                floor_immediately_above_map[f_below] = f_above # f_above is immediately above f_below

        # Find the lowest floor: a floor in all_floors that is not in floors_with_something_above
        # Assumes a single linear chain of floors.
        lowest_floor = None
        for floor in all_floors:
            if floor not in floors_with_something_above:
                lowest_floor = floor
                break # Found the lowest floor

        # If no lowest floor found but there are floors, try guessing based on name suffix
        # This handles single-floor cases or potentially malformed inputs, though standard miconic is linear.
        if lowest_floor is None and all_floors:
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # Fallback: Assume f<num> naming and highest number is lowest floor.
                 try:
                     def get_floor_num(f_name):
                         num_str = ''.join(filter(str.isdigit, f_name))
                         return int(num_str) if num_str else -1 # Use -1 if no number found
                     lowest_floor = max(all_floors, key=get_floor_num)
                 except Exception:
                      # If floor names are not f<num> or parsing fails,
                      # we cannot reliably order floors. Heuristic is broken.
                      self.floor_to_index = {} # Indicate failure to build map
                      print("Warning: Could not determine floor order from 'above' facts or names. Heuristic may be incorrect.")


        # Build ordered list of floors from lowest to highest
        floor_list = []
        current = lowest_floor
        # Traverse upwards using the map
        while current is not None and current in all_floors:
            floor_list.append(current)
            current = floor_immediately_above_map.get(current)

        # Create floor name to index mapping (lowest floor is index 0)
        self.floor_to_index = {floor: i for i, floor in enumerate(floor_list)}

        # 2. Store passenger destinations
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger, destination_floor = parts[1], parts[2]
                self.destinations[passenger] = destination_floor

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if floor indexing was successful during initialization
        if not self.floor_to_index:
             # Cannot compute heuristic without floor order
             return float('inf') # Indicate unsolvable or invalid problem setup

        # 1. Identify current lift floor
        lift_at_fact = next((fact for fact in state if match(fact, "lift-at", "*")), None)
        if lift_at_fact is None:
             # Lift location must be known in a valid state
             return float('inf') # Should not happen in standard miconic problems

        current_floor = get_parts(lift_at_fact)[1]
        current_idx = self.floor_to_index.get(current_floor)

        if current_idx is None:
             # Current lift floor not found in floor index map - indicates setup issue
             return float('inf')

        # 2. Identify unserved passengers
        # Unserved passengers are those not marked as 'served' but are either 'origin' or 'boarded'
        current_served = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        current_waiting = {get_parts(fact)[1] for fact in state if match(fact, "origin", "*", "*")}
        current_boarded = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        unserved_passengers = (current_waiting | current_boarded) - current_served

        # 3. Goal check
        if not unserved_passengers:
            return 0

        # 4. & 5. Distinguish waiting vs. boarded unserved passengers and get origins
        waiting_passengers = current_waiting - current_served
        boarded_passengers = current_boarded - current_served

        passenger_origins = {}
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "origin" and parts[1] in waiting_passengers:
                  passenger_origins[parts[1]] = parts[2]

        # 6. Collect needed floors
        floors_to_visit_for_pickup = {passenger_origins[p] for p in waiting_passengers if p in passenger_origins}
        floors_to_visit_for_dropoff = {self.destinations[p] for p in boarded_passengers if p in self.destinations}
        all_needed_floors = floors_to_visit_for_pickup | floors_to_visit_for_dropoff

        # 7. Convert needed floors to indices
        needed_indices = {self.floor_to_index[f] for f in all_needed_floors if f in self.floor_to_index}

        # 8. Calculate floor movements
        moves = 0
        if needed_indices:
            min_idx = min(needed_indices)
            max_idx = max(needed_indices)

            if min_idx == max_idx:
                 # Only one floor needed, just move to it
                 moves = abs(current_idx - min_idx)
            else:
                 # Need to cover the range [min_idx, max_idx] starting from current_idx
                 # Minimum moves = distance to nearest end + span
                 moves = (max_idx - min_idx) + min(abs(current_idx - min_idx), abs(current_idx - max_idx))

        # 9. Count board actions
        # Each waiting passenger needs 1 board action
        board_actions = len(waiting_passengers)

        # 10. Count depart actions
        # Each unserved passenger needs 1 depart action eventually
        depart_actions = len(unserved_passengers)

        # 11. Total heuristic
        total_cost = moves + board_actions + depart_actions

        return total_cost
