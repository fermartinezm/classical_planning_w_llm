from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """
    Extracts the predicate name and arguments from a PDDL fact string.
    e.g., '(predicate arg1 arg2)' -> ['predicate', 'arg1', 'arg2']
    """
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """
    Checks if a fact string matches a pattern of predicate and arguments.
    Uses fnmatch for wildcard matching.
    e.g., match('(at obj1 rooma)', 'at', 'obj1', '*') -> True
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for comparison
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all passengers served)
    by summing two components:
    1. A base cost: 2 actions (board + depart) for each unserved passenger.
    2. A movement cost: An estimate of the minimum number of lift moves required
       to visit all necessary floors (origin floors of waiting passengers and
       destination floors of boarded passengers).

    Assumptions:
    - The 'above' predicates in the static facts define a linear ordering of floors.
    - Passenger destinations are static and available in the initial static facts.
    - All passengers relevant to the goal are mentioned in the static 'destin' facts.
    - State representation uses strings like '(predicate arg1 arg2)'.
    - A 'lift-at' fact is always present in a valid state.

    Heuristic Initialization (__init__):
    - Parses the static facts to determine the linear order of floors using the
      'above' predicates. It builds a mapping from floor names to numerical indices.
    - Parses the static facts to store the destination floor for each passenger.
    - Collects the names of all passengers based on the available destinations.

    Step-By-Step Thinking for Computing Heuristic (__call__):
    1. Check if the current state is the goal state. If yes, return 0.
    2. Find the current floor of the lift.
    3. Identify all passengers who have not yet been served.
    4. Count the number of unserved passengers. This contributes 2 * num_unserved
       to the heuristic (one board and one depart action per passenger).
    5. Determine the set of 'required floors' that the lift must visit. This set
       includes:
       - The origin floor for every unserved passenger who is currently waiting
         at their origin.
       - The destination floor for every unserved passenger who is currently
         boarded in the lift.
       - The destination floor for every unserved passenger who is currently
         waiting at their origin (as they will eventually need to be dropped off).
    6. If there are no required floors (which should only happen if all passengers
       are served, handled in step 1), the move cost is 0.
    7. If there are required floors, find the minimum and maximum floor indices
       among these required floors.
    8. Calculate the estimated minimum number of moves required for the lift,
       starting from its current floor, to visit all floors within the range
       defined by the minimum and maximum required floor indices. This is
       calculated as `min(abs(current_lift_index - min_required_index),
       abs(current_lift_index - max_required_index)) + (max_required_index - min_required_index)`.
       This represents the distance to reach one end of the required floor range
       and then traverse the entire range.
    9. The total heuristic value is the sum of the base cost (from step 4) and
       the estimated move cost (from step 8).
    """
    def __init__(self, task):
        super().__init__(task) # Call parent constructor if needed, though Heuristic base might not have one
        self.goals = task.goals
        static_facts = task.static

        # Heuristic Initialization

        # 1. Parse floor order and create floor_to_index map
        # Build map: floor_lower -> floor_above
        floor_above_map = {}
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "?f_higher", "?f_lower"):
                f_higher, f_lower = get_parts(fact)[1:]
                floor_above_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Find the lowest floor (a floor that is not a value in floor_above_map)
        lowest_floor = None
        floors_that_are_above_others = set(floor_above_map.values())
        for floor in all_floors:
            if floor not in floors_that_are_above_others:
                lowest_floor = floor
                break

        # Build the ordered list of floors starting from the lowest
        self.floor_list = []
        current_floor = lowest_floor
        # Handle case with no floors or single floor (no 'above' facts)
        if lowest_floor is not None:
            while current_floor is not None:
                self.floor_list.append(current_floor)
                current_floor = floor_above_map.get(current_floor)
        elif all_floors:
             # If there are floors but no 'above' facts, assume they are all on the same level or order doesn't matter for moves (moves = 0)
             # Or, if there's only one floor, it's the lowest.
             if len(all_floors) == 1:
                 self.floor_list = list(all_floors)
             # If multiple floors but no above, the problem is ill-defined for vertical movement.
             # We proceed, but floor_to_index might be incomplete or based on arbitrary order.
             # For robustness, we could sort alphabetically or raise an error, but let's assume valid PDDL implies floor order.
             # If no lowest_floor found but all_floors is not empty, it might indicate a cycle or disconnected graph, which is invalid for a linear order.
             # A simple fallback is to just use the floors found, perhaps sorted.
             # However, given the domain structure, a linear 'above' chain is expected.
             pass # If lowest_floor is None and all_floors is empty, floor_list remains empty.

        # Create floor_to_index map
        self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_list)}

        # 2. Parse passenger destinations
        self.destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "?p", "?d"):
                p, d = get_parts(fact)[1:]
                self.destinations[p] = d

        # 3. Collect all passenger names relevant to the goal
        # Assume passengers mentioned in destin facts are all relevant passengers
        self.all_passengers = set(self.destinations.keys())

    def __call__(self, node):
        state = node.state
        task = node.task # Access task from node to check goal

        # Check for goal state
        if task.goal_reached(state):
             return 0

        # Step-By-Step Thinking for Computing Heuristic

        # 1. Find current lift floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "?f"):
                lift_floor = get_parts(fact)[1]
                break
        # Assuming lift_at fact is always present in a valid state

        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "?p")}
        unserved_passengers = {p for p in self.all_passengers if p not in served_passengers}
        num_unserved = len(unserved_passengers)

        # If no unserved passengers, but goal_reached was False, something is wrong.
        # This case should be covered by goal_reached check, but as a safeguard:
        if num_unserved == 0:
             return 0 # Should not be reached if goal_reached is correct

        # 3. Identify required floors to visit
        # These are origin floors of waiting unserved passengers
        # and destination floors of boarded unserved passengers
        # and destination floors of waiting unserved passengers (they need dropoff later)
        required_floors = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "origin", "?p", "?o"):
                 p, o = parts[1:]
                 if p in unserved_passengers:
                     required_floors.add(o) # Need to pick up
                     if p in self.destinations:
                         required_floors.add(self.destinations[p]) # Need to drop off later
            elif match(fact, "boarded", "?p"):
                 p = parts[1]
                 if p in unserved_passengers:
                     if p in self.destinations:
                         required_floors.add(self.destinations[p]) # Need to drop off

        # 4. Calculate minimum moves to visit required floors
        moves = 0
        # Only calculate moves if there are floors defined and floors to visit
        if self.floor_list and required_floors:
            # Ensure all required floors are in our floor_to_index map
            # (Should be true if static facts are consistent)
            valid_required_floors = {f for f in required_floors if f in self.floor_to_index}

            if valid_required_floors:
                required_indices = {self.floor_to_index[f] for f in valid_required_floors}
                min_visit_idx = min(required_indices)
                max_visit_idx = max(required_indices)
                lift_idx = self.floor_to_index.get(lift_floor) # Use .get for safety

                # If lift_floor is valid and in our floor map
                if lift_idx is not None:
                     # Minimum moves to cover the range [min_visit_idx, max_visit_idx] starting from lift_idx
                     # Go from lift_idx to one end, then sweep to the other end.
                     moves = min(abs(lift_idx - min_visit_idx), abs(lift_idx - max_visit_idx)) + (max_visit_idx - min_visit_idx)
                # If lift_idx is None (e.g., lift_floor not in floor_list, perhaps single floor case with no above facts),
                # moves remain 0, which is correct for a single floor.

        # 5. Calculate total heuristic
        # Base cost: 2 actions (board + depart) per unserved passenger
        # Add estimated move cost
        h = num_unserved * 2 + moves

        return h
