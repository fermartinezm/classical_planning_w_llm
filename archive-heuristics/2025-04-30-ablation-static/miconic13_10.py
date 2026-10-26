from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally
# In a real planning system, this would be imported.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            pass

        def __call__(self, node):
            raise NotImplementedError("This is a dummy base class. Implement __call__ in derived class.")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It consists of two main components:
    1. An estimate of the number of 'board' and 'depart' actions needed.
    2. An estimate of the minimum vertical movement actions needed to visit all
       required floors (origins for unboarded passengers, destinations for boarded passengers).

    # Assumptions
    - The 'above' predicate defines a total order on floors. (above f1 f2) means f1 is above f2.
    - Each 'board' and 'depart' action costs 1.
    - Each 'up' or 'down' action moving one floor costs 1.
    - The lift can carry multiple passengers.

    # Heuristic Initialization
    - Parses the floor ordering from the static 'above' facts to create a floor-to-index mapping.
      The index of a floor is the number of other floors it is above. This means the lowest floor has index 0.
    - Parses the destination floor for each passenger from the initial state or static facts.
    - Identifies all passenger names from initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers who are not yet 'served'.
    2. Categorize unserved passengers into those waiting at their origin ('unboarded unserved')
       and those already inside the lift ('boarded unserved').
    3. Calculate the 'board'/'depart' action cost:
       - Each unboarded unserved passenger needs one 'board' and one 'depart' action.
       - Each boarded unserved passenger needs one 'depart' action.
       - Total cost = (Number of unboarded unserved) + (Number of unserved).
    4. Identify the current floor of the lift.
    5. Determine the set of 'required floors' that the lift must visit:
       - For each unboarded unserved passenger, add their origin floor to the set.
       - For each boarded unserved passenger, add their destination floor (looked up from goals) to the set.
    6. Calculate the movement cost:
       - If the set of required floors is empty, the movement cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the required floors.
       - The estimated movement cost is the minimum distance from the current lift floor index to either the minimum or maximum required floor index, plus the distance between the minimum and maximum required floor indices. This estimates the travel needed to reach the range of required floors and traverse that range.
       - Movement cost = min(abs(current_floor_index - min_required_index), abs(current_floor_index - max_required_index)) + (max_required_index - min_required_index).
    7. The total heuristic value is the sum of the 'board'/'depart' action cost and the movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        super().__init__(task)

        # 1. Build floor_to_index map based on (above f_above f_below) facts
        direct_above_pairs = []
        all_floors = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_above, f_below = parts[1], parts[2]
                direct_above_pairs.append((f_above, f_below))
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Compute transitive closure of the 'above' relation
        is_above = {f: {f_other: False for f_other in all_floors} for f in all_floors}
        for f_above, f_below in direct_above_pairs:
            is_above[f_above][f_below] = True

        # Floyd-Warshall for transitive closure
        floor_list = list(all_floors) # Use a list for consistent iteration order
        for k in floor_list:
            for i in floor_list:
                for j in floor_list:
                    is_above[i][j] = is_above[i][j] or (is_above[i][k] and is_above[k][j])

        # Assign index to each floor based on the number of floors below it
        self.floor_to_index = {}
        for f in all_floors:
            count_below = sum(1 for f_below in all_floors if is_above[f][f_below])
            self.floor_to_index[f] = count_below

        # 2. Build passenger_to_dest map from (destin p d) facts in initial state/static
        self.passenger_to_dest = {}
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             if parts and parts[0] == "destin" and len(parts) == 3:
                 passenger, destination = parts[1], parts[2]
                 self.passenger_to_dest[passenger] = destination

        # 3. Get all passenger names from initial state and goals
        self.all_passengers = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ["origin", "destin", "boarded", "served"] and len(parts) > 1:
                 self.all_passengers.add(parts[1])
        for goal in task.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served" and len(parts) == 2:
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Get current lift floor
        current_floor_name = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_floor_name = parts[1]
                break

        # If lift location is not found, something is wrong with the state representation
        # or it's an unexpected terminal state. Assume valid states have lift-at.
        if current_floor_name is None:
             # This case should ideally not be reached in a solvable problem before the goal
             # If it happens, it implies an invalid state or potentially an unsolvable problem state
             # where the lift location is lost. Return infinity.
             # A goal state is handled by checking unserved_passengers first.
             return float('inf')

        current_floor_idx = self.floor_to_index.get(current_floor_name)
        # If current floor name is not in our precomputed index map, something is wrong
        if current_floor_idx is None:
             # This indicates a discrepancy between state facts and static/initial facts used for setup
             print(f"Error: Lift floor '{current_floor_name}' from state not found in floor index map.")
             return float('inf')


        # 2. Identify unserved, unboarded unserved, and boarded unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = {p for p in self.all_passengers if p not in served_passengers}

        if not unserved_passengers:
            return 0 # Goal state reached

        unboarded_unserved = set()
        boarded_unserved = set()
        passenger_origins = {} # Need origins for required floors calculation

        # Iterate through state facts to find current status of unserved passengers
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) == 3:
                p, o = parts[1], parts[2]
                if p in unserved_passengers:
                    unboarded_unserved.add(p)
                    passenger_origins[p] = o # Store origin for later

            elif parts and parts[0] == "boarded" and len(parts) == 2:
                p = parts[1]
                if p in unserved_passengers:
                    boarded_unserved.add(p)

        # Note: A passenger is either (origin p o) OR (boarded p) OR (served p).
        # The sets unboarded_unserved, boarded_unserved, and served_passengers should partition self.all_passengers.
        # We only care about unserved passengers here.
        # unserved_passengers = unboarded_unserved U boarded_unserved

        # 3. Calculate board/depart action cost
        # Each unboarded unserved needs board (1) + depart (1) = 2 actions
        # Each boarded unserved needs depart (1) action
        # Total cost = sum(2 for p in unboarded_unserved) + sum(1 for p in boarded_unserved)
        # This simplifies to:
        # Total cost = len(unboarded_unserved) + (len(unboarded_unserved) + len(boarded_unserved))
        # Total cost = len(unboarded_unserved) + len(unserved_passengers)
        board_depart_cost = len(unboarded_unserved) + len(unserved_passengers)


        # 4. Determine required floors
        required_floors = set()
        for p in unboarded_unserved:
            # Origin was stored in passenger_origins
            origin_floor = passenger_origins.get(p)
            if origin_floor: # Should always be found if in unboarded_unserved
                 required_floors.add(origin_floor)
            else:
                 # This implies an unboarded unserved passenger has no origin fact in state? Invalid state.
                 print(f"Error: Unboarded unserved passenger {p} has no origin fact in state.")
                 return float('inf')


        for p in boarded_unserved:
            # Destination is stored in self.passenger_to_dest
            dest_floor = self.passenger_to_dest.get(p)
            if dest_floor: # Should always be found if passenger exists
                 required_floors.add(dest_floor)
            else:
                 # This implies a boarded unserved passenger has no destination? Invalid setup.
                 print(f"Error: Boarded unserved passenger {p} has no destination defined.")
                 return float('inf')


        # 5. Calculate movement cost
        movement_cost = 0
        if required_floors:
            required_indices = []
            for f in required_floors:
                idx = self.floor_to_index.get(f)
                if idx is not None:
                    required_indices.append(idx)
                else:
                    # Required floor name not in our precomputed index map - indicates error
                    print(f"Error: Required floor '{f}' not found in floor index map.")
                    return float('inf')

            if required_indices: # Should be true if required_floors was not empty and all floors were indexed
                min_idx = min(required_indices)
                max_idx = max(required_indices)

                # Estimate moves to get to the range [min_idx, max_idx] and traverse it
                # min(dist to min_idx, dist to max_idx) + range_size
                movement_cost = min(abs(current_floor_idx - min_idx), abs(current_floor_idx - max_idx)) + (max_idx - min_idx)
            # else: required_indices is empty, movement_cost remains 0. This case should not happen
            # if required_floors was not empty and all floors were indexed.


        # 6. Total heuristic
        total_cost = board_depart_cost + movement_cost

        return total_cost
