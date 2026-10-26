from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
# In the actual planner environment, this import will be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            pass
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for malformed facts
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
    # The number of parts in the fact must be at least the number of arguments in the pattern.
    if len(parts) < len(args):
        return False

    # Check if each part matches the corresponding argument pattern.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of 'board' actions needed, the number of 'depart' actions
    needed, and an estimate of the lift movement cost.

    # Assumptions
    - Passengers are initially at their origin floor or boarded.
    - Passengers need to be transported from their origin to their destination.
    - The lift can carry multiple passengers.
    - Floors are ordered and can be mapped to integers based on 'above' facts.
    - The goal is to have all passengers 'served'.

    # Heuristic Initialization
    - Extracts all floor names and maps them to integers based on the 'above'
      facts and initial lift location. Assumes floors are named f1, f2, ...
      and ordered numerically. Includes a fallback to alphabetical sorting.
    - Extracts the destination floor for each passenger from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers who are not yet 'served'.
    2. If no passengers are unserved, the heuristic is 0 (goal state).
    3. Determine the current floor of the lift.
    4. For each unserved passenger:
       - Determine if the passenger is currently at their origin floor or boarded.
       - If the passenger is at their origin floor:
         - Increment the count of 'board' actions needed.
         - Add the passenger's origin floor (as integer) to the set of required pickup stops.
       - If the passenger is boarded:
         - Add the passenger's destination floor (as integer) to the set of required dropoff stops.
       - Every unserved passenger needs one 'depart' action eventually.
    5. Combine the required pickup and dropoff stops into a single sorted list of floors the lift must visit (as integers).
    6. Estimate the lift movement cost:
       - If there are no required stops, the movement cost is 0.
       - Otherwise, the movement cost is estimated as the distance from the current lift floor
         to the first required stop in the sorted list, plus the sum of distances
         between consecutive required stops in the sorted list. This estimates
         the cost of traversing all required floors sequentially.
    7. The total heuristic value is the sum of:
       - The total count of 'board' actions needed.
       - The total count of 'depart' actions needed (which is simply the number of unserved passengers).
       - The estimated lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        self.task = task # Store task for access to initial_state, static, goals

        # 1. Extract and map floors to integers
        floor_names = set()
        # Floors from above facts (static)
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                if len(parts) > 2:
                    floor_names.add(parts[1])
                    floor_names.add(parts[2])
        # Floor from initial lift location (initial_state)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "lift-at":
                 if len(parts) > 1:
                     floor_names.add(parts[1])

        # Sort floor names numerically (assuming f<number> format)
        # Handle potential errors if floor names don't follow f<number>
        sorted_floor_names = []
        if floor_names:
            try:
                # Attempt numerical sort
                sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
            except (ValueError, IndexError):
                 # Fallback: sort alphabetically if numerical parsing fails
                 sorted_floor_names = sorted(list(floor_names))
                 # print(f"Warning: Floor names not strictly in f<number> format. Sorting alphabetically: {sorted_floor_names}") # Optional warning


        self.floor_to_int = {f: i + 1 for i, f in enumerate(sorted_floor_names)}
        self.int_to_floor = {i + 1: f for i, f in enumerate(sorted_floor_names)}

        # 2. Extract passenger destinations
        self.destinations = {}
        # Destinations are typically in the initial state
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                if len(parts) > 2:
                    self.destinations[parts[1]] = parts[2]

        # Store all passenger names for easy iteration
        self.all_passengers = set(self.destinations.keys())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Identify unserved passengers
        # A passenger is unserved if the goal fact (served p) is not in the state
        U = {p for p in self.all_passengers if not (f"(served {p})" in state)}

        # 2. If U is empty, heuristic is 0 (goal state)
        if not U:
            return 0

        # 3. Determine current lift floor
        current_floor_fact = next((fact for fact in state if match(fact, "lift-at", "*")), None)
        if current_floor_fact is None:
             # This state should not be reachable in a valid miconic problem
             # where the lift always exists and is at some floor.
             # Return a large value indicating an invalid/unsolvable state from this point.
             return float('inf')

        C_floor = get_parts(current_floor_fact)[1]
        C_floor_int = self.floor_to_int.get(C_floor)
        if C_floor_int is None:
             # Should not happen if floor mapping is correct during init
             return float('inf')


        # 4. Identify required stops and count board actions
        pickup_stops_ints = set()
        dropoff_stops_ints = set()
        N_board = 0

        # Create a quick lookup for passenger states (origin or boarded)
        # and origin locations for unserved passengers
        unserved_passenger_locations = {} # {p: floor}
        boarded_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "origin" and len(parts) > 2 and parts[1] in U:
                 unserved_passenger_locations[parts[1]] = parts[2]
            elif parts[0] == "boarded" and len(parts) > 1 and parts[1] in U:
                 boarded_passengers.add(parts[1])

        for p in U:
            if p in unserved_passenger_locations: # Passenger is at origin
                origin_floor = unserved_passenger_locations[p]
                origin_floor_int = self.floor_to_int.get(origin_floor)
                if origin_floor_int is not None:
                    pickup_stops_ints.add(origin_floor_int)
                    N_board += 1
                else:
                     # Should not happen if floor mapping is correct
                     return float('inf')

            elif p in boarded_passengers: # Passenger is boarded
                dest_floor = self.destinations.get(p)
                if dest_floor:
                    dest_floor_int = self.floor_to_int.get(dest_floor)
                    if dest_floor_int is not None:
                        dropoff_stops_ints.add(dest_floor_int)
                    else:
                         # Should not happen if floor mapping is correct
                         return float('inf')
                else:
                     # Should not happen if destinations are correctly initialized
                     return float('inf')
            # else: # Passenger is unserved but neither at origin nor boarded
                  # This indicates an invalid state according to domain rules.
                  # Returning inf is appropriate.
                  # return float('inf') # This case is implicitly handled if p is not in U

        # 5. Calculate Movement_cost
        all_stops_needed_ints = sorted(list(pickup_stops_ints | dropoff_stops_ints))

        Movement_cost = 0
        if all_stops_needed_ints:
            # Cost to reach the first stop in the sorted list
            Movement_cost += abs(C_floor_int - all_stops_needed_ints[0])

            # Cost to traverse between consecutive stops in the sorted list
            for i in range(len(all_stops_needed_ints) - 1):
                Movement_cost += abs(all_stops_needed_ints[i] - all_stops_needed_ints[i+1])

        # 6. Calculate N_depart
        N_depart = len(U) # Every unserved passenger needs one depart action eventually.

        # 7. Total heuristic
        heuristic_value = N_board + N_depart + Movement_cost

        return heuristic_value
