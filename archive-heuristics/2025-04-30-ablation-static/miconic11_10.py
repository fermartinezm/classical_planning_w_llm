from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            pass
        def __call__(self, node):
            pass


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
    arg_idx = 0
    part_idx = 0

    while arg_idx < len(args) and part_idx < len(parts):
        if args[arg_idx] == "*":
            # Wildcard matches the current part. Move to the next arg and next part.
            arg_idx += 1
            part_idx += 1
        elif fnmatch(parts[part_idx], args[arg_idx]):
            # Non-wildcard matches. Move to the next arg and next part.
            arg_idx += 1
            part_idx += 1
        else:
            # Non-wildcard mismatch.
            return False

    # After the loop, check if all args and parts were consumed correctly.
    # All non-wildcard args must have been matched.
    # If the last arg was a wildcard, it matches any remaining parts.
    # If the last arg was not a wildcard, we must have consumed all parts.

    # If we still have args left, they must all be wildcards for a match
    while arg_idx < len(args):
        if args[arg_idx] != "*":
            return False # Unmatched non-wildcard arg
        arg_idx += 1

    # If we still have parts left, the last arg must have been a wildcard
    if part_idx < len(parts) and (arg_idx == 0 or args[-1] != "*"):
         return False # Unmatched parts

    # If we reached here, it's a match
    return True


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    The heuristic estimates the number of actions needed to serve all passengers.
    It sums the number of 'board' actions needed (for waiting passengers),
    the number of 'depart' actions needed (for unserved passengers), and
    the number of unique floors the lift must visit for pickups or dropoffs.

    # Assumptions
    - Floors are linearly ordered, and 'above' facts define this order.
      (above f_above f_below) means f_above is immediately above f_below.
    - Each waiting passenger needs one 'board' and one 'depart' action.
    - Each boarded passenger needs one 'depart' action.
    - The lift must visit each floor where a pickup or dropoff is required.

    # Heuristic Initialization
    - Parses 'above' facts from the initial state to determine the floor order and assign levels to floors.
    - Parses 'destin' facts from the initial state to map each passenger to their destination floor.
    - Identifies the set of all passengers that need to be served from the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of all passengers that need to be served (from the goal).
    2. Count the number of unserved passengers (those in the goal set but not currently 'served').
    3. Count the number of waiting passengers (those with an 'origin' fact in the current state). This is the estimate for 'board' actions needed.
    4. The number of unserved passengers is the estimate for 'depart' actions needed.
    5. Identify the set of 'ServiceFloors':
       - Add the origin floor for every waiting passenger.
       - Add the destination floor for every boarded passenger who is unserved.
    6. Count the number of unique floors in the 'ServiceFloors' set. This is the estimate for movement/stop actions needed.
    7. The heuristic value is the sum of (number of waiting passengers) + (number of unserved passengers) + (number of unique service floors).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels, passenger destinations,
        and the set of passengers to be served.
        """
        self.task = task
        self.goal_passengers = set()
        for goal in self.task.goals:
            # Goal is (served ?p) for all passengers ?p
            if match(goal, "served", "*"):
                self.goal_passengers.add(get_parts(goal)[1])

        # Parse floor levels from 'above' facts in the initial state
        # (above f_above f_below) means f_above is immediately above f_below
        above_map = {} # maps floor_above -> floor_below
        all_floors = set()
        for fact in self.task.initial_state:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                above_map[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the bottom floor (a floor that is not a value in above_map)
        bottom_floor = None
        floors_that_are_below = set(above_map.values())
        for floor in all_floors:
             if floor not in floors_that_are_below:
                bottom_floor = floor
                break

        self.floor_levels = {}
        if not all_floors:
             pass # No floors defined
        elif bottom_floor is None:
             # This case indicates a potential issue with the 'above' facts
             # (e.g., cyclic or disconnected graph, or only one floor).
             # If only one floor, assign level 1.
             if len(all_floors) == 1:
                 self.floor_levels = {list(all_floors)[0]: 1}
             else:
                 # Fallback for unexpected 'above' structure: Assume f1, f2, ... order
                 # This might be incorrect if the problem uses different naming conventions
                 print("Warning: Could not determine bottom floor from 'above' facts. Assuming f1, f2, ... order.")
                 try:
                     sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
                     self.floor_levels = {f: i+1 for i, f in enumerate(sorted_floors)}
                 except ValueError:
                      print("Error: Could not parse floor names like f1, f2. Floor levels cannot be determined.")
                      # Assign arbitrary levels or raise error depending on desired robustness
                      # For this problem, we expect fN > f(N-1) > ... > f1 structure
                      pass # Leave floor_levels empty, heuristic might be less effective

        else:
            # Build levels using BFS starting from the bottom floor
            self.floor_levels[bottom_floor] = 1
            q = [bottom_floor]
            visited = {bottom_floor}

            while q:
                next_q = []
                floors_at_current_level = q
                q = [] # Clear queue for next level
                current_level = self.floor_levels[floors_at_current_level[0]] # Level of floors just processed

                # Find floors immediately above floors_at_current_level
                for f_above, f_below in above_map.items():
                    if f_below in floors_at_current_level and f_above not in visited:
                        self.floor_levels[f_above] = current_level + 1
                        visited.add(f_above)
                        q.append(f_above) # Add to queue for the next level

        # Parse passenger destinations from initial state
        self.passenger_destinations = {}
        for fact in self.task.initial_state:
            if match(fact, "destin", "*", "*"):
                passenger, destination = get_parts(fact)[1:]
                self.passenger_destinations[passenger] = destination

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.goal_passengers - served_passengers
        num_unserved = len(unserved_passengers)

        # If no unserved passengers, goal is reached
        if num_unserved == 0:
            return 0

        # 2. Count waiting passengers (estimate for board actions)
        waiting_passengers = {get_parts(fact)[1] for fact in state if match(fact, "origin", "*", "*")}
        num_waiting = len(waiting_passengers)

        # 3. Identify boarded unserved passengers
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        boarded_unserved_passengers = boarded_passengers.intersection(unserved_passengers)

        # 4. Identify ServiceFloors
        service_floors = set()
        # Origins of waiting passengers
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1:]
                service_floors.add(f)

        # Destinations of boarded unserved passengers
        for p in boarded_unserved_passengers:
             # We need the destination for this passenger
             destination = self.passenger_destinations.get(p)
             if destination: # Should always exist if passenger is in the problem
                 service_floors.add(destination)
             # else: This case indicates an issue with problem definition or parsing

        num_service_floors = len(service_floors)

        # Heuristic calculation:
        # Number of board actions needed (at least one per waiting passenger)
        # Number of depart actions needed (at least one per unserved passenger)
        # Number of stops needed (at least one per service floor)
        heuristic_value = num_waiting + num_unserved + num_service_floors

        return heuristic_value
