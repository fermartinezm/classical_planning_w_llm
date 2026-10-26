from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque # For BFS

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    # Ensure the number of parts matches the number of args for a valid match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the estimated lift movement cost and the number of necessary board and depart actions.
    The movement cost is estimated as the total vertical distance the lift must traverse
    to cover the range of floors where pickups or dropoffs are required, starting from
    the current lift floor.

    # Assumptions
    - Floors are ordered linearly based on the 'above' predicate.
    - '(above f_lower f_higher)' means f_lower is immediately below f_higher.
    - Lift movement actions ('up', 'down') move between immediately adjacent floors.
    - Each 'board' and 'depart' action costs 1.
    - Each 'up' and 'down' action costs 1.
    - Passengers are independent regarding board/depart actions, but share the lift movement.

    # Heuristic Initialization
    - Parses static facts to determine the floor ordering and assign a level (integer) to each floor.
    - Parses static facts to store the destination floor for each passenger.
    - Identifies all passengers defined in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers who have not yet been served.
    3. If all passengers are served, the heuristic is 0.
    4. For each unserved passenger:
       - If the passenger is waiting at their origin floor:
         - Increment the count of necessary 'board' actions by 1.
         - Increment the count of necessary 'depart' actions by 1.
         - Add the passenger's origin floor and destination floor to the set of floors the lift must service.
       - If the passenger is boarded in the lift:
         - Increment the count of necessary 'depart' actions by 1.
         - Add the passenger's destination floor to the set of floors the lift must service.
    5. Determine the minimum and maximum floor levels among the floors that need servicing.
    6. Calculate the estimated movement cost: This is the vertical distance between the lowest floor the lift is currently at or needs to visit, and the highest floor the lift is currently at or needs to visit. This is `max(max_required_level, current_level) - min(min_required_level, current_level)`.
    7. The total heuristic value is the sum of the estimated movement cost, the total count of necessary 'board' actions, and the total count of necessary 'depart' actions.
    8. If there are unserved passengers but no service floors (meaning all unserved boarded passengers are already at the current lift floor which is their destination), the movement cost is 0, and the heuristic is just the number of necessary 'depart' actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels, passenger destinations,
        and the list of all passengers from the task definition.
        """
        self.goals = task.goals # Not strictly needed for this heuristic, but good practice

        # 1. Identify all floors and passengers
        self.all_floors = set()
        self.all_passengers = set()
        # Objects are typically listed in initial_state or facts depending on parser
        # Let's iterate through initial_state facts to find typed objects
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 2: # Assuming facts like (type object)
                 obj_type, obj_name = parts
                 if obj_type == 'floor':
                     self.all_floors.add(obj_name)
                 elif obj_type == 'passenger':
                     self.all_passengers.add(obj_name)

        # 2. Parse 'above' facts to build floor hierarchy (adjacency list)
        # Assuming (above f_lower f_higher) means f_lower is immediately below f_higher
        above_map = {} # {f_lower: f_higher}
        is_higher_floor = set() # Floors that are f_higher in some (above ...) fact
        for fact in task.static:
            if match(fact, "above", "*", "*"):
                f_lower, f_higher = get_parts(fact)[1:]
                above_map[f_lower] = f_higher
                is_higher_floor.add(f_higher)

        # 3. Find the lowest floor (a floor that is not a 'higher' floor in any 'above' fact)
        lowest_floor = None
        for floor in self.all_floors:
            if floor not in is_higher_floor:
                lowest_floor = floor
                break

        self.floor_levels = {}
        if lowest_floor:
            # Build levels using BFS starting from the lowest floor
            q = deque([(lowest_floor, 1)])
            visited = {lowest_floor}
            while q:
                current_f, level = q.popleft()
                self.floor_levels[current_f] = level

                next_f = above_map.get(current_f)
                if next_f and next_f in self.all_floors and next_f not in visited:
                    visited.add(next_f)
                    q.append((next_f, level + 1))

        # Fallback: If BFS didn't assign levels to all floors (e.g., disconnected graph, or no 'above' facts)
        # Assign levels based on sorting floor names. This is a weak fallback but provides *some* ordering.
        if len(self.floor_levels) != len(self.all_floors):
             unassigned_floors = sorted(list(self.all_floors), key=lambda f: (len(f), f)) # Sort by length then alpha
             # Clear potentially incomplete levels from BFS
             self.floor_levels = {}
             for i, floor in enumerate(unassigned_floors):
                 self.floor_levels[floor] = i + 1


        # 4. Parse 'destin' facts to store passenger destinations
        self.passenger_destinations = {}
        for fact in task.static:
            if match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1:]
                self.passenger_destinations[p] = f

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Find current lift floor
        current_f = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_f = get_parts(fact)[1]
                break

        # If lift location is unknown, heuristic is infinity (or a large number)
        # Assuming valid states always have lift-at.
        # If for some reason current_f is not found, we cannot compute movement cost.
        # Let's return a high value or handle gracefully. Returning 0 might lead to infinite loops.
        # A simple approach is to return a large number if current_f is missing or has no level.
        if current_f is None or current_f not in self.floor_levels:
             # This indicates an invalid state representation or initialization issue
             # In a real planner, this might be an error state. For heuristic, return high cost.
             return float('inf') # Or a large integer like 1000000


        # 2. Identify unserved passengers
        unserved_passengers = {p for p in self.all_passengers if '(served {})'.format(p) not in state}

        # 3. If all passengers are served, heuristic is 0
        if not unserved_passengers:
            return 0

        # 4. Collect necessary actions and service floors
        num_board = 0
        num_depart = 0
        service_floors = set() # Floors the lift must visit

        # Build dictionaries/sets for quick lookup of passenger states
        state_origins = {} # {p: origin_f} for waiting passengers in state
        state_boarded = set() # {p} for boarded passengers in state
        for fact in state:
             if match(fact, "origin", "*", "*"):
                 p, f = get_parts(fact)[1:]
                 if p in unserved_passengers: # Only care about unserved
                     state_origins[p] = f
             elif match(fact, "boarded", "*"):
                 p = get_parts(fact)[1]
                 if p in unserved_passengers: # Only care about unserved
                     state_boarded.add(p)


        for p in unserved_passengers:
            if p in state_origins: # Passenger is waiting at origin
                origin_f = state_origins[p]
                destin_f = self.passenger_destinations.get(p)
                if destin_f: # Should always exist for unserved passengers
                    num_board += 1
                    num_depart += 1
                    service_floors.add(origin_f)
                    service_floors.add(destin_f)

            elif p in state_boarded: # Passenger is boarded
                 destin_f = self.passenger_destinations.get(p)
                 if destin_f: # Should always exist for unserved passengers
                    num_depart += 1
                    service_floors.add(destin_f)

            # else: Unserved passenger is neither waiting nor boarded.
            # This shouldn't happen in a valid state for the miconic domain.
            # We ignore such passengers for heuristic calculation.

        # 5. Calculate movement cost
        movement_cost = 0
        if service_floors:
            # Ensure all service floors have a level assigned (fallback handles this)
            valid_service_floors = {f for f in service_floors if f in self.floor_levels}
            if valid_service_floors:
                min_req_level = min(self.floor_levels[f] for f in valid_service_floors)
                max_req_level = max(self.floor_levels[f] for f in valid_service_floors)
                current_level = self.floor_levels[current_f] # We already checked current_f has a level

                # Movement cost is the distance to cover the range [min_req_level, max_req_level]
                # starting from current_level.
                movement_cost = max(max_req_level, current_level) - min(min_req_level, current_level)
            # else: service_floors exist but none have levels (fallback failed or no floors defined)
            # movement_cost remains 0

        # 6. Total heuristic value
        h = movement_cost + num_board + num_depart

        return h
