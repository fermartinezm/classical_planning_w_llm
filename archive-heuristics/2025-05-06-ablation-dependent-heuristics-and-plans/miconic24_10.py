from fnmatch import fnmatch
import re
from heuristics.heuristic_base import Heuristic

# Helper functions

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
    # Check if the number of parts matches the number of arguments, unless wildcards are used
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip with min length to handle cases where pattern is shorter than fact parts
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_floor_number(floor_name):
    """Extract the numerical part from a floor name like 'f1', 'f10'."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        # Use 0-indexed floor numbers if floor names start from 1
        # Assuming floor names are f1, f2, ...
        return int(match.group(1)) - 1
    # If floor names might be different, this needs adjustment.
    # For standard benchmarks, f\d+ is common.
    raise ValueError(f"Unexpected floor name format: {floor_name}")


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the number of 'board' and 'depart' actions needed for passengers
    who are not yet served, and adds an estimate for the lift movement actions.
    The movement cost is estimated based on the number of distinct floors the
    lift must visit for pickups and dropoffs, plus an initial move if the lift
    is not currently at one of these required floors.

    # Assumptions
    - Floor names follow the pattern 'f' followed by a number (e.g., f1, f2, f10),
      and these numbers define the floor order (f1 is the lowest, f2 is the next, etc.).
    - Move actions (up/down) cost 1 regardless of the number of floors traversed,
      provided the target floor is in the correct direction (up is to a higher floor,
      down is to a lower floor) and the `above` predicate holds.
    - Passengers are either at their origin, boarded, or served.

    # Heuristic Initialization
    - Parse static facts to store the destination floor for each passenger.
    - Parse static facts and initial state to identify all floor objects.
    - Create a mapping from floor name to a 0-indexed floor number based on
      the numerical part of the floor name. This mapping is used internally
      if needed, but the current movement heuristic doesn't strictly require it.
      However, collecting floor names ensures we know all possible floors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the goal state is reached (all passengers are served). If yes, the heuristic is 0.
    2. Identify the current floor of the lift.
    3. Identify all passengers who are not yet served.
    4. For each not-served passenger:
       - If the passenger is at their origin floor (`origin ?p ?f`), they need to be picked up. Count this as one 'board' action needed. Add their origin floor to the set of 'pickup floors'.
       - If the passenger is boarded (`boarded ?p`), they need to be dropped off. Count this as one 'depart' action needed. Add their destination floor (looked up from initialization) to the set of 'dropoff floors'.
    5. The total number of 'board' and 'depart' actions needed is the sum of passengers at origin and passengers boarded. This is the 'action cost'.
    6. Determine the set of 'service floors', which is the union of 'pickup floors' and 'dropoff floors'. These are the floors the lift must visit.
    7. Calculate the 'movement cost':
       - If there are no service floors (meaning all passengers are served or there are no passengers needing service), the movement cost is 0.
       - If there are service floors:
         - The lift needs to visit each distinct service floor. A simple estimate is the number of distinct service floors (`len(service_floors)`).
         - Additionally, if the lift is not currently at one of the service floors, it needs at least one move to reach the first service floor. Add 1 to the movement cost in this case.
       - So, movement cost = `len(service_floors) + 1` if `service_floors` is not empty and the lift is not at a service floor, otherwise `len(service_floors)`.
    8. The total heuristic value is the sum of the 'action cost' and the 'movement cost'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.task = task # Store task to check goal state

        # Store destination for each passenger
        self.destinations = {}
        # Collect all floor names to determine floor order (optional for this heuristic, but good practice)
        floor_names = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                p, f = parts[1], parts[2]
                self.destinations[p] = f
            # Collect floor names from 'above' facts
            if parts[0] == 'above':
                f1, f2 = parts[1], parts[2]
                floor_names.add(f1)
                floor_names.add(f2)

        # Also collect floor names from initial state facts like (lift-at fX), (origin pY fZ)
        # This ensures we get all floors even if not all are in 'above' or 'destin'
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ['lift-at', 'origin']:
                 # The floor is the last argument
                 if len(parts) > 1: # Ensure there is a last argument
                     floor_names.add(parts[-1])

        # Although the current heuristic doesn't strictly need floor numbers,
        # parsing them and sorting floors can be useful for other heuristics
        # or for debugging. Keep the logic but the map is not used in __call__.
        self.floor_to_num = {}
        if floor_names:
            try:
                sorted_floor_names = sorted(list(floor_names), key=get_floor_number)
                self.floor_to_num = {f: i for i, f in enumerate(sorted_floor_names)}
            except ValueError as e:
                 # If floor names don't match expected pattern, the map will be empty
                 print(f"Warning: Could not parse floor names numerically. Error: {e}")
                 self.floor_to_num = {} # Ensure it's an empty dict


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if the goal state is reached.
        if self.task.goal_reached(state):
            return 0

        # 2. Identify the current floor of the lift.
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        # If lift_floor is None, the state is likely invalid or terminal (goal)
        # In a valid non-goal state, lift-at must be true for exactly one floor.
        if lift_floor is None:
             return float('inf') # Should not happen in valid states

        # 3. Identify passengers not served and their status
        all_passengers = set(self.destinations.keys())
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        not_served_passengers = all_passengers - served_passengers

        num_at_origin = 0
        num_boarded = 0
        pickup_floors = set()
        dropoff_floors = set()

        for p in not_served_passengers:
            is_at_origin = False
            is_boarded = False
            # Check state for passenger status
            for fact in state:
                if match(fact, "origin", p, "*"):
                    origin_floor = get_parts(fact)[2]
                    pickup_floors.add(origin_floor)
                    num_at_origin += 1
                    is_at_origin = True
                    break # Found origin, passenger is not boarded
                if match(fact, "boarded", p):
                    # Passenger is boarded, needs dropoff at destination
                    if p in self.destinations: # Ensure destination is known
                        dropoff_floors.add(self.destinations[p])
                        num_boarded += 1
                        is_boarded = True
                        break
            # If a not-served passenger is neither at origin nor boarded, something is wrong.
            # This check is mostly for debugging unexpected state representations.
            if not is_at_origin and not is_boarded:
                 # This passenger is in an unknown state, likely unreachable or invalid
                 # Return infinity to prune this state
                 return float('inf')


        # 5. Calculate action cost (board/depart)
        # Each passenger at origin needs 1 board action.
        # Each passenger boarded needs 1 depart action.
        action_cost = num_at_origin + num_boarded

        # 6. Determine service floors
        # These are the floors the lift must visit for pickups or dropoffs
        service_floors = pickup_floors | dropoff_floors

        # 7. Calculate movement cost
        movement_cost = 0
        if service_floors:
            # Heuristic movement cost:
            # Count the number of distinct floors that need a service (pickup or dropoff).
            # Add 1 if the lift is not currently at one of these service floors,
            # representing the initial move to reach the first service floor.
            movement_cost = len(service_floors)
            if lift_floor not in service_floors:
                 movement_cost += 1

        # 8. Total heuristic
        h = action_cost + movement_cost

        return h
