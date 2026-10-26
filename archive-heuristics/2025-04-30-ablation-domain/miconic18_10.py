from fnmatch import fnmatch
import re
from heuristics.heuristic_base import Heuristic

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
    # Simple check: ensure the number of parts is at least the number of args
    # and then check each part against the corresponding arg pattern.
    if len(parts) < len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def floor_name_to_index(floor_name):
    """Converts floor name 'fi' to integer index i."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid floor name format: {floor_name}")

def index_to_floor_name(index):
     """Converts integer index i to floor name 'fi'."""
     return f'f{index}'


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions and adds an estimate for the
    necessary vertical movement of the lift.

    # Assumptions
    - Floors are named 'f1', 'f2', ..., 'fn', where 'fi' is below 'fj' if i < j.
    - The lift can carry multiple passengers.
    - Actions are move-up, move-down, board, depart. Each costs 1.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from static facts.
    - Determines the mapping between floor names (f1, f2, ...) and integer indices (1, 2, ...).
    - Identifies the set of all passengers in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Categorize passengers based on the current state: waiting (at origin), boarded (in lift), or served (at destination).
    3. Identify the set of unserved passengers (all passengers minus served passengers).
    4. Count the number of waiting passengers (each requires a 'board' action).
    5. Count the number of unserved passengers (each requires a 'depart' action).
    6. Determine the set of floors the lift *must* visit to serve the unserved passengers:
       - Origin floors for all waiting passengers.
       - Destination floors for all boarded passengers.
       - Destination floors for all waiting passengers (they will need to be dropped off after boarding).
    7. Convert these required floor names into integer indices using the pre-calculated mapping.
    8. If there are no required floors to visit (and no unserved passengers), the heuristic is 0.
    9. If there are required floors, find the minimum and maximum floor indices among them.
    10. Estimate the number of move actions required to visit all floors within this range, starting from the current lift floor. This is calculated as the span of required floors (`max_idx - min_idx`) plus the minimum distance from the current floor to either end of the span (`min(abs(current_idx - min_idx), abs(current_idx - max_idx))`). This formula estimates the total vertical distance covered in a single sweep that visits all required floors in the range.
    11. The total heuristic value is the sum of the number of waiting passengers, the number of unserved passengers, and the estimated number of move actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and floor mapping.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations and identify all passengers
        self.destin_map = {}
        self.all_passengers = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destin_map[passenger] = floor
                self.all_passengers.add(passenger)

        # Extract floor mapping (assuming f1, f2, ... fn)
        # Find all floors mentioned in static facts that match the 'f\d+' pattern
        all_floors = set()
        for fact in static_facts:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('f') and re.fullmatch(r'f\d+', part):
                     all_floors.add(part)

        # Sort floors based on index (assuming f1 < f2 < ...)
        sorted_floors = sorted(list(all_floors), key=floor_name_to_index)
        self.floor_to_index = {floor: i + 1 for i, floor in enumerate(sorted_floors)}
        self.index_to_floor = {i + 1: floor for i, floor in enumerate(sorted_floors)}
        self.max_floor_index = len(sorted_floors)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify the current floor of the lift.
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        if current_floor is None:
             # Should not happen in a valid miconic state, but handle defensively
             # Returning a large value discourages search towards such states.
             return float('inf')

        current_floor_idx = floor_name_to_index(current_floor)

        # 2. Categorize passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        waiting_passengers = {get_parts(fact)[1] for fact in state if match(fact, "origin", "*", "*")}

        # 3. Identify unserved passengers
        unserved_passengers = self.all_passengers - served_passengers

        # If no unserved passengers, goal is reached.
        if not unserved_passengers:
            return 0

        # 4. Count board actions needed
        num_board_actions = len(waiting_passengers)

        # 5. Count depart actions needed
        num_depart_actions = len(unserved_passengers)

        # 6. Determine required floors to visit
        required_floors = set()
        for p in unserved_passengers:
            if p in waiting_passengers:
                # Passenger is waiting, need to visit origin and destination
                # Find origin floor from state facts
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", p, "*"):
                        origin_floor = get_parts(fact)[2]
                        break
                if origin_floor: # Should always be found if in waiting_passengers
                    required_floors.add(origin_floor)
                # Need to visit destination after boarding
                required_floors.add(self.destin_map[p])
            elif p in boarded_passengers:
                # Passenger is boarded, need to visit destination
                required_floors.add(self.destin_map[p])
            # Passengers who are unserved but neither waiting nor boarded
            # might indicate an invalid state or a passenger whose origin fact
            # was removed without boarding. We assume valid states.

        # 7. Convert required floors to indices
        required_indices = sorted([floor_name_to_index(f) for f in required_floors])

        # 8. Calculate move estimate
        move_estimate = 0
        if required_indices:
            min_req_idx = required_indices[0]
            max_req_idx = required_indices[-1]

            # Estimate moves based on covering the span of required floors
            # and reaching the span from the current floor.
            # This is the shortest distance to visit all points in [min_req_idx, max_req_idx]
            # starting from current_floor_idx.
            move_estimate = (max_req_idx - min_req_idx) + min(abs(current_floor_idx - min_req_idx), abs(current_floor_idx - max_req_idx))

        # 11. Calculate total heuristic
        # The heuristic is the sum of required actions (board, depart) and estimated moves.
        # Each waiting passenger needs a board action.
        # Each unserved passenger needs a depart action.
        # The move estimate covers the vertical travel needed to reach the required floors.
        total_heuristic = num_board_actions + num_depart_actions + move_estimate

        return total_heuristic
