from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconic7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by calculating the minimal movement needed for each passenger based on their current state (boarded or not) and the elevator's current position. The movement cost between floors is computed using the 'above' hierarchy.

    # Assumptions
    - Each passenger's destination (destin) is static and known from the problem's static facts.
    - The 'above' facts form a total order, allowing calculation of floor heights and distances.
    - The elevator can move between any two floors directly, with the cost being the difference in their heights.

    # Heuristic Initialization
    - Extract 'destin' facts for each passenger from static data.
    - Compute floor heights based on the 'above' facts to determine distances between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the elevator's current floor from the state.
    2. For each passenger:
        a. If already served, skip.
        b. If boarded, calculate distance from current floor to destination and add 1 for depart.
        c. If not boarded, find their origin, calculate distance from current floor to origin and from origin to destination, then add 2 for board and depart.
    3. Sum all individual costs for the total heuristic value.
    """

    def __init__(self, task):
        """Extract destin and above facts from static information."""
        self.destin = {}  # {passenger: destination_floor}
        self.floor_heights = {}  # {floor: height}

        # Parse static facts to build destin and track all floors
        for fact in task.static:
            if match(fact, 'destin', '*', '*'):
                parts = get_parts(fact)
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
            elif match(fact, 'above', '*', '*'):
                parts = get_parts(fact)
                f1, f2 = parts[1], parts[2]
                self.floor_heights.setdefault(f1, 0)
                self.floor_heights.setdefault(f2, 0)

        # Compute floor heights based on 'above' facts
        # Height is the number of floors above this one
        for fact in task.static:
            if match(fact, 'above', '*', '*'):
                parts = get_parts(fact)
                f2 = parts[2]
                self.floor_heights[f2] += 1

    def __call__(self, node):
        state = node.state
        current_floor = None
        served = set()
        boarded = set()
        origins = {}

        # Extract current elevator position, served, boarded, and origins
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at' and len(parts) == 2:
                current_floor = parts[1]
            elif parts[0] == 'served' and len(parts) == 2:
                served.add(parts[1])
            elif parts[0] == 'boarded' and len(parts) == 2:
                boarded.add(parts[1])
            elif parts[0] == 'origin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                origins[passenger] = floor

        if not current_floor:
            return float('inf')  # Invalid state

        total = 0
        for passenger, destin in self.destin.items():
            if passenger in served:
                continue
            if passenger in boarded:
                # Boarded: distance from current to destin + depart
                dest_height = self.floor_heights[destin]
                curr_height = self.floor_heights[current_floor]
                total += abs(dest_height - curr_height) + 1
            else:
                # Not boarded: check origin
                if passenger not in origins:
                    continue  # Should not happen
                origin = origins[passenger]
                # Distance from current to origin and origin to destin
                curr_h = self.floor_heights[current_floor]
                origin_h = self.floor_heights[origin]
                dest_h = self.floor_heights[destin]
                total += abs(curr_h - origin_h) + abs(origin_h - dest_h) + 2
        return total
