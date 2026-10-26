from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconic21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    Estimates the number of actions needed to serve all passengers by considering:
    - Current elevator position
    - Passengers' origins (if not boarded)
    - Passengers' destinations
    - Required board/depart actions and floor movements

    # Assumptions
    - Floors are named 'f' followed by a numeric identifier (e.g., 'f1', 'f2').
    - The 'above' relations form a total order where each floor is above all higher-numbered floors.
    - Distance between floors is the absolute difference of their numeric identifiers.

    # Heuristic Initialization
    - Extracts 'destin' predicates from static facts to map each passenger to their destination floor.
    - Parses floor names to determine numeric identifiers for distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify current elevator floor from the state.
    2. Determine served, boarded, and unserved passengers.
    3. For each unserved passenger:
        a. If boarded: Calculate distance from current floor to destination + 1 (depart).
        b. If not boarded: Calculate distance from current floor to origin, then to destination + 2 (board and depart).
    4. Sum all calculated actions for the heuristic value.
    """

    def __init__(self, task):
        """Extract static information: destinations and floor numeric mappings."""
        self.destin = {}  # passenger -> destination floor
        self.floor_numbers = {}  # floor name -> numeric value

        # Process static facts to get 'destin' and 'above' relations
        for fact in task.static:
            parts = get_parts(fact)
            if match(fact, 'destin', '*', '*'):
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
                # Ensure floor is parsed
                if floor not in self.floor_numbers:
                    self.floor_numbers[floor] = int(floor[1:])
            elif match(fact, 'above', '*', '*'):
                floor1, floor2 = parts[1], parts[2]
                for f in [floor1, floor2]:
                    if f not in self.floor_numbers:
                        self.floor_numbers[f] = int(f[1:])

    def __call__(self, node):
        """Compute heuristic estimate for the current state."""
        state = node.state
        current_lift_floor = None
        served = set()
        boarded = set()
        origins = {}  # passenger -> origin floor

        # Extract current lift location, served, boarded, and origins
        for fact in state:
            parts = get_parts(fact)
            if match(fact, 'lift-at', '*'):
                current_lift_floor = parts[1]
            elif match(fact, 'served', '*'):
                served.add(parts[1])
            elif match(fact, 'boarded', '*'):
                boarded.add(parts[1])
            elif match(fact, 'origin', '*', '*'):
                passenger, floor = parts[1], parts[2]
                origins[passenger] = floor

        if current_lift_floor is None:
            return 0  # Should not happen if state is valid

        current_floor_num = self.floor_numbers.get(current_lift_floor, 0)
        total = 0

        # Process each unserved passenger
        for passenger in self.destin:
            if passenger in served:
                continue
            if passenger in boarded:
                # Boarded: need to move to destination and depart
                dest = self.destin[passenger]
                dest_num = self.floor_numbers.get(dest, 0)
                distance = abs(current_floor_num - dest_num)
                total += distance + 1  # depart action
            else:
                # Not boarded: check origin
                if passenger not in origins:
                    continue  # Should not happen for valid states
                origin = origins[passenger]
                origin_num = self.floor_numbers.get(origin, 0)
                dest = self.destin[passenger]
                dest_num = self.floor_numbers.get(dest, 0)
                # Distance to origin and then to destination
                distance_to_origin = abs(current_floor_num - origin_num)
                distance_origin_dest = abs(origin_num - dest_num)
                total += distance_to_origin + distance_origin_dest + 2  # board + depart

        return total
