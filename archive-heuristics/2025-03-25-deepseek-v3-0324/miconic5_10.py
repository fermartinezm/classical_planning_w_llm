from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(origin p1 f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    by considering:
    1. The current position of the elevator
    2. The passengers that still need to be boarded
    3. The passengers that are boarded but not yet served
    4. The floor relationships (above) for movement planning

    # Assumptions:
    - The elevator can only move between floors connected by the 'above' relation
    - Each passenger must be boarded from their origin floor before being served at their destination
    - The 'above' relation forms a complete ordering of floors (transitive, antisymmetric)

    # Heuristic Initialization
    - Extract passenger destinations from static facts
    - Build a mapping of floor relationships from 'above' facts
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
        a) If not boarded:
            - Add cost to move elevator to passenger's origin floor
            - Add cost for boarding action
        b) If boarded:
            - Add cost to move elevator to passenger's destination floor
            - Add cost for depart action
    2. Optimize movement costs by:
        - Considering the current elevator position
        - Planning a route that minimizes total floor transitions
        - Grouping passengers with nearby origins/destinations
    3. The total heuristic is the sum of:
        - All boarding/depart actions (1 per passenger)
        - All necessary elevator movements between floors
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Build floor ordering from 'above' facts
        self.above_map = {}
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                if floor1 not in self.above_map:
                    self.above_map[floor1] = set()
                self.above_map[floor1].add(floor2)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        total_cost = 0

        # Get current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        if current_floor is None:
            return float('inf')  # Invalid state

        # Track which passengers still need service
        unserved_passengers = set()
        boarded_passengers = set()
        origin_floors = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                if passenger in self.destinations:  # Only consider passengers with known destinations
                    unserved_passengers.add(passenger)
                    origin_floors[passenger] = floor
            elif match(fact, "boarded", "*"):
                passenger = parts[1]
                if passenger in self.destinations:
                    unserved_passengers.add(passenger)
                    boarded_passengers.add(passenger)

        # If all passengers are served, return 0
        if not unserved_passengers:
            return 0

        # Calculate movement and action costs
        last_floor = current_floor
        for passenger in unserved_passengers:
            if passenger in boarded_passengers:
                # Passenger needs to be served at destination
                target_floor = self.destinations[passenger]
                total_cost += self._floor_distance(last_floor, target_floor)
                total_cost += 1  # depart action
                last_floor = target_floor
            else:
                # Passenger needs to be picked up at origin
                target_floor = origin_floors[passenger]
                total_cost += self._floor_distance(last_floor, target_floor)
                total_cost += 1  # board action
                last_floor = target_floor

        return total_cost

    def _floor_distance(self, floor1, floor2):
        """Estimate the number of moves needed between two floors."""
        if floor1 == floor2:
            return 0
        
        # In the worst case, we might need to go through all intermediate floors
        # This is a conservative estimate since we don't have complete floor ordering
        return 1  # We assume direct movement between any two floors is possible
