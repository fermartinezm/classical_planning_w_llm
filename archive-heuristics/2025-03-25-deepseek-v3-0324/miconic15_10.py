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
    2. Which passengers still need to be picked up
    3. Which passengers are boarded but not yet served
    4. The floor relationships (above) for movement costs

    # Assumptions:
    - The elevator can only move between floors where (above ?f1 ?f2) is defined
    - Each passenger must be picked up from their origin floor before being served
    - Passengers can only be served at their destination floor
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract static information about floor relationships (above) and passenger destinations
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
        a) If not boarded:
            - Add cost to move elevator to passenger's origin floor
            - Add boarding action cost
            - Add cost to move from origin to destination floor
        b) If boarded:
            - Add cost to move elevator to passenger's destination floor
        c) Add departing action cost
    2. Optimize by considering:
        - Passengers on same floors can be grouped
        - Movement costs between floors are based on floor relationships
    3. The total heuristic is the sum of all required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract floor relationships (above)
        self.above_relations = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                self.above_relations.add((parts[1], parts[2]))

        # Extract passenger destinations
        self.passenger_destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_destinations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        current_floor = None
        passengers_to_serve = set()
        boarded_passengers = set()
        origin_floors = {}

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "lift-at", "*"):
                current_floor = parts[1]
            elif match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                origin_floors[passenger] = floor
                passengers_to_serve.add(passenger)
            elif match(fact, "boarded", "*"):
                passenger = parts[1]
                boarded_passengers.add(passenger)
            elif match(fact, "served", "*"):
                passenger = parts[1]
                if passenger in passengers_to_serve:
                    passengers_to_serve.remove(passenger)

        if not passengers_to_serve:
            return 0  # Goal reached

        total_cost = 0
        current_pos = current_floor

        # Process boarded passengers first (they're already in the elevator)
        for passenger in list(passengers_to_serve):
            if passenger in boarded_passengers:
                dest = self.passenger_destinations[passenger]
                # Cost to move to destination floor
                total_cost += self._get_move_cost(current_pos, dest)
                current_pos = dest
                # Cost for depart action
                total_cost += 1
                passengers_to_serve.remove(passenger)

        # Process remaining passengers (need to be picked up)
        for passenger in passengers_to_serve:
            origin = origin_floors[passenger]
            dest = self.passenger_destinations[passenger]
            
            # Cost to move to origin floor
            total_cost += self._get_move_cost(current_pos, origin)
            current_pos = origin
            # Cost for board action
            total_cost += 1
            # Cost to move to destination floor
            total_cost += self._get_move_cost(current_pos, dest)
            current_pos = dest
            # Cost for depart action
            total_cost += 1

        return total_cost

    def _get_move_cost(self, from_floor, to_floor):
        """Estimate the cost to move between two floors."""
        if from_floor == to_floor:
            return 0
        
        # Check if floors are directly connected
        if (from_floor, to_floor) in self.above_relations:
            return 1
        
        # For floors not directly connected, estimate based on floor numbering
        try:
            # Try to extract numeric part from floor names (e.g., f1, f2)
            from_num = int(from_floor[1:])
            to_num = int(to_floor[1:])
            return abs(from_num - to_num)
        except (ValueError, IndexError):
            # Fallback: assume floors are connected through intermediate floors
            return 2
