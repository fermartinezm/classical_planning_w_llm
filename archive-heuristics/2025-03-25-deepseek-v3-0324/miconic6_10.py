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
    2. The origin and destination floors of unserved passengers
    3. Whether passengers are already boarded
    4. The floor hierarchy (above relations)

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor before being served
    - The 'above' relations form a complete ordering of floors
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract static information about destinations and floor hierarchy
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not boarded:
          - Add cost to move elevator to origin floor
          - Add boarding action
       b) Add cost to move elevator to destination floor
       c) Add departing action
    2. Optimize by:
       - Grouping passengers with same origin/destination floors
       - Considering current elevator position
       - Accounting for already boarded passengers
    3. The total heuristic is the sum of:
       - All required elevator movements (up/down between floors)
       - All boarding and departing actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        # Extract floor hierarchy from static facts
        self.above_relations = set()

        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                floor1, floor2 = parts[1], parts[2]
                self.above_relations.add((floor1, floor2))

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

        # If no elevator position found, return high cost
        if current_floor is None:
            return float('inf')

        # Track served passengers
        served_passengers = set()
        boarded_passengers = set()
        origin_floors = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "served", "*"):
                served_passengers.add(parts[1])
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(parts[1])
            elif match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                origin_floors[passenger] = floor

        # For each passenger that needs to be served
        for passenger in self.destinations:
            if passenger in served_passengers:
                continue

            dest_floor = self.destinations[passenger]

            if passenger in boarded_passengers:
                # Passenger is already boarded, just need to go to destination
                cost = self._movement_cost(current_floor, dest_floor)
                total_cost += cost + 1  # +1 for depart action
                current_floor = dest_floor
            else:
                # Passenger needs to be picked up first
                origin_floor = origin_floors.get(passenger)
                if origin_floor is None:
                    continue  # Passenger already picked up but not boarded?

                # Move to origin floor
                cost = self._movement_cost(current_floor, origin_floor)
                total_cost += cost + 1  # +1 for board action
                current_floor = origin_floor

                # Then move to destination floor
                cost = self._movement_cost(current_floor, dest_floor)
                total_cost += cost + 1  # +1 for depart action
                current_floor = dest_floor

        return total_cost

    def _movement_cost(self, from_floor, to_floor):
        """Estimate the number of actions needed to move between two floors."""
        if from_floor == to_floor:
            return 0

        # In the worst case, we might need to go through all intermediate floors
        # Since we don't have to be admissible, we'll use Manhattan distance
        # as a simple approximation
        return abs(int(from_floor[1:]) - int(to_floor[1:]))
