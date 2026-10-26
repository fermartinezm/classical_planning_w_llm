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
    A domain-dependent heuristic for the Miconic domain (elevator scheduling).

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    by considering:
    1. The current position of the elevator
    2. Passengers waiting to board (origin floors)
    3. Passengers already boarded needing to be served (destination floors)
    4. The floor hierarchy (above relations)

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor before being served at destination
    - The 'above' relations form a complete ordering of floors
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract static information about passenger destinations and floor hierarchy
    - Build a mapping of floor relationships for efficient path cost calculation

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not yet boarded:
          - Add cost to move from current floor to origin floor
          - Add boarding action (1)
          - Add cost to move from origin to destination floor
       b) If already boarded:
          - Add cost to move from current floor to destination floor
       c) Add departing action (1)
    2. For passengers already served, no actions are needed
    3. The total heuristic is the sum of all these actions
    4. Optimizations:
       - Group passengers by origin/destination floors to minimize elevator trips
       - Use floor hierarchy to calculate minimal path between floors
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.destinations[parts[1]] = parts[2]

        # Build floor hierarchy graph
        self.above_map = {}
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if parts[1] not in self.above_map:
                    self.above_map[parts[1]] = []
                self.above_map[parts[1]].append(parts[2])

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        total_cost = 0

        # Find current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        if current_floor is None:
            return float('inf')  # Invalid state

        # Process all passengers
        passengers = set()
        boarded_passengers = set()
        served_passengers = set()
        origin_floors = {}  # passenger -> origin floor

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                passengers.add(passenger)
                origin_floors[passenger] = floor
            elif match(fact, "boarded", "*"):
                passenger = parts[1]
                boarded_passengers.add(passenger)
                passengers.add(passenger)
            elif match(fact, "served", "*"):
                passenger = parts[1]
                served_passengers.add(passenger)
                passengers.add(passenger)

        # For each unserved passenger, calculate required actions
        for passenger in passengers:
            if passenger in served_passengers:
                continue  # Already served, no cost

            if passenger in boarded_passengers:
                # Passenger is boarded, need to go to destination
                dest = self.destinations[passenger]
                cost = self._floor_distance(current_floor, dest)
                total_cost += cost + 1  # depart action
            else:
                # Passenger needs to be picked up and delivered
                origin = origin_floors[passenger]
                dest = self.destinations[passenger]
                
                # Cost to go to origin floor
                cost = self._floor_distance(current_floor, origin)
                total_cost += cost + 1  # board action
                
                # Cost to go to destination floor
                cost = self._floor_distance(origin, dest)
                total_cost += cost + 1  # depart action

        return total_cost

    def _floor_distance(self, floor1, floor2):
        """
        Calculate the minimal number of elevator moves needed between two floors.
        Since floors are completely ordered, we can calculate this directly.
        """
        if floor1 == floor2:
            return 0

        # Check if floor1 is above floor2 in the hierarchy
        if floor2 in self.above_map.get(floor1, []):
            return 1  # directly above

        # Count how many floors are between them in the hierarchy
        # Since the 'above' relations form a complete ordering, we can do this
        # by finding the longest chain between them
        queue = [(floor1, 0)]
        visited = set()

        while queue:
            current, dist = queue.pop(0)
            if current == floor2:
                return dist
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.above_map.get(current, []):
                queue.append((neighbor, dist + 1))

        # If no path found (shouldn't happen in valid problems)
        return float('inf')
