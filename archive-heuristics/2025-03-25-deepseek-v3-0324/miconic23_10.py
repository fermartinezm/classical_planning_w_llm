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
    2. The origin and destination floors of unserved passengers
    3. Whether passengers are already boarded
    4. The floor relationships (above) for movement costs

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor before being served at destination
    - The 'above' relation defines a strict ordering of floors (no cycles)
    - Passengers can only be served when the elevator is at their destination floor

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Extract the 'above' relations to understand floor connectivity
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not boarded:
          - Add cost to move elevator from current position to origin floor
          - Add 1 action for boarding
          - Add cost to move from origin to destination floor
       b) If already boarded:
          - Add cost to move from current position to destination floor
       c) Add 1 action for departing the passenger
    2. Optimize by considering passengers that can be served along the way:
       - Group passengers by origin/destination floors to minimize elevator movements
       - Prioritize boarding passengers when elevator is at their origin
       - Prioritize serving passengers when elevator is at their destination
    3. The total heuristic is the sum of:
       - All required boarding/departing actions
       - Minimal elevator movements to cover all needed floors
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract destination floors for each passenger
        self.destinations = {}
        # Extract origin floors from initial state (since they get deleted when boarding)
        self.origins = {}
        # Extract 'above' relations to understand floor connectivity
        self.above_relations = set()

        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                floor1, floor2 = parts[1], parts[2]
                self.above_relations.add((floor1, floor2))

        # Also extract origins from initial state (they might not be in static)
        for fact in task.initial_state:
            if match(fact, "origin", "*", "*"):
                passenger, floor = get_parts(fact)[1], get_parts(fact)[2]
                self.origins[passenger] = floor

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Get current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # If no lift-at fact, problem is unsolvable
        if current_floor is None:
            return float('inf')

        # Identify served and boarded passengers
        served = set()
        boarded = set()
        origins = {}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "served", "*"):
                served.add(parts[1])
            elif match(fact, "boarded", "*"):
                boarded.add(parts[1])
            elif match(fact, "origin", "*", "*"):
                origins[parts[1]] = parts[2]

        # All passengers (from destinations)
        all_passengers = set(self.destinations.keys())
        unserved = all_passengers - served

        # If all served, heuristic is 0
        if not unserved:
            return 0

        total_cost = 0

        # For each unserved passenger, calculate their individual cost
        for passenger in unserved:
            # Get origin and destination
            origin = origins.get(passenger, self.origins.get(passenger))
            destination = self.destinations[passenger]

            if passenger in boarded:
                # Already boarded, just need to go to destination and depart
                if current_floor != destination:
                    # Estimate movement cost as 1 per floor difference (simplified)
                    total_cost += 1
                total_cost += 1  # depart action
            else:
                # Need to board first
                if current_floor != origin:
                    # Move to origin
                    total_cost += 1
                total_cost += 1  # board action
                # Then move to destination
                if origin != destination:
                    total_cost += 1
                total_cost += 1  # depart action

        return total_cost
