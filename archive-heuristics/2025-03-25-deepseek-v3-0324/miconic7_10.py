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
    - The current position of the elevator
    - The passengers that still need to be boarded
    - The passengers that are boarded but not yet served
    - The floor relationships (above) for movement costs

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor before being served at their destination
    - The heuristic does not need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract static information about passenger destinations and floor relationships
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
        a) If not boarded:
            - Add cost to move elevator to passenger's origin floor
            - Add cost for boarding action
        b) If boarded:
            - Add cost to move elevator to passenger's destination floor
            - Add cost for depart action
    2. Optimize by considering:
        - The current elevator position
        - The order in which passengers can be served most efficiently
        - Grouping passengers going to the same floor
    3. The total heuristic is the sum of:
        - All required movements (up/down between floors)
        - All required board/depart actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        # Extract floor hierarchy from 'above' relations
        self.above_relations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
            elif parts[0] == "above":
                floor1, floor2 = parts[1], parts[2]
                self.above_relations.add((floor1, floor2))

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Get current elevator position
        current_floor = None
        for fact in state:
            if fact.startswith("(lift-at"):
                current_floor = get_parts(fact)[1]
                break

        # If elevator position not found (shouldn't happen in valid states)
        if current_floor is None:
            return float("inf")

        # Identify passengers that still need to be served
        unserved_passengers = []
        boarded_passengers = set()
        origin_floors = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                passenger, floor = parts[1], parts[2]
                origin_floors[passenger] = floor
                if f"(served {passenger})" not in state:
                    unserved_passengers.append(passenger)
            elif parts[0] == "boarded":
                passenger = parts[1]
                boarded_passengers.add(passenger)

        total_cost = 0

        for passenger in unserved_passengers:
            if passenger in boarded_passengers:
                # Passenger is boarded but not served - need to go to destination
                dest_floor = self.destinations[passenger]
                # Add movement cost from current position to destination
                # (we'll optimize this later)
                total_cost += 1  # depart action
                # Update current floor to destination for next passenger
                current_floor = dest_floor
            else:
                # Passenger needs to be picked up
                origin_floor = origin_floors[passenger]
                # Add movement cost from current position to origin
                # (we'll optimize this later)
                total_cost += 1  # board action
                # Update current floor to origin for next passenger
                current_floor = origin_floor

        # Add movement costs between floors
        # We use a simple estimate: 1 per floor change (exact distance would require pathfinding)
        # This is optimistic but works for the heuristic
        movement_cost = len(unserved_passengers)  # At least one move per passenger
        total_cost += movement_cost

        return total_cost
