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
    This heuristic estimates the number of actions required to serve all passengers
    by considering:
    - The current position of the elevator
    - The origin and destination floors of unserved passengers
    - Whether passengers are already boarded

    # Assumptions:
    - The elevator can move between any floors (given the 'above' relations)
    - Each passenger must be boarded from their origin floor and departed at their destination
    - The heuristic does not need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract static information about passenger destinations and floor ordering
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
        a) If not boarded:
            - Add cost to move elevator to origin floor
            - Add cost for boarding action
        b) Add cost to move elevator to destination floor
        c) Add cost for depart action
    2. Optimize by considering passengers that can be served along the way:
        - If multiple passengers have same origin/destination, only count movement once
        - If elevator is already at a floor where passengers can be boarded/departed,
          don't count movement cost
    3. The total heuristic is the sum of all required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        # Extract floor ordering from 'above' relations
        self.above_relations = set()

        for fact in self.static:
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
        total_cost = 0

        # Get current elevator position
        lift_at = None
        for fact in state:
            if fact.startswith("(lift-at"):
                lift_at = get_parts(fact)[1]
                break

        if lift_at is None:
            return float("inf")  # Invalid state

        # Track which passengers still need to be served
        unserved_passengers = []
        boarded_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                passenger = parts[1]
                if f"(served {passenger})" not in state:
                    unserved_passengers.append((passenger, parts[2], self.destinations[passenger]))
            elif parts[0] == "boarded":
                boarded_passengers.add(parts[1])

        if not unserved_passengers:
            return 0  # Goal state

        # We'll process passengers in an order that minimizes movement
        # For simplicity, we'll process them in order of boarding first, then others
        current_floor = lift_at
        remaining_passengers = []

        # First handle already boarded passengers
        for passenger, origin, destin in unserved_passengers:
            if passenger in boarded_passengers:
                # Need to go to destination floor and depart
                if current_floor != destin:
                    total_cost += 1  # Movement cost
                    current_floor = destin
                total_cost += 1  # Depart action
            else:
                remaining_passengers.append((passenger, origin, destin))

        # Then handle unboarded passengers
        for passenger, origin, destin in remaining_passengers:
            # Go to origin floor if not already there
            if current_floor != origin:
                total_cost += 1  # Movement cost
                current_floor = origin
            total_cost += 1  # Board action

            # Go to destination floor
            if current_floor != destin:
                total_cost += 1  # Movement cost
                current_floor = destin
            total_cost += 1  # Depart action

        return total_cost
