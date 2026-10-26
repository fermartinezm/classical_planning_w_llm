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
    in the Miconic elevator domain. It considers:
    - The current floor of the elevator.
    - The origin and destination floors of unserved passengers.
    - Whether passengers are already boarded.

    # Assumptions:
    - The elevator can move between floors in one action (up or down).
    - Boarding and departing passengers each take one action.
    - The heuristic does not need to be admissible (can overestimate).

    # Heuristic Initialization
    - Extract static information about passenger destinations and floor ordering.
    - Store goal conditions (all passengers must be served).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       - If not boarded:
         - Add cost to move elevator to passenger's origin floor.
         - Add cost to board the passenger (1 action).
       - If boarded:
         - Add cost to move elevator to passenger's destination floor.
         - Add cost to depart the passenger (1 action).
    2. Sum all costs for unserved passengers.
    3. The heuristic value is the total estimated actions required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        # Extract floor ordering from static facts
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
        total_cost = 0

        # Get current elevator floor
        current_floor = None
        for fact in state:
            if fact.startswith("(lift-at"):
                current_floor = get_parts(fact)[1]
                break

        if current_floor is None:
            return float("inf")  # Invalid state

        # Find all unserved passengers
        served_passengers = {
            get_parts(fact)[1] for fact in state if fact.startswith("(served")}
        all_passengers = {
            get_parts(fact)[1] for fact in state if fact.startswith("(origin")}
        unserved_passengers = all_passengers - served_passengers

        for passenger in unserved_passengers:
            # Check if passenger is already boarded
            boarded = any(
                fact.startswith(f"(boarded {passenger}") for fact in state)

            if boarded:
                # Need to go to destination and depart
                dest_floor = self.destinations[passenger]
                if current_floor != dest_floor:
                    total_cost += 1  # Move to destination floor
                total_cost += 1  # Depart action
            else:
                # Need to go to origin and board
                origin_floor = None
                for fact in state:
                    if fact.startswith(f"(origin {passenger}"):
                        origin_floor = get_parts(fact)[2]
                        break
                
                if origin_floor is None:
                    return float("inf")  # Invalid state

                if current_floor != origin_floor:
                    total_cost += 1  # Move to origin floor
                total_cost += 1  # Board action

        return total_cost
