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
    in the Miconic elevator domain. It considers:
    - The current position of the elevator.
    - The origin and destination floors of unserved passengers.
    - Whether passengers are already boarded or need to be picked up.

    # Assumptions:
    - The elevator can move between floors in one action (up or down).
    - Boarding and departing each take one action per passenger.
    - The order of serving passengers affects the heuristic value (we prioritize efficiency).

    # Heuristic Initialization
    - Extract goal conditions (all passengers must be served).
    - Extract static information about floor relationships (above) and passenger destinations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unserved passengers (those without the 'served' predicate).
    2. For each unserved passenger:
       - If not boarded: 
         * Add cost to move elevator to origin floor (if not already there).
         * Add cost for boarding action.
       - If boarded:
         * Add cost to move elevator to destination floor (if not already there).
         * Add cost for departing action.
    3. Optimize the order of serving passengers to minimize elevator movements.
    4. Sum all actions required to serve all passengers.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract destin (destination floors) from static facts
        self.destin = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destin[passenger] = floor

        # Build a graph of floor relationships (above)
        self.above = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                self.above.add((floor1, floor2))

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        total_cost = 0

        # Current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # Passengers to serve (those not yet served)
        unserved_passengers = []
        boarded_passengers = set()
        origin_floors = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "served", "*"):
                continue  # Already served, no cost
            elif match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                origin_floors[passenger] = floor
                unserved_passengers.append(passenger)
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(parts[1])

        # For each unserved passenger, calculate required actions
        for passenger in unserved_passengers:
            if passenger in boarded_passengers:
                # Passenger is boarded - need to go to destination
                dest_floor = self.destin[passenger]
                if current_floor != dest_floor:
                    total_cost += 1  # Move to destination floor
                total_cost += 1  # Depart action
                current_floor = dest_floor
            else:
                # Passenger needs to be picked up
                origin_floor = origin_floors[passenger]
                if current_floor != origin_floor:
                    total_cost += 1  # Move to origin floor
                total_cost += 1  # Board action
                current_floor = origin_floor

        return total_cost
