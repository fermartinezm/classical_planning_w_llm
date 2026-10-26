from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(above f1 f2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconic22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    based on their current state (waiting, boarded) and the elevator's location.

    # Assumptions
    - Each passenger needs to board, potentially requires the elevator to move, and then depart.
    - The heuristic focuses on the minimum number of moves and boarding/departing actions.
    - It assumes the elevator can carry all boarded passengers to their destinations in one go.

    # Heuristic Initialization
    - Extracts the origin and destination floors for each passenger from the initial state and static facts.
    - Determines the 'above' relationships between floors to estimate movement costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify passengers who are not yet served.
    2. For each unserved passenger:
       - If the passenger is waiting (at their origin floor):
         - Calculate the cost to move the elevator to the passenger's origin floor (if necessary).
         - Add the cost of boarding the passenger.
         - Calculate the cost to move the elevator to the passenger's destination floor (if necessary).
         - Add the cost of departing the passenger.
       - If the passenger is boarded:
         - Calculate the cost to move the elevator to the passenger's destination floor (if necessary).
         - Add the cost of departing the passenger.
    3. Sum the costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Origin and destination floors for each passenger.
        - 'Above' relationships between floors.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = set()

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_destinations[parts[1]] = parts[2]
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                self.above.add((parts[1], parts[2]))

        for fact in task.initial_state:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_origins[parts[1]] = parts[2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        lift_at = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_at = get_parts(fact)[1]
                break

        served_passengers = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])

        boarded_passengers = set()
        for fact in state:
            if match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])

        unserved_passengers = set(self.passenger_destinations.keys()) - served_passengers

        total_cost = 0

        for passenger in unserved_passengers:
            if passenger in boarded_passengers:
                dest_floor = self.passenger_destinations[passenger]
                if lift_at != dest_floor:
                    total_cost += 1  # Move to destination
                total_cost += 1  # Depart
            else:
                origin_floor = self.passenger_origins.get(passenger)
                dest_floor = self.passenger_destinations[passenger]

                if origin_floor is None:
                    continue

                if lift_at != origin_floor:
                    total_cost += 1  # Move to origin
                total_cost += 1  # Board

                if lift_at != dest_floor:
                    total_cost += 1  # Move to destination
                total_cost += 1  # Depart

        if not unserved_passengers:
            return 0

        return total_cost
