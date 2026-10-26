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


class miconic12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's location.

    # Assumptions:
    - Each passenger needs to board, potentially requires elevator movement, and then depart.
    - The elevator can only be at one floor at a time.
    - The heuristic considers the number of passengers who still need to board and depart.

    # Heuristic Initialization
    - Extract the 'above' relationships between floors to determine movement costs.
    - Identify all passengers and their origin and destination floors.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Extract relevant information from the current state:
        -   The current location of the lift.
        -   The set of passengers who are waiting at their origin.
        -   The set of passengers who are boarded.
        -   The set of passengers who are served.

    2.  Calculate the cost for boarding passengers:
        -   For each passenger waiting at their origin, estimate the cost to move the lift to their origin floor (if necessary) and board them.

    3.  Calculate the cost for departing passengers:
        -   For each boarded passenger, estimate the cost to move the lift to their destination floor (if necessary) and depart them.

    4.  Sum the costs:
        -   The total heuristic value is the sum of the boarding and departing costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract 'above' relationships to determine floor order.
        self.above = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)[1], get_parts(fact)[2]
                if f1 not in self.above:
                    self.above[f1] = []
                self.above[f1].append(f2)

        # Build a dictionary of passenger origins and destinations.
        self.passenger_origins = {}
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                passenger, floor = get_parts(fact)[1], get_parts(fact)[2]
                self.passenger_destinations[passenger] = floor

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Get the current lift location.
        lift_location = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_location = get_parts(fact)[1]
                break

        # Get the set of passengers who are waiting at their origin.
        waiting_passengers = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                waiting_passengers.add(get_parts(fact)[1])

        # Get the set of boarded passengers.
        boarded_passengers = set()
        for fact in state:
            if match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])

        # Get the set of served passengers.
        served_passengers = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])

        # If the goal is reached, return 0.
        if self.goals <= state:
            return 0

        cost = 0

        # Calculate cost for boarding passengers.
        for passenger in waiting_passengers:
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor = get_parts(fact)[2]
                    break

            if origin_floor is not None:
                if lift_location != origin_floor:
                    cost += 1  # Cost to move the lift to the origin floor.
                cost += 1  # Cost to board the passenger.

        # Calculate cost for departing passengers.
        for passenger in boarded_passengers:
            destination_floor = self.passenger_destinations[passenger]

            if lift_location != destination_floor:
                cost += 1  # Cost to move the lift to the destination floor.
            cost += 1  # Cost to depart the passenger.

        return cost
