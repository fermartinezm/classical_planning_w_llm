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


class miconic14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It considers the number of passengers who are not yet served, the number of
    passengers who are boarded but not yet at their destination, and the distance
    the lift needs to travel to pick up and drop off passengers.

    # Assumptions
    - The lift can carry any number of passengers.
    - The heuristic focuses on minimizing the number of lift movements and boarding/departing actions.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.
    - Create a data structure representing the 'above' relationships between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of passengers who are not yet served.
    2. Identify the current location of the lift.
    3. For each unserved passenger:
       - Determine the origin and destination floors.
       - Estimate the cost of moving the lift from its current location to the origin floor.
       - Add the cost of boarding the passenger.
       - Add the cost of moving the lift from the origin floor to the destination floor.
       - Add the cost of departing the passenger.
    4. Sum up the costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger.
        - 'Above' relationships between floors.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract destination floor for each passenger.
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                passenger = get_parts(fact)[1]
                destination = get_parts(fact)[2]
                self.passenger_destinations[passenger] = destination

        # Extract 'above' relationships between floors.
        self.above = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                floor1 = get_parts(fact)[1]
                floor2 = get_parts(fact)[2]
                self.above.add((floor1, floor2))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the state is a goal state.
        if self.goals <= state:
            return 0

        # Find the current location of the lift.
        lift_location = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_location = get_parts(fact)[1]
                break

        if lift_location is None:
            return float('inf')  # No lift location found, unsolvable state.

        # Count unserved passengers and estimate the cost to serve them.
        unserved_passengers = 0
        total_cost = 0

        for passenger, destination in self.passenger_destinations.items():
            if f"(served {passenger})" not in state:
                unserved_passengers += 1

                # Check if the passenger is already boarded.
                if f"(boarded {passenger})" in state:
                    # Passenger is boarded, move to destination and depart.
                    total_cost += self.floor_distance(lift_location, destination)
                    total_cost += 1  # Depart action
                    lift_location = destination  # Update lift location
                else:
                    # Passenger is not boarded, move to origin, board, move to destination, and depart.
                    origin = None
                    for fact in state:
                        if match(fact, "origin", passenger, "*"):
                            origin = get_parts(fact)[2]
                            break

                    if origin is None:
                        continue  # Passenger has no origin, skip

                    total_cost += self.floor_distance(lift_location, origin)
                    total_cost += 1  # Board action
                    total_cost += self.floor_distance(origin, destination)
                    total_cost += 1  # Depart action
                    lift_location = destination  # Update lift location

        return total_cost

    def floor_distance(self, start_floor, end_floor):
        """Estimate the number of moves required to travel between two floors."""
        distance = 0
        current = start_floor
        
        if current == end_floor:
            return 0

        # Simple but potentially inaccurate distance calculation.
        # It assumes that we can move directly between any two floors that have an 'above' relationship.
        # A more accurate calculation would require a graph search.
        
        # Check if end_floor is above start_floor
        above = False
        temp = current
        while temp != end_floor:
            found_next = False
            for f1, f2 in self.above:
                if f1 == temp:
                    temp = f2
                    distance += 1
                    found_next = True
                    break
            if not found_next:
                break
        if temp == end_floor:
            above = True
        
        # If end_floor is not above start_floor, check if start_floor is above end_floor
        if not above:
            distance = 0
            temp = current
            while temp != end_floor:
                found_next = False
                for f1, f2 in self.above:
                    if f2 == temp:
                        temp = f1
                        distance += 1
                        found_next = True
                        break
                if not found_next:
                    return float('inf')
            if temp != end_floor:
                return float('inf')

        return distance
