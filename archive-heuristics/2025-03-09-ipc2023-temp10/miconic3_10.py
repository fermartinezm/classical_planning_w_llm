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


class miconic3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers in the Miconic domain.
    It considers the number of passengers who are waiting to board, currently boarded, and the elevator's movements.

    # Assumptions
    - Each passenger needs to board the elevator at their origin floor.
    - Each passenger needs to depart the elevator at their destination floor.
    - The elevator needs to move between floors.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Store the 'above' relationships between floors to estimate elevator movement costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of passengers who are not yet served.
    2. For each unserved passenger:
        a. If the passenger is not boarded, estimate the cost to move the elevator to the passenger's origin floor and board them.
        b. If the passenger is boarded, estimate the cost to move the elevator to the passenger's destination floor and depart them.
    3. Sum the costs for all unserved passengers to get the heuristic value.
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
        self.above_relationships = set()

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_destinations[parts[1]] = parts[2]
            elif match(fact, "above", "*", "*"):
                self.above_relationships.add(get_parts(fact)[1:3])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get the current elevator location
        elevator_location = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                elevator_location = get_parts(fact)[1]
                break

        if elevator_location is None:
            return float('inf')  # No elevator location found, unsolvable

        # Passengers that still need to be served
        unserved_passengers = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                unserved_passengers.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                unserved_passengers.add(get_parts(fact)[1])

        if not unserved_passengers:
            # Check if all goals are met
            all_served = True
            for goal in self.goals:
                if goal not in state:
                    all_served = False
                    break
            if all_served:
                return 0  # All passengers served, goal reached

        total_cost = 0
        for passenger in unserved_passengers:
            # Check if the passenger is boarded
            is_boarded = False
            for fact in state:
                if match(fact, "boarded", passenger):
                    is_boarded = True
                    break

            if not is_boarded:
                # Passenger is not boarded, estimate cost to board
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        break
                if origin_floor is None:
                    continue  # Passenger has no origin

                # Estimate cost to move elevator to origin floor
                if elevator_location != origin_floor:
                    total_cost += self.estimate_move_cost(elevator_location, origin_floor)

                total_cost += 1  # Board action

            else:
                # Passenger is boarded, estimate cost to depart
                destination_floor = self.passenger_destinations.get(passenger)
                if destination_floor is None:
                    continue  # Passenger has no destination

                # Estimate cost to move elevator to destination floor
                if elevator_location != destination_floor:
                    total_cost += self.estimate_move_cost(elevator_location, destination_floor)

                total_cost += 1  # Depart action

        return total_cost

    def estimate_move_cost(self, start_floor, end_floor):
        """Estimates the cost to move the elevator between two floors."""
        cost = 0
        current_floor = start_floor
        if start_floor == end_floor:
            return 0

        # Determine direction
        direction = 1 if self.is_above(start_floor, end_floor) else -1

        while current_floor != end_floor:
            next_floor = self.get_next_floor(current_floor, direction)
            if next_floor is None:
                return float('inf')  # No path exists
            current_floor = next_floor
            cost += 1

        return cost

    def is_above(self, floor1, floor2):
        """Checks if floor1 is above floor2 based on the static 'above' relationships."""
        return (floor1, floor2) in self.above_relationships

    def get_next_floor(self, current_floor, direction):
        """Finds the next floor in the given direction based on 'above' relationships."""
        if direction == 1:  # Moving up
            for f1, f2 in self.above_relationships:
                if f2 == current_floor:
                    return f1
        else:  # Moving down
            for f1, f2 in self.above_relationships:
                if f1 == current_floor:
                    return f2
        return None
