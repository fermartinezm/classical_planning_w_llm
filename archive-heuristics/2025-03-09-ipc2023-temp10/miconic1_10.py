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


class miconic1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's location.
    It considers boarding, departing, and moving the elevator.

    # Assumptions
    - Each unboarded passenger needs to board the elevator.
    - Each boarded passenger needs to depart at their destination.
    - The elevator needs to move to the origin floor of each unboarded passenger and to the destination floor of each boarded passenger.
    - The heuristic ignores the capacity of the elevator.

    # Heuristic Initialization
    - Store the destination floor for each passenger.
    - Store the 'above' relationships between floors to calculate the number of moves between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the current elevator location.
    3. Identify passengers who are waiting to board (origin predicate).
    4. Identify passengers who are already on board (boarded predicate).
    5. For each waiting passenger:
       - Calculate the cost to move the elevator to the passenger's origin floor.
       - Add 1 (for the board action) to the heuristic value.
    6. For each boarded passenger:
       - Calculate the cost to move the elevator to the passenger's destination floor.
       - Add 1 (for the depart action) to the heuristic value.
    7. If all passengers are served, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger.
        - 'above' relationships between floors.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store the destination floor for each passenger.
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                destination = parts[2]
                self.passenger_destinations[passenger] = destination

        # Store the 'above' relationships between floors.
        self.above = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                self.above.add((parts[1], parts[2]))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        heuristic_value = 0

        # Check if the goal is reached
        if all(goal in state for goal in self.goals):
            return 0

        # Extract the current elevator location.
        elevator_location = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                elevator_location = get_parts(fact)[1]
                break

        if elevator_location is None:
            return float('inf')  # No elevator location found

        # Identify passengers who are waiting to board (origin predicate).
        waiting_passengers = []
        for fact in state:
            if match(fact, "origin", "*", "*"):
                waiting_passengers.append(get_parts(fact)[1])

        # Identify passengers who are already on board (boarded predicate).
        boarded_passengers = []
        for fact in state:
            if match(fact, "boarded", "*"):
                boarded_passengers.append(get_parts(fact)[1])

        # For each waiting passenger:
        for passenger in waiting_passengers:
            # Calculate the cost to move the elevator to the passenger's origin floor.
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor = get_parts(fact)[2]
                    break

            if origin_floor is None:
                return float('inf')  # No origin floor found for passenger

            heuristic_value += self.floor_distance(elevator_location, origin_floor)

            # Add 1 (for the board action) to the heuristic value.
            heuristic_value += 1

        # For each boarded passenger:
        for passenger in boarded_passengers:
            # Calculate the cost to move the elevator to the passenger's destination floor.
            destination_floor = self.passenger_destinations.get(passenger)

            if destination_floor is None:
                return float('inf')  # No destination floor found for passenger

            heuristic_value += self.floor_distance(elevator_location, destination_floor)

            # Add 1 (for the depart action) to the heuristic value.
            heuristic_value += 1

        return heuristic_value

    def floor_distance(self, start_floor, end_floor):
        """
        Calculates the number of moves required to go from start_floor to end_floor.
        """
        if start_floor == end_floor:
            return 0

        # Find the shortest path using BFS
        queue = [(start_floor, 0)]
        visited = {start_floor}

        while queue:
            floor, distance = queue.pop(0)

            if floor == end_floor:
                return distance

            # Check floors above
            for f1, f2 in self.above:
                if f1 == floor and f2 not in visited:
                    queue.append((f2, distance + 1))
                    visited.add(f2)
            # Check floors below
            for f1, f2 in self.above:
                if f2 == floor and f1 not in visited:
                    queue.append((f1, distance + 1))
                    visited.add(f1)

        return float('inf')  # No path found
