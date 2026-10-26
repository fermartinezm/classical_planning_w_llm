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


class miconic8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current state (waiting, boarded) and the elevator's location.
    It considers the number of boardings, departures, and elevator movements required.

    # Assumptions
    - Each passenger needs to board the elevator at their origin floor and depart at their destination floor.
    - The elevator needs to move between floors to pick up and drop off passengers.
    - The heuristic assumes that the elevator always takes the shortest path to the next floor.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Determine the 'above' relationships between floors to calculate movement costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify passengers who are not yet served.
    2. For each unserved passenger:
       - If the passenger is waiting at their origin floor, estimate the cost to move the elevator to that floor and board the passenger.
       - If the passenger is boarded, estimate the cost to move the elevator to their destination floor and depart the passenger.
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
        self.above_floors = {}

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_destinations[parts[1]] = parts[2]
            elif match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if parts[1] not in self.above_floors:
                    self.above_floors[parts[1]] = []
                self.above_floors[parts[1]].append(parts[2])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the goal is reached
        if all(goal in state for goal in self.goals):
            return 0

        # Get the current elevator location
        elevator_location = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                elevator_location = get_parts(fact)[1]
                break

        if elevator_location is None:
            return float('inf')  # Should not happen, but handle it to avoid errors

        # Identify unserved passengers
        unserved_passengers = []
        for passenger in self.passenger_destinations:
            if f"(served {passenger})" not in state:
                unserved_passengers.append(passenger)

        total_cost = 0
        for passenger in unserved_passengers:
            if f"(boarded {passenger})" not in state:
                # Passenger is waiting at their origin
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        break
                if origin_floor is None:
                    continue # Passenger already boarded, or origin not found

                # Estimate cost to move elevator to origin and board
                total_cost += self.estimate_movement_cost(elevator_location, origin_floor) + 1  # Move + Board
            else:
                # Passenger is boarded
                destination_floor = self.passenger_destinations[passenger]

                # Estimate cost to move elevator to destination and depart
                total_cost += self.estimate_movement_cost(elevator_location, destination_floor) + 1  # Move + Depart

        return total_cost

    def estimate_movement_cost(self, start_floor, end_floor):
        """Estimates the number of up/down actions required to move between floors."""
        if start_floor == end_floor:
            return 0

        # Simple heuristic: Count the number of floors between start and end
        # This assumes that the 'above' relations form a total order
        cost = 0
        current_floor = start_floor
        while current_floor != end_floor:
            found_next = False
            for higher_floor, lower_floors in self.above_floors.items():
                if current_floor == higher_floor:
                    if end_floor in lower_floors:
                        current_floor = end_floor
                        cost += 1
                        found_next = True
                        break
                    elif any(f in lower_floors for f in self.above_floors):
                        # Move down to the first floor that is below the current floor
                        current_floor = lower_floors[0]
                        cost += 1
                        found_next = True
                        break
            if not found_next:
                # Move up
                for higher_floor, lower_floors in self.above_floors.items():
                    if current_floor in lower_floors:
                        current_floor = higher_floor
                        cost += 1
                        found_next = True
                        break
            if not found_next:
                #No path found
                return float('inf')
        return cost
