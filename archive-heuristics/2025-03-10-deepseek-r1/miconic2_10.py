from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconic2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    Estimates the number of actions required to serve all passengers by considering each passenger's state (boarded or not) and the elevator's current position. The heuristic calculates the minimal actions needed for each passenger to reach their destination, assuming each floor movement takes one action.

    # Assumptions
    - Each passenger's origin and destination are static.
    - Elevator movement between any two floors takes one action if permitted by the 'above' hierarchy.
    - Boarding and departing each passenger require one action each.
    - Overestimates by treating each passenger's path independently, which may count overlapping movements multiple times.

    # Heuristic Initialization
    - Extracts origin and destination floors for each passenger from static facts.
    - Stores these in dictionaries for quick lookup during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger:
        a. Skip if already served.
        b. If boarded, check if elevator is at destination to determine actions needed.
        c. If not boarded, check elevator's current position relative to the passenger's origin and destination to compute required actions.
    2. Sum actions for all unserved passengers to get the heuristic value.
    """

    def __init__(self, task):
        self.origin = {}  # Maps passenger to origin floor
        self.destin = {}  # Maps passenger to destination floor

        # Extract origin and destination from static facts
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'origin':
                passenger, floor = parts[1], parts[2]
                self.origin[passenger] = floor
            elif parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor

    def __call__(self, node):
        state = node.state
        total = 0

        # Determine current elevator floor
        current_floor = next((fact[1:-1].split()[1] for fact in state if fact.startswith('(lift-at ')), None)

        for passenger in self.origin:
            if f'(served {passenger})' in state:
                continue  # Passenger already served

            if f'(boarded {passenger})' in state:
                # Passenger is boarded, need to depart
                dest = self.destin[passenger]
                if current_floor == dest:
                    total += 1  # Depart action
                else:
                    total += 2  # Move to destination and depart
            else:
                # Passenger is not boarded
                origin = self.origin[passenger]
                dest = self.destin[passenger]
                if current_floor == origin:
                    total += 1  # Board action
                    if origin != dest:
                        total += 2  # Move to destination and depart
                    else:
                        total += 1  # Depart action
                else:
                    total += 2  # Move to origin and board
                    if origin != dest:
                        total += 2  # Move to destination and depart
                    else:
                        total += 1  # Depart action

        return total
