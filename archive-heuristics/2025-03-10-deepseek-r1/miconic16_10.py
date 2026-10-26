from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Miconic16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the necessary movements, boardings, and departures. For each passenger, it calculates the cost based on their current state (not boarded, boarded, or served) and the elevator's current position.

    # Assumptions
    - The elevator can move between any two floors in one action if there's an 'above' relation.
    - Each passenger requires their own sequence of boarding and departing actions.
    - The heuristic does not account for optimal ordering of passengers to minimize movements.

    # Heuristic Initialization
    - Extracts the origin and destination floors for each passenger from the static facts.
    - Uses the 'origin' and 'destin' predicates to map passengers to their respective floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the elevator's current floor from the state.
    2. For each passenger:
        a. If already served, add 0 to the total cost.
        b. If boarded but not served:
            i. Calculate the cost to move to the destination floor (1 action if not already there).
            ii. Add the depart action (1 action).
        c. If not boarded:
            i. Calculate the cost to move to the origin floor (1 action if not already there).
            ii. Add the board action (1 action).
            iii. Calculate the cost to move to the destination floor (1 action if different from origin).
            iv. Add the depart action (1 action).
    3. Sum the costs for all passengers to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting origin and destination floors for each passenger."""
        self.origin = {}  # Maps passenger to origin floor
        self.destin = {}  # Maps passenger to destination floor

        # Process static facts to extract origin and destin predicates
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'origin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.origin[passenger] = floor
            elif parts[0] == 'destin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_lift_pos = None

        # Find the current elevator position
        for fact in state:
            if fact.startswith('(lift-at '):
                parts = fact[1:-1].split()
                current_lift_pos = parts[1]
                break

        if current_lift_pos is None:
            return 0  # Elevator position not found (invalid state)

        total_cost = 0

        # Process each passenger
        for passenger in self.origin:
            # Check if the passenger is already served
            if f'(served {passenger})' in state:
                continue

            # Check if the passenger is boarded
            boarded = f'(boarded {passenger})' in state

            origin_floor = self.origin.get(passenger)
            dest_floor = self.destin.get(passenger)

            if boarded:
                # Need to move to destination and depart
                move_cost = 0 if current_lift_pos == dest_floor else 1
                total_cost += move_cost + 1  # depart action
            else:
                # Need to move to origin, board, move to destination, depart
                move_to_origin = 0 if current_lift_pos == origin_floor else 1
                move_to_dest = 0 if origin_floor == dest_floor else 1
                total_cost += move_to_origin + 1 + move_to_dest + 1

        return total_cost
