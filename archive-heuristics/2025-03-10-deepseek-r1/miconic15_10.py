from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconic15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering each passenger's current state (boarded or not) and the elevator's current position. For each unserved passenger, it calculates the steps needed to board them (if not boarded) and then depart them at their destination.

    # Assumptions
    - Each passenger requires boarding and departing actions unless already boarded.
    - The elevator can move between any two floors in a single action if they are connected via the 'above' relation.
    - Movement costs are counted per passenger, potentially overcounting when multiple passengers share the same path.

    # Heuristic Initialization
    - Extract each passenger's origin and destination from static facts.
    - These are fixed and do not change during planning.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the elevator's current floor from the state.
    2. For each passenger not yet served:
        a. If the passenger is boarded:
            i. Add 1 action to depart if the elevator is at the destination.
            ii. Add 1 movement action to reach the destination if not already there.
        b. If the passenger is not boarded:
            i. Add 1 movement action to reach the origin if not already there.
            ii. Add 1 action to board.
            iii. Add 1 movement action to reach the destination if different from origin.
            iv. Add 1 action to depart.
    3. Sum all calculated actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static origin and destination information."""
        self.origin = {}
        self.destin = {}
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                self.origin[passenger] = floor
            elif parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        current_floor = None
        served = set()
        boarded = set()

        # Extract current elevator position, served and boarded passengers
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'lift-at':
                current_floor = parts[1]
            elif parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'boarded':
                boarded.add(parts[1])

        total_cost = 0

        for passenger in self.origin:
            if passenger in served:
                continue
            if passenger in boarded:
                # Passenger is boarded, need to depart
                dest = self.destin[passenger]
                if current_floor != dest:
                    total_cost += 1  # Move to destination
                total_cost += 1  # Depart action
            else:
                # Passenger needs to be boarded
                origin = self.origin[passenger]
                dest = self.destin[passenger]
                if current_floor != origin:
                    total_cost += 1  # Move to origin
                total_cost += 1  # Board action
                if origin != dest:
                    total_cost += 1  # Move to destination
                total_cost += 1  # Depart action

        return total_cost
