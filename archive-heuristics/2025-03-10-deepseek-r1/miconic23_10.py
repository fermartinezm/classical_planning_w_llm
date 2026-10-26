from heuristics.heuristic_base import Heuristic

class Miconic23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the current elevator position, boarded passengers, and the remaining steps to board and depart each unserved passenger.

    # Assumptions
    - Each passenger must be boarded at their origin floor and departed at their destination.
    - Moving between any two floors takes one action (up or down), as the 'above' relation allows direct movement.
    - Boarding and departing each take one action.
    - The elevator can handle passengers in any order, but the heuristic sums individual costs without considering path optimizations.

    # Heuristic Initialization
    - Extracts the origin and destination floors for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
        a. If not boarded, calculate the steps to move to their origin (if needed), then to their destination.
        b. Add 2 actions for boarding and departing.
        c. If already boarded, calculate steps to move to their destination (if needed) and add 1 action for departing.
    2. Sum all steps and actions for a total estimated cost.
    """

    def __init__(self, task):
        self.origin = {}
        self.destin = {}
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                self.origin[passenger] = floor
            elif parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor

    def __call__(self, node):
        state = node.state
        current_lift = None
        served = set()
        boarded = set()

        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'lift-at':
                current_lift = parts[1]
            elif parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'boarded':
                boarded.add(parts[1])

        total = 0

        for passenger in self.origin:
            if passenger in served:
                continue
            if passenger in boarded:
                dest = self.destin[passenger]
                if current_lift != dest:
                    total += 1  # Move to destination
                total += 1  # Depart action
            else:
                origin = self.origin[passenger]
                dest = self.destin[passenger]
                if current_lift != origin:
                    total += 1  # Move to origin
                if origin != dest:
                    total += 1  # Move to destination
                total += 2  # Board and depart actions

        return total
