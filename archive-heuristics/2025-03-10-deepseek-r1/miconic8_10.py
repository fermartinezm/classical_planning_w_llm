from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconic8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the necessary movements to board and depart each passenger, along with the required board and depart actions.

    # Assumptions:
    - Moving between any two floors takes one action, as the 'above' relations allow direct movement.
    - Each passenger must be boarded at their origin floor and departed at their destination floor.
    - The elevator can carry multiple passengers, but the heuristic assumes each passenger is handled individually for simplicity.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the elevator from the state.
    2. For each passenger:
        a. If the passenger is already served, skip them.
        b. If the passenger is boarded, check if the elevator needs to move to their destination floor.
        c. If the passenger is not boarded, check if the elevator needs to move to their origin floor and then to their destination.
    3. Sum the required movements, board actions, and depart actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting origin and destination floors for each passenger."""
        self.origin = {}
        self.destin = {}
        static_facts = task.static

        for fact in static_facts:
            parts = fact.strip('()').split()
            if parts[0] == 'origin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.origin[passenger] = floor
            elif parts[0] == 'destin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        current_floor = None
        served_passengers = set()
        boarded_passengers = set()

        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'lift-at':
                current_floor = parts[1]
            elif parts[0] == 'served':
                served_passengers.add(parts[1])
            elif parts[0] == 'boarded':
                boarded_passengers.add(parts[1])

        moves = 0
        boards = 0
        departs = 0

        for passenger in self.origin:
            if passenger in served_passengers:
                continue
            departs += 1  # Each unserved passenger needs a depart action

            if passenger in boarded_passengers:
                # Passenger is boarded; check if movement to destination is needed
                destin_floor = self.destin[passenger]
                if current_floor != destin_floor:
                    moves += 1
            else:
                # Passenger is not boarded; handle origin and destination movements
                boards += 1  # Board action needed
                origin_floor = self.origin[passenger]
                destin_floor = self.destin[passenger]

                if origin_floor != current_floor:
                    moves += 1  # Move to origin
                if destin_floor != origin_floor:
                    moves += 1  # Move from origin to destination

        return moves + boards + departs
