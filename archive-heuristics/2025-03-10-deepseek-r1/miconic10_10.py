from heuristics.heuristic_base import Heuristic

class miconic10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the current state of each passenger (not boarded, boarded, served) and the elevator's current position. For each unserved passenger, the heuristic adds the necessary actions to board them, move to their destination, and depart them.

    # Assumptions
    - Each passenger has a single origin and destination floor.
    - The elevator can move between any two floors in one action if there's an 'above' relation.
    - The 'above' relations form a total order, allowing direct movement between any two floors.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Store this information in a dictionary for quick lookup during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the elevator's current floor from the state.
    2. For each passenger:
        a. If served, contribute 0.
        b. If boarded, check if the elevator is at the destination. If yes, add 1 (depart). If no, add 2 (move + depart).
        c. If not boarded, check origin and destination:
            i. If origin == destination, add 2 or 3 actions based on elevator's current position.
            ii. Otherwise, add 3 or 4 actions based on whether the elevator is at the origin.
    3. Sum the costs for all passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting passenger info from static facts."""
        self.passenger_info = {}

        # Extract origin and destination for each passenger from static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'origin' and len(parts) == 3:
                passenger = parts[1]
                floor = parts[2]
                if passenger not in self.passenger_info:
                    self.passenger_info[passenger] = {}
                self.passenger_info[passenger]['origin'] = floor
            elif parts[0] == 'destin' and len(parts) == 3:
                passenger = parts[1]
                floor = parts[2]
                if passenger not in self.passenger_info:
                    self.passenger_info[passenger] = {}
                self.passenger_info[passenger]['destin'] = floor

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_floor = None
        served = set()
        boarded = set()

        # Extract current elevator position, served and boarded passengers
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'lift-at':
                current_floor = parts[1]
            elif parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'boarded':
                boarded.add(parts[1])

        if current_floor is None:
            # Elevator position must be present
            return float('inf')

        total = 0
        for passenger in self.passenger_info:
            if passenger in served:
                continue
            info = self.passenger_info[passenger]
            origin = info.get('origin', None)
            destin = info.get('destin', None)
            if origin is None or destin is None:
                continue  # Invalid passenger info

            if passenger in boarded:
                # Boarded: need to move to destination and depart
                if current_floor == destin:
                    total += 1  # depart
                else:
                    total += 2  # move + depart
            else:
                # Not boarded: handle origin and destination
                if origin == destin:
                    if current_floor == origin:
                        total += 2  # board + depart
                    else:
                        total += 3  # move + board + depart
                else:
                    if current_floor == origin:
                        total += 3  # board + move + depart
                    else:
                        total += 4  # move to origin + board + move to destin + depart
        return total
