from heuristics.heuristic_base import Heuristic

class miconic4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the necessary elevator movements, boardings, and departures. It assumes that each movement between floors takes one action, regardless of distance.

    # Assumptions
    - The elevator can move between any two floors in one action if there's an 'above' relationship.
    - Boarding and departing each passenger requires one action.
    - The order of visiting floors is optimized to minimize movements, requiring one action per unique floor visited.

    # Heuristic Initialization
    - Extracts the origin and destination floors for each passenger from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Current Elevator Position**: Determine the current floor of the elevator.
    2. **Unserved Passengers**: Identify passengers not yet served.
    3. **Required Stops**:
       - Collect origin floors for unserved passengers not yet boarded.
       - Collect destination floors for all unserved passengers.
    4. **Movement Actions**: Calculate the number of movements as the number of required stops minus one if the elevator is already at a required stop, otherwise the number of required stops.
    5. **Boarding and Departure Actions**: Each unboarded passenger requires a boarding action, and each unserved passenger requires a departure action.
    6. **Total Estimate**: Sum movement, boarding, and departure actions.
    """

    def __init__(self, task):
        self.origin = {}
        self.destin = {}
        static_facts = task.static
        for fact in static_facts:
            parts = fact[1:-1].split()
            if parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                self.origin[passenger] = floor
            elif parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor
        self.passengers = list(self.origin.keys())

    def __call__(self, node):
        state = node.state
        current_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_floor = fact[1:-1].split()[1]
                break
        unserved = [p for p in self.passengers if f'(served {p})' not in state]
        if not unserved:
            return 0
        required_origins = set()
        required_destinations = set()
        boarded_count = 0
        for p in unserved:
            if f'(boarded {p})' in state:
                boarded_count += 1
            else:
                required_origins.add(self.origin[p])
            required_destinations.add(self.destin[p])
        required_stops = required_origins.union(required_destinations)
        if current_floor in required_stops:
            movement_actions = max(len(required_stops) - 1, 0)
        else:
            movement_actions = len(required_stops)
        board_actions = len(unserved) - boarded_count
        depart_actions = len(unserved)
        return movement_actions + board_actions + depart_actions
