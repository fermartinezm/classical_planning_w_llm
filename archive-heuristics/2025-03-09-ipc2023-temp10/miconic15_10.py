from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's location.
    It considers the number of board, depart, up, and down actions required.

    # Assumptions:
    - The elevator can only be at one floor at a time.
    - Passengers must board the elevator at their origin floor and depart at their destination floor.
    - The heuristic assumes the elevator will move directly to the origin or destination floor,
      ignoring intermediate floors. This may overestimate the cost.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Identify the initial elevator location.
    - Store the 'above' relationships between floors to estimate travel costs.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current elevator location.
    2.  For each passenger:
        a. Check if the passenger is served. If so, no actions are needed for that passenger.
        b. If the passenger is not boarded:
           i. Calculate the cost to move the elevator from its current location to the passenger's origin floor.
           ii. Add a board action cost.
        c. If the passenger is boarded:
           i. Calculate the cost to move the elevator from its current location to the passenger's destination floor.
           ii. Add a depart action cost.
    3.  Sum the costs for all passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = {}

        for fact in static_facts:
            fact_str = fact[1:-1]
            parts = fact_str.split()
            if parts[0] == 'destin':
                self.passenger_destinations[parts[1]] = parts[2]
            elif parts[0] == 'above':
                self.above[(parts[1], parts[2])] = True

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Extract elevator location
        elevator_location = None
        for fact in state:
            fact_str = fact[1:-1]
            parts = fact_str.split()
            if parts[0] == 'lift-at':
                elevator_location = parts[1]
                break

        if elevator_location is None:
            return float('inf')  # No elevator location found, unsolvable

        # Extract passenger information
        served_passengers = set()
        boarded_passengers = set()
        passenger_origins_current = {}

        for fact in state:
            fact_str = fact[1:-1]
            parts = fact_str.split()
            if parts[0] == 'served':
                served_passengers.add(parts[1])
            elif parts[0] == 'boarded':
                boarded_passengers.add(parts[1])
            elif parts[0] == 'origin':
                passenger_origins_current[parts[1]] = parts[2]

        total_cost = 0

        # Iterate through all passengers and estimate cost
        for passenger, destination in self.passenger_destinations.items():
            if passenger in served_passengers:
                continue  # Passenger already served

            if passenger not in boarded_passengers:
                # Passenger needs to board
                if passenger in passenger_origins_current:
                    origin = passenger_origins_current[passenger]
                    total_cost += self.estimate_move_cost(elevator_location, origin)
                    total_cost += 1  # Board action
                else:
                    # Passenger already boarded, but origin is not in current state
                    # This should not happen, but handle it to avoid errors
                    total_cost += 1 # Board action
            else:
                # Passenger needs to depart
                total_cost += self.estimate_move_cost(elevator_location, destination)
                total_cost += 1  # Depart action

        # Check if the goal is reached
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break

        if goal_reached:
            return 0

        return total_cost

    def estimate_move_cost(self, start_floor, end_floor):
        """Estimates the cost to move the elevator between two floors."""
        if start_floor == end_floor:
            return 0

        # Simplistic cost: number of floors to traverse
        cost = 0
        current = start_floor
        
        # Determine direction
        up = False
        down = False
        if (start_floor, end_floor) in self.above:
            down = True
        elif (end_floor, start_floor) in self.above:
            up = True
        else:
            # No direct above/below relationship, return high cost
            return 5

        if up:
            while current != end_floor:
                found_next = False
                for f1, f2 in self.above:
                    if f1 == current and (f2, end_floor) in self.above:
                        current = f2
                        cost += 1
                        found_next = True
                        break
                if not found_next:
                    return 5 #No path found
        elif down:
            while current != end_floor:
                found_next = False
                for f1, f2 in self.above:
                    if f2 == current and (end_floor, f1) in self.above:
                        current = f1
                        cost += 1
                        found_next = True
                        break
                if not found_next:
                    return 5 #No path found
        
        return cost
