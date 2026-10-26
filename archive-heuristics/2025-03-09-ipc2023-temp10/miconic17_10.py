from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It considers the number of passengers who are not yet served, the current
    location of the lift, and the origin and destination floors of each passenger.
    The heuristic prioritizes boarding passengers at the current lift location,
    moving the lift towards the destinations of boarded passengers, and
    serving passengers at their destination floors.

    # Assumptions:
    - Each passenger needs to board the lift at their origin floor, travel to
      their destination floor, and then depart the lift.
    - The lift can only be at one floor at a time.
    - The heuristic assumes that the lift will move to the origin floor of
      the passengers before they board, and to the destination floor after
      they board.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Store the 'above' relationships between floors to estimate movement costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of passengers who are not yet served.
    2. For each unserved passenger:
       a. If the passenger is not boarded:
          i. Estimate the cost to move the lift to the passenger's origin floor.
          ii. Add the cost of boarding the passenger.
       b. If the passenger is boarded:
          i. Estimate the cost to move the lift to the passenger's destination floor.
          ii. Add the cost of departing the passenger.
    3. The total heuristic value is the sum of these estimated costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = {}

        for fact in static_facts:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'origin':
                passenger = fact_parts[1]
                floor = fact_parts[2]
                self.passenger_origins[passenger] = floor
            elif fact_parts[0] == 'destin':
                passenger = fact_parts[1]
                floor = fact_parts[2]
                self.passenger_destinations[passenger] = floor
            elif fact_parts[0] == 'above':
                floor1 = fact_parts[1]
                floor2 = fact_parts[2]
                if floor1 not in self.above:
                    self.above[floor1] = []
                self.above[floor1].append(floor2)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        if self.goal_reached(state):
            return 0

        lift_location = None
        for fact in state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'lift-at':
                lift_location = fact_parts[1]
                break

        unserved_passengers = 0
        for goal in self.goals:
            if goal not in state:
                unserved_passengers += 1

        heuristic_value = 0
        for passenger in self.passenger_origins.keys():
            origin_fact = f'(origin {passenger} {self.passenger_origins[passenger]})'
            destin_fact = f'(destin {passenger} {self.passenger_destinations[passenger]})'
            boarded_fact = f'(boarded {passenger})'
            served_fact = f'(served {passenger})'

            if served_fact in state:
                continue

            if boarded_fact not in state:
                # Passenger not boarded: estimate cost to move lift to origin and board
                origin_floor = self.passenger_origins[passenger]
                if lift_location != origin_floor:
                    heuristic_value += self.estimate_move_cost(lift_location, origin_floor)
                heuristic_value += 1  # Board action
            else:
                # Passenger boarded: estimate cost to move lift to destination and depart
                destination_floor = self.passenger_destinations[passenger]
                if lift_location != destination_floor:
                    heuristic_value += self.estimate_move_cost(lift_location, destination_floor)
                heuristic_value += 1  # Depart action

        return heuristic_value

    def estimate_move_cost(self, start_floor, end_floor):
        """Estimate the cost to move the lift from start_floor to end_floor."""
        cost = 0
        current_floor = start_floor
        if start_floor == end_floor:
            return 0

        # Simple heuristic: count the number of floors between start and end
        floors = sorted(list(set(list(self.passenger_origins.values()) + list(self.passenger_destinations.values()))))
        start_index = floors.index(start_floor)
        end_index = floors.index(end_floor)
        cost = abs(start_index - end_index)

        return cost

    def goal_reached(self, state):
        """Check if all goals are reached in the given state."""
        return self.goals <= state
