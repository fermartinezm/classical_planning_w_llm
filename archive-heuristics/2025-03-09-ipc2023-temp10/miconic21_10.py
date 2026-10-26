from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's location.

    # Assumptions:
    - Each passenger needs to board the lift, travel to their destination, and depart.
    - The elevator needs to move between floors to pick up and drop off passengers.
    - The heuristic focuses on the number of passengers yet to be served and the distance
      between the elevator and the passengers' origins and destinations.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Determine the set of passengers.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the elevator.
    2. Identify the passengers who are not yet served.
    3. For each unserved passenger:
       - If the passenger is not yet boarded:
         - Estimate the cost to move the elevator to the passenger's origin floor.
         - Add the cost of boarding the passenger.
       - If the passenger is boarded:
         - Estimate the cost to move the elevator to the passenger's destination floor.
         - Add the cost of departing the passenger.
    4. Sum the costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.passengers = set()

        for fact in static_facts:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'destin':
                passenger = fact_parts[1]
                destination = fact_parts[2]
                self.passenger_destinations[passenger] = destination
                self.passengers.add(passenger)

        for fact in static_facts:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'above':
                pass

        for fact in task.initial_state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'origin':
                passenger = fact_parts[1]
                origin = fact_parts[2]
                self.passenger_origins[passenger] = origin
                self.passengers.add(passenger)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        lift_location = None
        for fact in state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'lift-at':
                lift_location = fact_parts[1]
                break

        if lift_location is None:
            return float('inf')  # No lift location found, unsolvable

        unserved_passengers = set()
        for passenger in self.passengers:
            served = False
            for fact in state:
                fact_parts = fact[1:-1].split()
                if fact_parts[0] == 'served' and fact_parts[1] == passenger:
                    served = True
                    break
            if not served:
                unserved_passengers.add(passenger)

        heuristic_value = 0
        for passenger in unserved_passengers:
            boarded = False
            for fact in state:
                fact_parts = fact[1:-1].split()
                if fact_parts[0] == 'boarded' and fact_parts[1] == passenger:
                    boarded = True
                    break

            if not boarded:
                # Passenger is not yet boarded
                if passenger in self.passenger_origins:
                    origin_floor = self.passenger_origins[passenger]
                    if lift_location != origin_floor:
                        heuristic_value += 1  # Cost to move lift to origin
                    heuristic_value += 1  # Cost to board
            else:
                # Passenger is boarded
                if passenger in self.passenger_destinations:
                    destination_floor = self.passenger_destinations[passenger]
                    if lift_location != destination_floor:
                        heuristic_value += 1  # Cost to move lift to destination
                    heuristic_value += 1  # Cost to depart

        # Check if the goal is reached
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break

        if goal_reached:
            return 0

        return heuristic_value
