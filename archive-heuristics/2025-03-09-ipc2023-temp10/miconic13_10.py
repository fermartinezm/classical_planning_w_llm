from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on the current state of the elevator and the passengers. It considers
    the number of passengers waiting at their origin floors, the number of
    passengers boarded on the elevator, and the distance the elevator needs to
    travel to pick up and drop off passengers.

    # Assumptions:
    - The elevator can carry any number of passengers.
    - The heuristic assumes that the elevator will serve passengers in an optimal order,
      minimizing the total travel distance.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.
    - Build a dictionary representing the 'above' relationships between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the elevator.
    2. Identify all passengers who are not yet served.
    3. Divide the unserved passengers into two groups: boarded and waiting.
    4. For each waiting passenger, determine their origin floor.
    5. For each boarded passenger, determine their destination floor.
    6. Calculate the minimum number of moves required to pick up all waiting passengers
       and drop off all boarded passengers. This involves calculating the distances
       between the current elevator location, the origin floors of waiting passengers,
       and the destination floors of boarded passengers.
    7. The heuristic value is the sum of the number of board actions, depart actions,
       and the number of up/down actions (elevator moves).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_destinations = {}
        for fact in static_facts:
            if "destin" in fact:
                parts = fact[1:-1].split()
                passenger = parts[1]
                destination = parts[2]
                self.passenger_destinations[passenger] = destination

        self.above = {}
        for fact in static_facts:
            if "above" in fact:
                parts = fact[1:-1].split()
                floor1 = parts[1]
                floor2 = parts[2]
                if floor1 not in self.above:
                    self.above[floor1] = []
                self.above[floor1].append(floor2)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the current state is the goal state.
        if self.goals <= state:
            return 0

        # Extract the current elevator location.
        elevator_location = None
        for fact in state:
            if "lift-at" in fact:
                elevator_location = fact[1:-1].split()[1]
                break

        # Identify unserved passengers.
        unserved_passengers = set()
        for passenger in self.passenger_destinations:
            served_predicate = "(served {})".format(passenger)
            if served_predicate not in state:
                unserved_passengers.add(passenger)

        # Divide unserved passengers into boarded and waiting.
        boarded_passengers = set()
        waiting_passengers = set()
        for passenger in unserved_passengers:
            boarded_predicate = "(boarded {})".format(passenger)
            if boarded_predicate in state:
                boarded_passengers.add(passenger)
            else:
                waiting_passengers.add(passenger)

        # Get origin floors of waiting passengers.
        passenger_origins = {}
        for fact in state:
            if "origin" in fact:
                parts = fact[1:-1].split()
                passenger = parts[1]
                origin = parts[2]
                if passenger in waiting_passengers:
                    passenger_origins[passenger] = origin

        # Get destination floors of boarded passengers.
        passenger_destinations_current = {}
        for passenger in boarded_passengers:
            passenger_destinations_current[passenger] = self.passenger_destinations[passenger]

        # Estimate the number of actions required.
        num_board = len(waiting_passengers)
        num_depart = len(boarded_passengers)
        num_moves = 0

        # Calculate the moves needed to pick up and drop off passengers.
        locations_to_visit = []
        locations_to_visit.append(elevator_location)
        for passenger, origin in passenger_origins.items():
            locations_to_visit.append(origin)
        for passenger, destination in passenger_destinations_current.items():
            locations_to_visit.append(destination)

        # A simple estimate is to visit each location in order.
        for i in range(len(locations_to_visit) - 1):
            floor1 = locations_to_visit[i]
            floor2 = locations_to_visit[i+1]
            num_moves += self.floor_distance(floor1, floor2)

        return num_board + num_depart + num_moves

    def floor_distance(self, floor1, floor2):
        """Estimates the distance between two floors based on 'above' relationships."""
        if floor1 == floor2:
            return 0

        # A simple estimate is the absolute difference in floor numbers.
        # This assumes that the floors are numbered sequentially.
        try:
            floor1_num = int(floor1[1:])  # Extract the number from floor name (e.g., f1 -> 1)
            floor2_num = int(floor2[1:])
            return abs(floor1_num - floor2_num)
        except:
            return 1 # If conversion fails, return a default distance of 1

