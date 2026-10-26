from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on the current state of the elevator and the passengers. It considers
    the number of passengers waiting to board, those already boarded, and the
    elevator's movements.

    # Assumptions:
    - Each passenger needs to board the elevator at their origin floor.
    - Each passenger needs to depart the elevator at their destination floor.
    - The elevator can serve multiple passengers at once.
    - The heuristic assumes that the elevator will move to the origin floor of each
      unboarded passenger and then to the destination floor of each boarded passenger.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Determine the initial elevator location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the lift.
    2. Identify passengers who have not yet boarded.
    3. Identify passengers who have boarded but not yet served.
    4. For each unboarded passenger, estimate the cost to move the lift to their origin floor and have them board.
    5. For each boarded passenger, estimate the cost to move the lift to their destination floor and have them depart.
    6. Sum up all the costs to get the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting passenger origins, destinations, and initial lift location."""
        self.goals = task.goals
        self.static = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.initial_lift_location = None

        for fact in self.static:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'destin':
                self.passenger_destinations[fact_parts[1]] = fact_parts[2]

        for fact in task.initial_state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'lift-at':
                self.initial_lift_location = fact_parts[1]

        for fact in task.initial_state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'origin':
                self.passenger_origins[fact_parts[1]] = fact_parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Check if the goal is reached
        if self.goal_reached(state):
            return 0

        lift_location = None
        for fact in state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'lift-at':
                lift_location = fact_parts[1]
                break

        unboarded_passengers = []
        boarded_passengers = []

        for fact in state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'origin':
                unboarded_passengers.append(fact_parts[1])
            elif fact_parts[0] == 'boarded':
                boarded_passengers.append(fact_parts[1])

        total_cost = 0

        # Cost for unboarded passengers (move to origin and board)
        for passenger in unboarded_passengers:
            origin_floor = self.passenger_origins[passenger]
            if lift_location != origin_floor:
                total_cost += 1  # Cost to move the lift
            total_cost += 1  # Cost to board

        # Cost for boarded passengers (move to destination and depart)
        for passenger in boarded_passengers:
            destination_floor = self.passenger_destinations[passenger]
            if lift_location != destination_floor:
                total_cost += 1  # Cost to move the lift
            total_cost += 1  # Cost to depart

        return total_cost

    def goal_reached(self, state):
        """Check if all goal conditions are satisfied in the given state."""
        return self.goals <= state
