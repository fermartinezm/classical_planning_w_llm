from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's location.
    It considers boarding, departing, and moving the elevator.

    # Assumptions:
    - Each passenger needs to board, travel, and depart.
    - The elevator needs to move between floors.
    - The heuristic does not account for optimal elevator scheduling.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Determine the set of floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify passengers who are not yet served.
    2. For each unserved passenger:
       - If the passenger is not boarded:
         - Estimate the cost to move the elevator to the passenger's origin floor (if needed).
         - Add the cost of boarding the passenger.
       - If the passenger is boarded:
         - Estimate the cost to move the elevator to the passenger's destination floor (if needed).
         - Add the cost of departing the passenger.
    3. The total heuristic value is the sum of these individual costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.floors = set()

        for fact in static_facts:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'destin':
                self.passenger_destinations[fact_parts[1]] = fact_parts[2]
            elif fact_parts[0] == 'above':
                self.floors.add(fact_parts[1])
                self.floors.add(fact_parts[2])

        for fact in task.initial_state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'origin':
                self.passenger_origins[fact_parts[1]] = fact_parts[2]

        all_passengers = set(self.passenger_origins.keys())
        for passenger in self.passenger_destinations.keys():
            all_passengers.add(passenger)

    def __call__(self, node):
        """Estimate the minimum cost to serve all passengers."""
        state = node.state

        if self.goal_reached(state):
            return 0

        def match(fact, *args):
            """Utility function to check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Find the current elevator location.
        elevator_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                elevator_floor = fact[1:-1].split()[1]
                break

        if not elevator_floor:
            return float('inf')  # Elevator must be at some floor

        # Identify unserved passengers.
        unserved_passengers = set()
        for passenger in self.passenger_origins.keys():
            served = False
            for fact in state:
                if match(fact, "served", passenger):
                    served = True
                    break
            if not served:
                unserved_passengers.add(passenger)

        for fact in state:
            fact_parts = fact[1:-1].split()
            if fact_parts[0] == 'boarded':
                passenger = fact_parts[1]
                served = False
                for f in state:
                    if match(f, "served", passenger):
                        served = True
                        break
                if not served:
                    unserved_passengers.add(passenger)

        total_cost = 0

        for passenger in unserved_passengers:
            boarded = False
            for fact in state:
                if match(fact, "boarded", passenger):
                    boarded = True
                    break

            if not boarded:
                # Passenger needs to board
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = fact[1:-1].split()[2]
                        break
                if not origin_floor:
                    for p,o in self.passenger_origins.items():
                        if p == passenger:
                            origin_floor = o
                            break

                if not origin_floor:
                    return float('inf')

                if elevator_floor != origin_floor:
                    total_cost += 1  # Cost to move elevator
                total_cost += 1  # Cost to board

            else:
                # Passenger needs to depart
                destination_floor = self.passenger_destinations[passenger]

                if elevator_floor != destination_floor:
                    total_cost += 1  # Cost to move elevator
                total_cost += 1  # Cost to depart

        return total_cost
