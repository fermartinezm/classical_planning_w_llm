from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on the current state of the elevator and the passengers.

    # Assumptions:
    - Each passenger needs to board the elevator at their origin floor.
    - The elevator needs to move to the passenger's destination floor.
    - Each passenger needs to depart the elevator at their destination floor.
    - The elevator can serve multiple passengers at the same time.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Determine the possible floors based on the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the elevator.
    2. Identify the passengers who are not yet served.
    3. For each unserved passenger:
       - If the passenger is not yet boarded:
         - Calculate the number of floors the elevator needs to move to reach the passenger's origin floor.
         - Add 1 action for the board action.
       - If the passenger is boarded:
         - Calculate the number of floors the elevator needs to move to reach the passenger's destination floor.
         - Add 1 action for the depart action.
    4. Sum the number of actions for all unserved passengers.
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

    def __call__(self, node):
        """Estimate the minimum cost to serve all passengers."""
        state = node.state

        def match(fact, *args):
            """Utility function to check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Find the current floor of the lift.
        for fact in state:
            if match(fact, 'lift-at', '*'):
                lift_floor = fact[9:-2]
                break
        else:
            return float('inf')  # No lift location found

        # Find unserved passengers.
        unserved_passengers = set()
        for passenger in self.passenger_origins.keys():
            served = False
            for fact in state:
                if match(fact, 'served', passenger):
                    served = True
                    break
            if not served:
                unserved_passengers.add(passenger)

        heuristic_value = 0
        for passenger in unserved_passengers:
            boarded = False
            for fact in state:
                if match(fact, 'boarded', passenger):
                    boarded = True
                    break

            if not boarded:
                # Passenger needs to board.
                if passenger in self.passenger_origins:
                    origin_floor = self.passenger_origins[passenger]
                    if lift_floor != origin_floor:
                        heuristic_value += 1  # Cost to move lift to origin
                    heuristic_value += 1  # Cost to board
                else:
                    return float('inf')  # Passenger origin not found
            else:
                # Passenger needs to depart.
                if passenger in self.passenger_destinations:
                    destination_floor = self.passenger_destinations[passenger]
                    if lift_floor != destination_floor:
                        heuristic_value += 1  # Cost to move lift to destination
                    heuristic_value += 1  # Cost to depart
                else:
                    return float('inf')  # Passenger destination not found

        # Goal check: if all passengers are served, the heuristic value is 0
        all_served = True
        for passenger in self.passenger_origins.keys():
            served = False
            for fact in state:
                if match(fact, 'served', passenger):
                    served = True
                    break
            if not served:
                all_served = False
                break

        if all_served:
            return 0

        return heuristic_value
