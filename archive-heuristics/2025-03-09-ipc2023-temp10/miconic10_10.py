from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's location.

    # Assumptions:
    - Each passenger needs to board the elevator at their origin floor.
    - The elevator needs to move to the passenger's destination floor.
    - Each passenger needs to depart the elevator at their destination floor.
    - The elevator can serve multiple passengers at the same time.

    # Heuristic Initialization
    - Extract origin and destination floors for each passenger from the static facts.
    - Identify all floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the passengers who are not yet served.
    2. For each unserved passenger:
       a. If the passenger is not boarded:
          i. Calculate the cost to move the elevator to the passenger's origin floor.
          ii. Add the cost of boarding the passenger.
       b. If the passenger is boarded:
          i. Calculate the cost to move the elevator to the passenger's destination floor.
          ii. Add the cost of departing the passenger.
    3. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.floors = set()

        for fact in static_facts:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == 'destin':
                self.passenger_destinations[parts[1]] = parts[2]
            elif parts[0] == 'above':
                self.floors.add(parts[1])
                self.floors.add(parts[2])

        for fact in task.initial_state:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == 'origin':
                self.passenger_origins[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the minimum cost to serve all passengers."""
        state = node.state
        served_passengers = set()
        boarded_passengers = set()
        unboarded_passengers = set()
        lift_at = None

        for fact in state:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == 'served':
                served_passengers.add(parts[1])
            elif parts[0] == 'boarded':
                boarded_passengers.add(parts[1])
            elif parts[0] == 'lift-at':
                lift_at = parts[1]
        
        unserved_passengers = set(self.passenger_destinations.keys()) - served_passengers

        if not unserved_passengers:
            return 0

        total_cost = 0
        for passenger in unserved_passengers:
            if passenger not in boarded_passengers:
                origin_floor = self.passenger_origins.get(passenger)
                if origin_floor is None:
                    for fact in state:
                        fact = fact[1:-1]
                        parts = fact.split()
                        if parts[0] == 'origin' and parts[1] == passenger:
                            origin_floor = parts[2]
                            break
                
                if origin_floor is None:
                    continue

                if lift_at != origin_floor:
                    total_cost += 1  # Cost to move the elevator to the origin floor
                total_cost += 1  # Cost to board the passenger
            else:
                destination_floor = self.passenger_destinations[passenger]
                if lift_at != destination_floor:
                    total_cost += 1  # Cost to move the elevator to the destination floor
                total_cost += 1  # Cost to depart the passenger

        return total_cost
