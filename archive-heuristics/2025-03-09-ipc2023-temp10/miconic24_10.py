from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's position.

    # Assumptions:
    - Each passenger needs to board the lift at their origin floor.
    - The lift needs to move to the passenger's destination floor.
    - Each passenger needs to depart the lift at their destination floor.
    - The heuristic ignores the lift capacity.

    # Heuristic Initialization
    - Extract origin and destination information for each passenger from the static facts.
    - Determine the set of floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify passengers who are not yet served.
    2. For each unserved passenger:
       a. If the passenger is not boarded:
          i. Estimate the cost to move the lift to the passenger's origin floor.
          ii. Add a boarding cost (1 action).
       b. If the passenger is boarded:
          i. Estimate the cost to move the lift to the passenger's destination floor.
          ii. Add a departing cost (1 action).
    3. Sum the costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.floors = set()

        for fact in self.static_facts:
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
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        served_passengers = set()
        boarded_passengers = set()
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

        if lift_at is None:
            return float('inf')

        unserved_passengers = set()
        for passenger in self.passenger_destinations:
            if '(served ' + passenger + ')' not in state:
                unserved_passengers.add(passenger)

        if not unserved_passengers:
            return 0

        total_cost = 0
        for passenger in unserved_passengers:
            if '(boarded ' + passenger + ')' not in state:
                origin_floor = None
                for fact in state:
                    fact = fact[1:-1]
                    parts = fact.split()
                    if parts[0] == 'origin' and parts[1] == passenger:
                        origin_floor = parts[2]
                        break
                if origin_floor is None:
                    for p,f in self.passenger_origins.items():
                        if p == passenger:
                            origin_floor = f
                            break
                if origin_floor is None:
                    return float('inf')

                total_cost += self.floor_distance(lift_at, origin_floor)
                total_cost += 1  # Boarding cost
            else:
                destination_floor = self.passenger_destinations[passenger]
                total_cost += self.floor_distance(lift_at, destination_floor)
                total_cost += 1  # Departing cost

        return total_cost

    def floor_distance(self, floor1, floor2):
        """Estimates the number of moves between two floors."""
        if floor1 == floor2:
            return 0

        above_facts = set()
        for fact in self.static_facts:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == 'above':
                above_facts.add((parts[1], parts[2]))

        def find_path(start, end, path=None):
            if path is None:
                path = [start]
            if start == end:
                return path
            for f1, f2 in above_facts:
                if f1 == start and f2 not in path:
                    new_path = find_path(f2, end, path + [f2])
                    if new_path:
                        return new_path
            for f1, f2 in above_facts:
                if f2 == start and f1 not in path:
                    new_path = find_path(f1, end, path + [f1])
                    if new_path:
                        return new_path
            return None

        path = find_path(floor1, floor2)
        if path:
            return len(path) - 1
        else:
            return float('inf')
