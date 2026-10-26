from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic2Heuristic(Heuristic):
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
    - Build a data structure representing the 'above' relationships between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify passengers who are not yet served.
    2. For each unserved passenger:
       a. If the passenger is not boarded:
          i. Estimate the cost to move the elevator to the passenger's origin floor.
          ii. Add the cost of the 'board' action.
       b. If the passenger is boarded:
          i. Estimate the cost to move the elevator to the passenger's destination floor.
          ii. Add the cost of the 'depart' action.
    3. Return the sum of these costs as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = {}

        for fact in static_facts:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] == 'destin':
                self.passenger_destinations[fact_parts[1]] = fact_parts[2]
            elif fact_parts[0] == 'above':
                f1 = fact_parts[1]
                f2 = fact_parts[2]
                if f1 not in self.above:
                    self.above[f1] = []
                self.above[f1].append(f2)

        for fact in task.initial_state | static_facts:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] == 'origin':
                self.passenger_origins[fact_parts[1]] = fact_parts[2]

    def _extract_objects_from_fact(self, fact):
        """Extract objects from a PDDL fact string."""
        return fact[1:-1].split()

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        lift_at = None
        boarded_passengers = set()
        served_passengers = set()
        unserved_passengers = set()

        for fact in state:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] == 'lift-at':
                lift_at = fact_parts[1]
            elif fact_parts[0] == 'boarded':
                boarded_passengers.add(fact_parts[1])
            elif fact_parts[0] == 'served':
                served_passengers.add(fact_parts[1])

        for passenger in self.passenger_origins.keys():
            if passenger not in served_passengers:
                unserved_passengers.add(passenger)

        if not unserved_passengers:
            return 0

        total_cost = 0
        for passenger in unserved_passengers:
            if passenger not in boarded_passengers:
                origin_floor = self.passenger_origins[passenger]
                total_cost += self._estimate_move_cost(lift_at, origin_floor)
                total_cost += 1  # Cost of board action
            else:
                destination_floor = self.passenger_destinations[passenger]
                total_cost += self._estimate_move_cost(lift_at, destination_floor)
                total_cost += 1  # Cost of depart action

        return total_cost

    def _estimate_move_cost(self, current_floor, target_floor):
        """Estimate the cost to move the elevator from current_floor to target_floor."""
        if current_floor == target_floor:
            return 0

        # Find the shortest path between the floors using the 'above' relationships.
        # This is a simplified approach and may not be the most accurate, but it's efficient.
        # A more sophisticated approach would involve graph search.
        cost = 0
        found = False

        def recursive_search(current, target, path):
            nonlocal found, cost
            if found:
                return

            if current == target:
                found = True
                cost = len(path)
                return

            if current in self.above:
                for next_floor in self.above[current]:
                    if next_floor not in path:
                        recursive_search(next_floor, target, path + [next_floor])

            # Search downwards
            for floor1, floors_above in self.above.items():
                if current in floors_above:
                    if floor1 not in path:
                        recursive_search(floor1, target, path + [floor1])

        recursive_search(current_floor, target_floor, [current_floor])

        if not found:
            # If no path is found, return a large cost to discourage this path.
            return 100

        return cost
