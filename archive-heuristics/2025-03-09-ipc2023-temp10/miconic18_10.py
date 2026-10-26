from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's position.

    # Assumptions:
    - Each passenger needs to board the elevator at their origin floor.
    - The elevator needs to move to the passenger's destination floor.
    - Each passenger needs to depart the elevator at their destination floor.
    - The elevator can only be at one floor at a time.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Determine the set of floors.
    - Precompute the "above" relationships between floors to calculate movement costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current elevator location.
    2. For each passenger who is not yet served:
       - If the passenger is not yet boarded:
         - Calculate the cost to move the elevator from its current location to the passenger's origin floor.
         - Add a boarding cost (1).
       - If the passenger is boarded:
         - Calculate the cost to move the elevator from its current location to the passenger's destination floor.
         - Add a departing cost (1).
    3. Sum up all the costs for all passengers.
    4. If all passengers are served, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = {}
        self.floors = set()

        for fact in static_facts:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == 'destin':
                self.passenger_destinations[parts[1]] = parts[2]
            elif parts[0] == 'above':
                self.above[(parts[1], parts[2])] = True

        for fact in static_facts:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == 'destin':
                self.floors.add(parts[2])
            if parts[0] == 'above':
                self.floors.add(parts[1])
                self.floors.add(parts[2])

    def __call__(self, node):
        """Estimate the minimum cost to serve all passengers."""
        state = node.state
        
        if self.is_goal_state(state):
            return 0

        elevator_floor = None
        for fact in state:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == 'lift-at':
                elevator_floor = parts[1]
                break

        total_cost = 0
        for passenger in self.passenger_destinations:
            served = False
            boarded = False
            origin_floor = None

            for fact in state:
                fact = fact[1:-1]
                parts = fact.split()
                if parts[0] == 'served' and parts[1] == passenger:
                    served = True
                    break
                if parts[0] == 'boarded' and parts[1] == passenger:
                    boarded = True
                    break
                if parts[0] == 'origin' and parts[1] == passenger:
                    origin_floor = parts[2]

            if not served:
                if not boarded:
                    # Calculate cost to move elevator to origin floor and board
                    if origin_floor is not None:
                        total_cost += self.calculate_movement_cost(elevator_floor, origin_floor)
                        total_cost += 1  # Boarding cost
                else:
                    # Calculate cost to move elevator to destination floor and depart
                    destination_floor = self.passenger_destinations[passenger]
                    total_cost += self.calculate_movement_cost(elevator_floor, destination_floor)
                    total_cost += 1  # Departing cost

        return total_cost

    def calculate_movement_cost(self, start_floor, end_floor):
        """Calculates the number of up/down actions required to move between floors."""
        if start_floor == end_floor:
            return 0

        cost = 0
        current = start_floor
        
        if self.is_above(start_floor, end_floor):
            while current != end_floor:
                found_next = False
                for f1, f2 in self.above:
                    if f1 == current and self.is_above(f2, end_floor) or f2 == end_floor:
                        current = f2
                        cost += 1
                        found_next = True
                        break
                if not found_next:
                    return float('inf')
        else:
            while current != end_floor:
                found_next = False
                for f1, f2 in self.above:
                    if f2 == current and self.is_above(end_floor, f1) or f1 == end_floor:
                        current = f1
                        cost += 1
                        found_next = True
                        break
                if not found_next:
                    return float('inf')

        return cost

    def is_above(self, floor1, floor2):
        """Checks if floor1 is above floor2 using the precomputed 'above' relationships."""
        if floor1 == floor2:
            return False
        
        visited = set()
        queue = [floor1]
        
        while queue:
            current = queue.pop(0)
            if current == floor2:
                return True
            if current in visited:
                continue
            visited.add(current)
            
            for f1, f2 in self.above:
                if f1 == current:
                    queue.append(f2)
        return False

    def is_goal_state(self, state):
        """Check if the state is a goal state."""
        return self.goals <= state
