from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It considers the number of passengers who are not yet served, the current
    location of the lift, and the distances to the origin and destination floors
    of the passengers.

    # Assumptions:
    - The lift can only carry one passenger at a time.
    - The heuristic assumes that the lift will always move to the closest
      passenger origin or destination floor.
    - The heuristic does not take into account the capacity of the lift.

    # Heuristic Initialization
    - Extract the 'above' relationships between floors from the static facts.
    - Create a dictionary to store the above relationships for efficient lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the lift.
    2. Identify all passengers who are not yet served.
    3. For each unserved passenger:
       a. Determine the origin and destination floors.
       b. If the passenger is not yet boarded, calculate the cost to reach the origin floor from the current lift location.
       c. If the passenger is boarded, calculate the cost to reach the destination floor from the current lift location.
       d. If the passenger is neither boarded nor at their origin, add 1 to account for the boarding action.
       e. If the passenger is boarded but not at their destination, add 1 to account for the depart action.
       f. Sum the costs for all unserved passengers.
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract 'above' relationships from static facts.
        self.above = {}
        for fact in static_facts:
            parts = fact[1:-1].split()
            if parts[0] == 'above':
                f1, f2 = parts[1], parts[2]
                if f1 not in self.above:
                    self.above[f1] = []
                self.above[f1].append(f2)

    def __call__(self, node):
        """Estimate the number of actions needed to reach a goal state."""
        state = node.state

        # Check if the current state is a goal state.
        if self.goal_reached(state):
            return 0

        # Extract the current lift location.
        lift_location = None
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'lift-at':
                lift_location = parts[1]
                break

        if lift_location is None:
            return float('inf')  # No lift location found, unsolvable state

        # Identify unserved passengers and their origin/destination floors.
        unserved_passengers = []
        passenger_origins = {}
        passenger_destinations = {}
        boarded_passengers = set()

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'origin':
                passenger, origin = parts[1], parts[2]
                passenger_origins[passenger] = origin
                if f'(served {passenger})' not in state:
                    unserved_passengers.append(passenger)
            elif parts[0] == 'destin':
                passenger, destination = parts[1], parts[2]
                passenger_destinations[passenger] = destination
            elif parts[0] == 'boarded':
                passenger = parts[1]
                boarded_passengers.add(passenger)

        # Calculate the estimated cost for each unserved passenger.
        total_cost = 0
        for passenger in unserved_passengers:
            origin = passenger_origins[passenger]
            destination = passenger_destinations[passenger]

            if passenger in boarded_passengers:
                # Passenger is boarded, calculate cost to destination.
                cost_to_destination = self.floor_distance(lift_location, destination)
                total_cost += cost_to_destination + 1 # Move to destination + depart
            else:
                # Passenger is not boarded, calculate cost to origin.
                cost_to_origin = self.floor_distance(lift_location, origin)
                total_cost += cost_to_origin + 1 # Move to origin + board

        return total_cost

    def floor_distance(self, start_floor, end_floor):
        """Estimate the number of up/down actions required to move between floors."""
        if start_floor == end_floor:
            return 0

        # Find a path from start_floor to end_floor using the 'above' relationships.
        visited = {start_floor}
        queue = [(start_floor, 0)]  # (floor, distance)

        while queue:
            current_floor, distance = queue.pop(0)

            if current_floor == end_floor:
                return distance

            # Check floors above
            if current_floor in self.above:
                for next_floor in self.above[current_floor]:
                    if next_floor not in visited:
                        visited.add(next_floor)
                        queue.append((next_floor, distance + 1))

            # Check floors below (reverse lookup)
            for floor, above_floors in self.above.items():
                if current_floor in above_floors and floor not in visited:
                    visited.add(floor)
                    queue.append((floor, distance + 1))

        return float('inf')  # No path found, unsolvable state

    def goal_reached(self, state):
        """Check if the current state is a goal state."""
        return self.goals <= state
