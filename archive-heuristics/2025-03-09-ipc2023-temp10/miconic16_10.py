from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their origin and destination floors, and the current elevator position.
    It considers the number of passengers waiting at different floors, the distance
    of the elevator from those floors, and the number of passengers already boarded.

    # Assumptions:
    - Each passenger needs to be boarded and then depart at their destination.
    - The elevator needs to move to the origin floor of each passenger to board them.
    - The elevator needs to move to the destination floor of each boarded passenger to let them depart.
    - The heuristic assumes that the elevator will serve all passengers in a single run,
      without returning to previously visited floors.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Build a data structure representing the 'above' relationships between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current elevator location from the state.
    2. Identify passengers who are waiting at their origin floors (not yet boarded).
    3. Identify passengers who are already boarded but not yet served.
    4. Calculate the cost for each waiting passenger:
       - Cost to move the elevator to the passenger's origin floor.
       - Cost to board the passenger (1 action).
    5. Calculate the cost for each boarded passenger:
       - Cost to move the elevator to the passenger's destination floor.
       - Cost to depart the passenger (1 action).
    6. Sum up all the costs to get the estimated number of actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = {}

        for fact in static_facts:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == 'destin':
                self.passenger_destinations[parts[1]] = parts[2]
            elif parts[0] == 'above':
                if parts[1] not in self.above:
                    self.above[parts[1]] = []
                self.above[parts[1]].append(parts[2])

    def __call__(self, node):
        """Estimate the minimum cost to serve all passengers."""
        state = node.state

        def match(fact, *args):
            """Utility function to check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract current elevator location
        elevator_floor = None
        for fact in state:
            if match(fact, 'lift-at', '*'):
                elevator_floor = fact[1:-1].split()[1]
                break

        if elevator_floor is None:
            return float('inf')  # Elevator location is unknown

        # Identify waiting passengers (origin floor, not boarded)
        waiting_passengers = []
        for fact in state:
            if match(fact, 'origin', '*', '*'):
                passenger = fact[1:-1].split()[1]
                waiting_passengers.append(passenger)

        # Identify boarded passengers (boarded, not served)
        boarded_passengers = []
        for fact in state:
            if match(fact, 'boarded', '*'):
                passenger = fact[1:-1].split()[1]
                boarded_passengers.append(passenger)

        # Check if the goal is reached
        if self.goal_reached(state):
            return 0

        # Calculate the cost
        cost = 0

        # Cost for waiting passengers
        for passenger in waiting_passengers:
            origin_floor = None
            for fact in state:
                if match(fact, 'origin', passenger, '*'):
                    origin_floor = fact[1:-1].split()[2]
                    break
            if origin_floor is None:
                continue

            cost += self.floor_distance(elevator_floor, origin_floor)
            cost += 1  # Board action

        # Cost for boarded passengers
        for passenger in boarded_passengers:
            destination_floor = self.passenger_destinations.get(passenger)
            if destination_floor is None:
                continue

            cost += self.floor_distance(elevator_floor, destination_floor)
            cost += 1  # Depart action

        return cost

    def floor_distance(self, start_floor, end_floor):
        """Estimates the number of up/down actions to move between floors."""
        if start_floor == end_floor:
            return 0

        # Simple heuristic: count the number of floors between start and end
        floors = [start_floor]
        visited = {start_floor}
        queue = [start_floor]
        distances = {start_floor: 0}

        while queue:
            current_floor = queue.pop(0)
            if current_floor == end_floor:
                return distances[current_floor]

            # Check floors above
            if current_floor in self.above:
                for next_floor in self.above[current_floor]:
                    if next_floor not in visited:
                        visited.add(next_floor)
                        queue.append(next_floor)
                        distances[next_floor] = distances[current_floor] + 1

            # Check floors below
            for floor, above_floors in self.above.items():
                if end_floor in above_floors and floor not in visited:
                    visited.add(floor)
                    queue.append(floor)
                    distances[floor] = distances[current_floor] + 1

        return float('inf')

    def goal_reached(self, state):
        """Check if all goal conditions are satisfied in the given state."""
        return self.goals <= state
