from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their origins, destinations, and the current elevator location.
    It considers boarding, departing, and moving the elevator.

    # Assumptions:
    - Each passenger needs one 'board' and one 'depart' action.
    - The elevator needs to move between floors to pick up and drop off passengers.
    - The heuristic assumes the elevator always takes the shortest path to the next floor.

    # Heuristic Initialization
    - Extract passenger origins and destinations from the static facts.
    - Create a data structure to quickly look up the origin and destination of each passenger.
    - Determine the order of floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify passengers who are not yet served.
    2. For each unserved passenger, determine the number of actions required:
       - One 'board' action at the origin floor.
       - One 'depart' action at the destination floor.
       - Actions to move the elevator from its current location to the passenger's origin.
       - Actions to move the elevator from the passenger's origin to the passenger's destination.
    3. Sum the number of actions for all unserved passengers.
    4. Divide the sum by a factor (e.g., 2) to account for the fact that some actions may serve multiple passengers.
    5. If all passengers are served, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.floors_above = {}

        # Extract passenger origins and destinations
        for fact in self.static_facts:
            fact_str = fact[1:-1]
            parts = fact_str.split()
            if parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.passenger_destinations[passenger] = floor
        for fact in task.initial_state:
            fact_str = fact[1:-1]
            parts = fact_str.split()
            if parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                self.passenger_origins[passenger] = floor

        # Extract floor order
        self.floors_above = {}
        for fact in self.static_facts:
            fact_str = fact[1:-1]
            parts = fact_str.split()
            if parts[0] == 'above':
                floor1 = parts[1]
                floor2 = parts[2]
                if floor1 not in self.floors_above:
                    self.floors_above[floor1] = []
                self.floors_above[floor1].append(floor2)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Check if the goal is reached
        if self.goal_reached(state):
            return 0

        # Find the current elevator location
        elevator_location = None
        for fact in state:
            fact_str = fact[1:-1]
            parts = fact_str.split()
            if parts[0] == 'lift-at':
                elevator_location = parts[1]
                break

        # Find unserved passengers
        unserved_passengers = []
        for passenger in self.passenger_origins:
            served = False
            for fact in state:
                fact_str = fact[1:-1]
                parts = fact_str.split()
                if parts[0] == 'served' and parts[1] == passenger:
                    served = True
                    break
            if not served:
                unserved_passengers.append(passenger)

        # Calculate the estimated cost
        total_cost = 0
        for passenger in unserved_passengers:
            origin = self.passenger_origins.get(passenger)
            destination = self.passenger_destinations.get(passenger)

            # Check if the passenger is already boarded
            boarded = False
            for fact in state:
                fact_str = fact[1:-1]
                parts = fact_str.split()
                if parts[0] == 'boarded' and parts[1] == passenger:
                    boarded = True
                    break

            # Add cost for boarding and departing
            if not boarded:
                total_cost += 1  # Board action
            total_cost += 1  # Depart action

            # Add cost for moving the elevator
            if not boarded:
                total_cost += self.floor_distance(elevator_location, origin)
                elevator_location = origin
            total_cost += self.floor_distance(elevator_location, destination)
            elevator_location = destination

        # Normalize the cost
        total_cost /= 2  # Account for the fact that moving the elevator can serve multiple passengers

        return int(total_cost)

    def goal_reached(self, state):
        """Check if all goal conditions are met in the given state."""
        return self.goals <= state

    def floor_distance(self, start_floor, end_floor):
        """Estimate the number of moves required to go from start_floor to end_floor."""
        # This is a very basic heuristic. A more sophisticated one could use the 'above' predicates.
        # This implementation assumes that the floors are ordered and the distance is the absolute difference in their order.
        # It is not guaranteed to be accurate, but it is fast to compute.
        # It is also not admissible, but it is intended to guide the search.
        distance = 0
        current = start_floor
        while current != end_floor:
            found_next = False
            for higher_floor, lower_floors in self.floors_above.items():
                if current == higher_floor:
                    if end_floor in lower_floors:
                        current = end_floor
                        distance += 1
                        found_next = True
                        break
                    else:
                        for lower_floor in lower_floors:
                            if self.is_ancestor(lower_floor, end_floor):
                                current = lower_floor
                                distance += 1
                                found_next = True
                                break
                        if found_next:
                            break
            if not found_next:
                # If we can't find a floor above, try to find a floor below
                for higher_floor, lower_floors in self.floors_above.items():
                    if end_floor == higher_floor:
                        current = higher_floor
                        distance += 1
                        found_next = True
                        break
        return distance

    def is_ancestor(self, start_floor, end_floor):
        """Check if end_floor is an ancestor of start_floor based on the above predicates."""
        if start_floor == end_floor:
            return True

        queue = [start_floor]
        visited = set()

        while queue:
            current_floor = queue.pop(0)
            if current_floor in visited:
                continue
            visited.add(current_floor)

            if current_floor in self.floors_above:
                for lower_floor in self.floors_above[current_floor]:
                    if lower_floor == end_floor:
                        return True
                    queue.append(lower_floor)

        return False
