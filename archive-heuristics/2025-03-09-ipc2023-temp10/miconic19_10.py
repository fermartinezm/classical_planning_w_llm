from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on their current locations and destinations, and the elevator's position.

    # Assumptions:
    - Each passenger needs to board the lift, travel to their destination floor, and depart.
    - The elevator moves one floor at a time.
    - The heuristic focuses on minimizing elevator movements and passenger handling.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger.
    - Determine the initial elevator floor.
    - Store the 'above' relationships between floors to calculate movement costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify passengers who are not yet served.
    2. For each unserved passenger:
       a. If the passenger is not boarded:
          i. Calculate the number of floors the elevator needs to move from its current location to the passenger's origin floor.
          ii. Add 1 action for boarding the passenger.
       b. If the passenger is boarded:
          i. Calculate the number of floors the elevator needs to move from its current location to the passenger's destination floor.
          ii. Add 1 action for departing the passenger.
    3. Sum the movement costs and boarding/departing actions for all unserved passengers.
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting passenger information and floor relationships."""
        self.goals = task.goals
        self.static = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = {}
        self.lift_at = None

        for fact in self.static:
            fact = fact.replace("(", "").replace(")", "")
            parts = fact.split()
            if parts[0] == "destin":
                self.passenger_destinations[parts[1]] = parts[2]
            elif parts[0] == "above":
                self.above[(parts[1], parts[2])] = True

        for fact in task.initial_state:
            fact = fact.replace("(", "").replace(")", "")
            parts = fact.split()
            if parts[0] == "origin":
                self.passenger_origins[parts[1]] = parts[2]
            elif parts[0] == "lift-at":
                self.lift_at = parts[1]

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        served_passengers = set()
        boarded_passengers = set()
        passenger_origins_current = {}
        lift_at_current = None

        for fact in state:
            fact = fact.replace("(", "").replace(")", "")
            parts = fact.split()
            if parts[0] == "served":
                served_passengers.add(parts[1])
            elif parts[0] == "boarded":
                boarded_passengers.add(parts[1])
            elif parts[0] == "origin":
                passenger_origins_current[parts[1]] = parts[2]
            elif parts[0] == "lift-at":
                lift_at_current = parts[1]

        if all(goal in state for goal in self.goals):
            return 0

        total_cost = 0
        unserved_passengers = set(self.passenger_destinations.keys()) - served_passengers

        for passenger in unserved_passengers:
            if passenger in boarded_passengers:
                destination_floor = self.passenger_destinations[passenger]
                total_cost += self.floor_distance(lift_at_current, destination_floor) + 1 # Move + depart
            else:
                origin_floor = passenger_origins_current.get(passenger, self.passenger_origins.get(passenger))
                if origin_floor is not None:
                    total_cost += self.floor_distance(lift_at_current, origin_floor) + 1 # Move + board
        return total_cost

    def floor_distance(self, floor1, floor2):
        """Calculate the number of floors between floor1 and floor2."""
        distance = 0
        current = floor1
        while current != floor2:
            found_next = False
            for f1, f2 in self.above:
                if f1 == current and (f2 == floor2 or self.is_above(f2, floor2)):
                    current = f2
                    distance += 1
                    found_next = True
                    break
                elif f2 == current and (f1 == floor2 or self.is_above(f1, floor2)):
                    current = f1
                    distance += 1
                    found_next = True
                    break
            if not found_next:
                # Handle cases where floors are not directly connected by 'above'
                # This is a simplification and might not be accurate for all instances
                return 1000  # Return a large value to penalize such paths
        return distance

    def is_above(self, floor1, floor2):
        """Check if floor1 is above floor2 based on the static 'above' relationships."""
        if (floor1, floor2) in self.above:
            return True
        for f1, f2 in self.above:
            if f1 == floor1 and self.is_above(f2, floor2):
                return True
        return False
