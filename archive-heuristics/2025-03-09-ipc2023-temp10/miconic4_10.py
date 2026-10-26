from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(above f1 f2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconic4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It considers the number of passengers who are waiting to board, are currently boarded,
    and the distance the elevator needs to travel to pick up and drop off passengers.

    # Assumptions
    - The elevator can carry any number of passengers.
    - The heuristic focuses on minimizing the number of elevator movements and boarding/departing actions.
    - It assumes that the elevator will always move to the closest passenger needing service.

    # Heuristic Initialization
    - Extract the initial locations and destinations of all passengers from the static facts.
    - Create a data structure to represent the "above" relationships between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current elevator location from the state.
    2. Identify passengers who are waiting to board (origin) and passengers who are boarded but not yet served (destin).
    3. Calculate the cost for boarding all waiting passengers:
       - For each waiting passenger, determine the distance (number of floors) between the elevator's current location and the passenger's origin floor.
       - Sum these distances to get the total boarding cost.
    4. Calculate the cost for serving all boarded passengers:
       - For each boarded passenger, determine the distance between the elevator's current location and the passenger's destination floor.
       - Sum these distances to get the total serving cost.
    5. Add the number of boarding and departing actions required.
    6. Return the sum of the boarding cost, serving cost, boarding actions, and departing actions as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger origins, destinations, and floor relationships.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = {}

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_destinations[parts[1]] = parts[2]
            elif match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if parts[1] not in self.above:
                    self.above[parts[1]] = []
                self.above[parts[1]].append(parts[2])

    def __call__(self, node):
        """
        Estimate the number of actions required to serve all passengers.
        """
        state = node.state

        # Extract the current elevator location.
        elevator_location = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                elevator_location = get_parts(fact)[1]
                break

        if elevator_location is None:
            return float('inf')  # If elevator location is unknown, return infinity.

        # Identify passengers who are waiting to board and passengers who are boarded.
        waiting_passengers = []
        boarded_passengers = []
        for fact in state:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                waiting_passengers.append((parts[1], parts[2]))  # (passenger, origin floor)
            elif match(fact, "boarded", "*"):
                boarded_passengers.append(get_parts(fact)[1])  # passenger

        # Calculate the cost for boarding all waiting passengers.
        boarding_cost = 0
        for passenger, origin_floor in waiting_passengers:
            boarding_cost += self.floor_distance(elevator_location, origin_floor)

        # Calculate the cost for serving all boarded passengers.
        serving_cost = 0
        for passenger in boarded_passengers:
            if passenger in self.passenger_destinations:
                destination_floor = self.passenger_destinations[passenger]
                serving_cost += self.floor_distance(elevator_location, destination_floor)

        # Add the number of boarding and departing actions required.
        num_boarding_actions = len(waiting_passengers)
        num_departing_actions = len(boarded_passengers)

        # If the goal is reached, return 0
        if node.state >= self.goals:
            return 0

        # Return the sum of the boarding cost, serving cost, boarding actions, and departing actions.
        return boarding_cost + serving_cost + num_boarding_actions + num_departing_actions

    def floor_distance(self, floor1, floor2):
        """
        Calculate the distance (number of floors) between two floors.
        This is a simple estimate and does not account for the actual floor layout.
        """
        # Find the shortest path between floor1 and floor2 using BFS
        queue = [(floor1, 0)]
        visited = {floor1}

        while queue:
            curr_floor, dist = queue.pop(0)
            if curr_floor == floor2:
                return dist

            # Check floors above
            if curr_floor in self.above:
                for next_floor in self.above[curr_floor]:
                    if next_floor not in visited:
                        queue.append((next_floor, dist + 1))
                        visited.add(next_floor)

            # Check floors below (reverse lookup)
            for floor_a, floors_b in self.above.items():
                if floor2 == floor_a:
                    return 0
                if curr_floor in floors_b:
                    prev_floor = floor_a
                    if prev_floor not in visited:
                        queue.append((prev_floor, dist + 1))
                        visited.add(prev_floor)

        return float('inf')  # If no path is found, return infinity.
