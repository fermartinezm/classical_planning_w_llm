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


class miconic23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers in the Miconic domain.
    It considers the number of passengers who are waiting to be boarded, currently boarded, and the
    distance the elevator needs to travel to pick up and drop off passengers.

    # Assumptions
    - The elevator can carry any number of passengers.
    - The heuristic assumes that the elevator will always take the shortest path to the next floor.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding and departing a passenger is 1.

    # Heuristic Initialization
    - The heuristic initializes by extracting the 'above' relationships between floors from the static facts.
    - It constructs a dictionary representing the floor hierarchy, allowing efficient calculation of the
      distance between any two floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract information about the current state:
       - Identify the current location of the elevator.
       - Identify passengers who are waiting to be boarded (origin).
       - Identify passengers who are currently boarded.
       - Identify passengers who have already been served.
       - Extract the destination floors for each passenger.

    2. Calculate the cost for boarding all waiting passengers:
       - For each waiting passenger, determine the distance between the elevator's current floor and the
         passenger's origin floor.
       - Add the boarding cost (1) to the total cost.

    3. Calculate the cost for dropping off all boarded passengers:
       - For each boarded passenger, determine the distance between the elevator's current floor and the
         passenger's destination floor.
       - Add the departing cost (1) to the total cost.

    4. If there are both waiting and boarded passengers, calculate the optimal route:
       - Find the closest passenger to pick up and the closest passenger to drop off.
       - Move to the closest passenger's origin floor, board them, and add the boarding cost.
       - Move to the closest passenger's destination floor, depart them, and add the departing cost.

    5. If there are only waiting passengers, calculate the cost to pick them up and drop them off:
       - Move to each passenger's origin floor, board them, and add the boarding cost.
       - Move to each passenger's destination floor, depart them, and add the departing cost.

    6. If there are only boarded passengers, calculate the cost to drop them off:
       - Move to each passenger's destination floor, depart them, and add the departing cost.

    7. The total heuristic value is the sum of all calculated costs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the floor hierarchy from the static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build a dictionary representing the floor hierarchy.
        self.floor_hierarchy = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)[1], get_parts(fact)[2]
                if f1 not in self.floor_hierarchy:
                    self.floor_hierarchy[f1] = []
                self.floor_hierarchy[f1].append(f2)

    def __call__(self, node):
        """
        Estimate the number of actions needed to serve all passengers.
        """
        state = node.state
        goal = node.task.goals

        # If the goal is reached, the heuristic value is 0.
        if node.task.goal_reached(state):
            return 0

        # Extract information from the current state.
        lift_at = next((get_parts(fact)[1] for fact in state if match(fact, "lift-at", "*")), None)
        origins = [(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "origin", "*", "*")]
        boarded = [get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")]
        served = [get_parts(fact)[1] for fact in state if match(fact, "served", "*")]
        destinations = {}
        for fact in node.task.static:
            if match(fact, "destin", "*", "*"):
                passenger, floor = get_parts(fact)[1], get_parts(fact)[2]
                destinations[passenger] = floor

        # Calculate the heuristic cost.
        cost = 0

        # Cost for boarding waiting passengers.
        for passenger, origin in origins:
            cost += 1  # Boarding cost
            if lift_at != origin:
                cost += self.floor_distance(lift_at, origin)

        # Cost for departing boarded passengers.
        for passenger in boarded:
            destination = destinations[passenger]
            cost += 1  # Departing cost
            if lift_at != destination:
                cost += self.floor_distance(lift_at, destination)

        return cost

    def floor_distance(self, start_floor, end_floor):
        """
        Calculate the distance between two floors based on the floor hierarchy.
        This assumes that the elevator will always take the shortest path.
        """
        if start_floor == end_floor:
            return 0

        # Simple BFS to find the shortest path.
        queue = [(start_floor, 0)]  # (floor, distance)
        visited = {start_floor}

        while queue:
            current_floor, distance = queue.pop(0)

            if current_floor == end_floor:
                return distance

            # Check floors above.
            if current_floor in self.floor_hierarchy:
                for next_floor in self.floor_hierarchy[current_floor]:
                    if next_floor not in visited:
                        queue.append((next_floor, distance + 1))
                        visited.add(next_floor)

            # Check floors below (reverse lookup).
            for floor, above_floors in self.floor_hierarchy.items():
                if end_floor == start_floor:
                    return 0
                if end_floor in above_floors:
                    return 1
                if current_floor in above_floors:
                    if floor not in visited:
                        queue.append((floor, distance + 1))
                        visited.add(floor)

        # If no path is found, return a large value (should not happen in well-formed problems).
        return 1000
