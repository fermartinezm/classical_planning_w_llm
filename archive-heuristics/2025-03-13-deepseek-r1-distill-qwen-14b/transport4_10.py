from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move along roads between locations.
    - Each package must be picked up and dropped off exactly once.

    # Heuristic Initialization
    - Extract the goal locations for each package.
    - Build a graph of road connections from static facts.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each package, determine if it is already at its goal location. If yes, no actions are needed.
    2. If the package is not at its goal, check if it is inside a vehicle or on the ground.
    3. For packages inside a vehicle, find the vehicle's current location.
    4. For packages on the ground, use their current location.
    5. Compute the shortest path from the current location to the goal location using the road network.
    6. For each package, the number of actions is the sum of driving actions (path length) and handling actions (pickup and drop-off).
    7. Sum the actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (road connections) into a graph structure.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, package, location = get_parts(goal)
            if predicate == "at":
                self.goal_locations[package] = location

        # Build road network graph
        self.roads = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 not in self.roads:
                    self.roads[l1] = []
                self.roads[l1].append(l2)
                if l2 not in self.roads:
                    self.roads[l2] = []
                self.roads[l2].append(l1)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Track where packages and vehicles are currently located
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, location = args
                current_locations[obj] = location

        total_actions = 0  # Initialize action counter

        # For each package, determine if it's at the goal
        for package, goal_location in self.goal_locations.items():
            if package not in current_locations:
                # Package is not present in the state (should not happen in valid state)
                continue

            current_location = current_locations[package]

            if current_location == goal_location:
                # Package is already at the goal
                continue

            # Determine if the package is in a vehicle or on the ground
            in_vehicle = current_location not in self.roads  # If it's a vehicle, it's not a location
            if in_vehicle:
                # Find the vehicle's location
                vehicle = current_location
                # Vehicle's location is where the vehicle is currently
                vehicle_location = None
                for fact in state:
                    if match(fact, "at", "*", "*"):
                        obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                        if obj == vehicle:
                            vehicle_location = loc
                            break
                if vehicle_location is None:
                    # Vehicle not found (invalid state)
                    continue
                current_location = vehicle_location

            # Find the shortest path from current_location to goal_location
            path = self.breadth_first_search(current_location, goal_location)
            if path is None:
                # No path exists (should not happen in valid problem)
                continue

            # Number of drive actions is the number of steps in the path minus one
            drive_actions = len(path) - 1

            # Add pickup and drop-off actions if the package is not already in the vehicle
            if not in_vehicle:
                drive_actions += 1  # Pickup
            drive_actions += 1  # Drop-off

            total_actions += drive_actions

        return total_actions

    def breadth_first_search(self, start, goal):
        """
        Perform BFS to find the shortest path from start to goal in the road network.
        """
        if start == goal:
            return [start]

        visited = set()
        queue = deque()
        queue.append((start, [start]))

        while queue:
            current, path = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.roads.get(current, []):
                if neighbor == goal:
                    return path + [neighbor]
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found
