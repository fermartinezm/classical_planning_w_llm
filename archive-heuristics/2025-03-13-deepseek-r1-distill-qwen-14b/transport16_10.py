<code-file-heuristic-transport>
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
    This heuristic estimates the number of actions needed to transport all packages to their goal locations.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - The goal is to have all packages at their specified destinations.

    # Heuristic Initialization
    - Extract the goal locations for each package and the static facts (e.g., road connections) from the task.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each package, determine its current location and whether it's inside a vehicle.
    2. If the package is already at its goal location, no actions are needed.
    3. If the package is not at its goal:
       a. If it's not in a vehicle, calculate the cost to load it into a vehicle.
       b. Calculate the cost to drive the vehicle to the destination location.
       c. Calculate the cost to unload the package at the destination.
    4. Sum the costs for all packages to get the total heuristic value.
    5. Use a BFS-based approach to find the shortest path between locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (road connections).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (road connections)

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, package, location = get_parts(goal)
            if predicate == "at":
                self.goal_locations[package] = location

        # Build a graph of connected locations for BFS
        self.graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                self.graph[loc1].append(loc2)
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                self.graph[loc2].append(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # Track where packages and vehicles are currently located
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, location = args
                current_locations[obj] = location

        total_cost = 0  # Initialize action cost counter

        # For each package, determine if it's at the goal location
        for package, goal_location in self.goal_locations.items():
            # Skip if already at goal
            if package not in current_locations:
                continue  # Package doesn't exist in current state
            current_location = current_locations[package]

            if current_location == goal_location:
                continue  # Already at goal

            # Check if the package is inside a vehicle
            in_vehicle = current_location not in self.graph

            if in_vehicle:
                # Find the vehicle's physical location
                vehicle = None
                for fact in state:
                    if match(fact, "in", package, "*"):
                        vehicle = get_parts(fact)[2]
                        break
                # Get the vehicle's current location
                for fact in state:
                    if match(fact, "at", vehicle, "*"):
                        vehicle_location = get_parts(fact)[2]
                        break
                # Check if the vehicle is already at the goal location
                if vehicle_location == goal_location:
                    continue  # Vehicle is already at goal, no action needed

                # Calculate the cost to drive the vehicle to the goal location
                path = self.bfs_shortest_path(vehicle_location, goal_location)
                if path is None:
                    continue  # No path exists (shouldn't happen in solvable states)
                drive_cost = len(path)  # Each move is one action
                total_cost += drive_cost

            else:
                # Package is on the ground, need to load it into a vehicle
                # Find any vehicle that can carry the package
                # (Assuming all vehicles can carry any package for simplicity)
                # Load action cost
                total_cost += 1
                # Find the shortest path from current location to any vehicle's location
                # (Assuming vehicles are at some location in the graph)
                # For simplicity, assume a vehicle is available at current_location
                # Drive to goal location
                path = self.bfs_shortest_path(current_location, goal_location)
                if path is None:
                    continue  # No path exists (shouldn't happen in solvable states)
                drive_cost = len(path)
                total_cost += drive_cost
                # Unload the package at the goal location
                total_cost += 1

        return total_cost

    def bfs_shortest_path(self, start, goal):
        """
        Find the shortest path from start to goal using BFS.
        Returns the path as a list of locations, or None if no path exists.
        """
        if start == goal:
            return []
        visited = set()
        queue = deque()
        queue.append((start, []))
        while queue:
            current, path = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    return path + [neighbor]
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        return None
</code-file-heuristic-transport>