from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    from their current locations to their target locations using the shortest path.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move along roads between locations.
    - The goal is to have all packages at their respective target locations.

    # Heuristic Initialization
    - Extract the target location for each package from the goal conditions.
    - Build a graph representation of the road network using static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, determine its current location and whether it is in a vehicle.
    2. Calculate the shortest path from the current location to the target location using Dijkstra's algorithm.
    3. If the package is in a vehicle, add the necessary unloading actions.
    4. If the package is not in a vehicle, add the necessary loading and moving actions.
    5. Sum the actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        # Build road network graph
        self.graph = {}
        for fact in self.static:
            if fact.startswith('(road'):
                l1, l2 = fact[5:-1].split()
                if l1 not in self.graph:
                    self.graph[l1] = []
                if l2 not in self.graph:
                    self.graph[l2] = []
                self.graph[l1].append(l2)
                self.graph[l2].append(l1)

    def __call__(self, node):
        """Estimate the minimum cost to transport all packages to their goal locations."""
        state = node.state

        # Extract current locations of all objects
        current_locations = {}
        for fact in state:
            if fact.startswith('(at'):
                parts = fact[3:-1].split()
                obj = parts[0]
                loc = parts[1]
                current_locations[obj] = loc
            elif fact.startswith('(in'):
                parts = fact[3:-1].split()
                obj = parts[0]
                vehicle = parts[1]
                current_locations[obj] = vehicle

        total_actions = 0

        # For each package, calculate required actions
        for package in self.goal_locations:
            goal_loc = self.goal_locations[package]
            current_loc = current_locations.get(package, None)

            if current_loc is None:
                continue  # Package not present in state (should not happen)

            # If package is already at goal, no actions needed
            if current_loc == goal_loc:
                continue

            # Check if package is in a vehicle
            in_vehicle = current_loc not in self.graph

            # Find the physical location of the vehicle if in vehicle
            if in_vehicle:
                vehicle = current_loc
                vehicle_loc = current_locations.get(vehicle, None)
                if vehicle_loc is None:
                    vehicle_loc = current_loc  # Assume vehicle is at current_loc
                current_loc = vehicle_loc

            # Calculate shortest path from current location to goal
            if current_loc not in self.graph or goal_loc not in self.graph:
                # No path available (should not happen in solvable states)
                continue

            # Dijkstra's algorithm to find shortest path
            heap = []
            heapq.heappush(heap, (0, current_loc))
            distances = {node: float('infinity') for node in self.graph}
            distances[current_loc] = 0
            visited = set()

            while heap:
                dist, u = heapq.heappop(heap)
                if u in visited:
                    continue
                if u == goal_loc:
                    break
                visited.add(u)
                for v in self.graph[u]:
                    if v not in visited:
                        new_dist = distances[u] + 1
                        if new_dist < distances[v]:
                            distances[v] = new_dist
                            heapq.heappush(heap, (new_dist, v))

            path_length = distances.get(goal_loc, 0)
            if path_length == float('infinity'):
                # No path available (should not happen in solvable states)
                continue

            # Estimate actions based on current state
            if in_vehicle:
                # Unload the package from the vehicle
                total_actions += 1
                # Move the vehicle to the goal location
                total_actions += path_length
            else:
                # Load the package into a vehicle
                total_actions += 1
                # Move the vehicle to the goal location
                total_actions += path_length
                # Unload the package at the goal
                total_actions += 1

        return total_actions
