from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(road l1 l2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    to their goal locations. It considers:
    - The distance packages need to be moved (using a simple graph distance)
    - Whether vehicles need to pick up/drop packages
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - Vehicles can share the transportation workload
    - Road connections are bidirectional (though the domain allows one-way roads)
    - The heuristic doesn't track which vehicle transports which package

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph of road connections for distance calculations
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a. Find the shortest path distance from current location to goal
        b. If the package is not in a vehicle:
            - Add 1 action for picking up (if a vehicle is at the location)
            - Add distance actions for moving the vehicle to the goal
            - Add 1 action for dropping
        c. If the package is in a vehicle:
            - Add distance actions for moving the vehicle to the goal
            - Add 1 action for dropping
    2. For vehicles that need to move to pick up packages:
        a. Add the distance from their current location to the package location
    3. The total heuristic is the sum of all these actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build road graph for distance calculations
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = set()
                if l2 not in self.road_graph:
                    self.road_graph[l2] = set()
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1)

        # Store capacity information
        self.capacity_predecessors = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_predecessors[s2] = s1

    def _shortest_path_distance(self, start, goal):
        """Calculate shortest path distance between two locations using BFS."""
        if start == goal:
            return 0

        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == goal:
                return dist
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        
        # If no path found, return a large number (but finite)
        return float('inf')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current locations of packages and vehicles
        current_locations = {}
        in_vehicle = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                if obj.startswith("v"):  # It's a vehicle
                    vehicle_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif parts[0] == "capacity":
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            if package not in current_locations:
                continue  # Package not in this state (shouldn't happen)
                
            current_loc = current_locations[package]
            if current_loc == goal_loc:
                continue  # Package already at goal

            # If package is in a vehicle, use vehicle's location
            if package in in_vehicle:
                vehicle = in_vehicle[package]
                current_loc = vehicle_locations[vehicle]
                # Cost to move vehicle to goal and drop package
                distance = self._shortest_path_distance(current_loc, goal_loc)
                total_cost += distance + 1  # +1 for drop action
            else:
                # Package is not in a vehicle
                # Find nearest vehicle that can pick it up
                min_pickup_cost = float('inf')
                
                for vehicle, vehicle_loc in vehicle_locations.items():
                    # Check if vehicle has capacity to pick up
                    if vehicle in vehicle_capacities:
                        size = vehicle_capacities[vehicle]
                        if size in self.capacity_predecessors:  # Can pick up at least one package
                            # Cost to move vehicle to package, pick up, move to goal, and drop
                            move_to_package = self._shortest_path_distance(vehicle_loc, current_loc)
                            move_to_goal = self._shortest_path_distance(current_loc, goal_loc)
                            cost = move_to_package + 1 + move_to_goal + 1
                            if cost < min_pickup_cost:
                                min_pickup_cost = cost
                
                if min_pickup_cost != float('inf'):
                    total_cost += min_pickup_cost
                else:
                    # No vehicle can pick up - use direct distance as fallback
                    distance = self._shortest_path_distance(current_loc, goal_loc)
                    total_cost += distance * 2  # Approximate with drive actions

        return total_cost
