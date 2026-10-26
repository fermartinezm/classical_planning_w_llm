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
    - The distance packages need to be moved (using road connections)
    - Whether packages need to be picked up or dropped
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - Vehicles can share the transportation workload
    - Road connections are bidirectional (though this isn't strictly required)
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a road graph for distance calculations
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a. If the package is in a vehicle:
            - Add cost for dropping it at current vehicle location
            - Treat it as being at the vehicle's location
        b. Find the closest vehicle that can carry the package (considering capacity)
        c. Calculate the distance from package to vehicle and vehicle to goal
        d. Add costs for:
            - Moving vehicle to package (if needed)
            - Picking up package
            - Moving vehicle to goal location
            - Dropping package
    2. Sum all these costs to get the total heuristic estimate
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

        # Build road graph for distance calculations
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)

        # Extract capacity information
        self.capacity_predecessors = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_predecessors[s2] = s1

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # If we're already in a goal state, return 0
        if self.goals <= state:
            return 0

        # Track current locations of packages and vehicles
        current_locations = {}
        in_vehicle = {}
        vehicle_capacities = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                pkg, vehicle = parts[1], parts[2]
                in_vehicle[pkg] = vehicle
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            # Skip if package is already at goal
            if package in current_locations and current_locations[package] == goal_loc:
                continue

            # If package is in a vehicle, we need to drop it first
            if package in in_vehicle:
                vehicle = in_vehicle[package]
                current_loc = current_locations[vehicle]
                total_cost += 1  # Cost for drop action
            else:
                current_loc = current_locations[package]

            # Find closest vehicle that can carry this package
            min_vehicle_cost = float('inf')
            best_vehicle = None
            best_vehicle_loc = None

            for vehicle, capacity in vehicle_capacities.items():
                # Check if vehicle has sufficient capacity to carry the package
                # (We assume any capacity > c0 can carry any package)
                if capacity != 'c0':
                    vehicle_loc = current_locations[vehicle]
                    
                    # Calculate distance from vehicle to package and package to goal
                    dist_vehicle_to_pkg = self._shortest_path_distance(vehicle_loc, current_loc)
                    dist_pkg_to_goal = self._shortest_path_distance(current_loc, goal_loc)
                    
                    total_vehicle_cost = (dist_vehicle_to_pkg +  # drive to package
                                         1 +                     # pick-up
                                         dist_pkg_to_goal +      # drive to goal
                                         1)                      # drop
                    
                    if total_vehicle_cost < min_vehicle_cost:
                        min_vehicle_cost = total_vehicle_cost
                        best_vehicle = vehicle
                        best_vehicle_loc = vehicle_loc

            if best_vehicle is not None:
                total_cost += min_vehicle_cost
            else:
                # No vehicle can carry this package - return a large penalty
                return float('inf')

        return total_cost

    def _shortest_path_distance(self, start, end):
        """Calculate the shortest path distance between two locations using BFS."""
        if start == end:
            return 0

        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.road_graph.get(current, []):
                queue.append((neighbor, dist + 1))
        
        # If no path exists, return a large number
        return float('inf')
