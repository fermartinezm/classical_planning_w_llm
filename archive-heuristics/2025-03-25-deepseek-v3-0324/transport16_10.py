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
    - The road network is bidirectional (though this is handled via static facts)
    - Vehicles can carry multiple packages if their capacity allows
    - Packages can only be transported by vehicles (not on their own)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a road graph from static facts for pathfinding
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal:
        a) If package is in a vehicle:
            - Add cost to drop it at current location (if needed)
            - Calculate path from vehicle's current location to package's goal
        b) If package is at a location:
            - Find nearest vehicle that can carry it (considering capacity)
            - Calculate path from vehicle's current location to package
            - Add pick-up cost
            - Calculate path from package location to goal
            - Add drop cost
    2. Sum all these costs with appropriate weights
    3. For vehicles that need to transport multiple packages, share the driving costs
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
        
        # Build road graph
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
        
        # Store capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def _bfs_distance(self, start, goal):
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
                queue.append((neighbor, dist + 1))
        
        # If no path found, return large number
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # Check if goal is already reached
        if all(goal in state for goal in self.goals):
            return 0
            
        # Track package locations and vehicle info
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    package_locations[obj] = loc
                elif obj.startswith('v'):
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                pkg, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(pkg)
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
        
        total_cost = 0
        
        for package, goal_loc in self.goal_locations.items():
            # Skip if package is already at goal
            if package in package_locations and package_locations[package] == goal_loc:
                continue
                
            # Package is in a vehicle
            in_vehicle = None
            for vehicle, packages in packages_in_vehicles.items():
                if package in packages:
                    in_vehicle = vehicle
                    break
            
            if in_vehicle:
                # Need to drop package at goal
                current_vehicle_loc = vehicle_locations[in_vehicle]
                dist = self._bfs_distance(current_vehicle_loc, goal_loc)
                total_cost += dist + 1  # +1 for drop action
            else:
                # Package is at a location - need to pick up and transport
                current_pkg_loc = package_locations[package]
                
                # Find best vehicle to use (nearest with capacity)
                best_cost = float('inf')
                for vehicle, vehicle_loc in vehicle_locations.items():
                    # Check if vehicle has capacity
                    current_size = vehicle_capacities.get(vehicle, None)
                    if current_size is None or current_size == 'c0':
                        continue  # No capacity
                        
                    # Calculate cost to pick up and transport
                    dist_to_pkg = self._bfs_distance(vehicle_loc, current_pkg_loc)
                    dist_to_goal = self._bfs_distance(current_pkg_loc, goal_loc)
                    cost = dist_to_pkg + 1 + dist_to_goal + 1  # +1 for pick and drop
                    
                    if cost < best_cost:
                        best_cost = cost
                
                if best_cost != float('inf'):
                    total_cost += best_cost
                else:
                    # No vehicle can transport this package - large penalty
                    total_cost += 1000
        
        return total_cost
