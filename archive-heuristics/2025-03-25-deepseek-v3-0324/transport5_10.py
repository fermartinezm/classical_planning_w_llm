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
    - The road network is bidirectional (though this is handled by the static facts)
    - Vehicles can carry multiple packages if their capacity allows
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Store capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal:
        a) If package is in a vehicle:
            - Add cost to drop it at current vehicle location
            - Then treat as if it's at that location
        b) Find nearest vehicle that can carry it (considering capacity)
        c) Calculate distance from package to vehicle and from vehicle to goal
        d) Add costs for:
            - Vehicle moving to package (if needed)
            - Picking up package
            - Vehicle moving to goal location
            - Dropping package
    2. Sum costs for all packages, sharing vehicle movements when possible
    3. Add penalty if vehicles need to make multiple trips due to capacity
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
        
        # Build road graph (bidirectional)
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
        
        # Store capacity information
        self.capacity_predecessors = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_predecessors[s2] = s1

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
        
        # If no path found, return large number (but finite)
        return 1000

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # If all goals are satisfied, return 0
        if self.goals <= state:
            return 0
            
        # Extract current state information
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("p"):  # Package
                    package_locations[obj] = loc
                else:  # Vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                packages_in_vehicles[package] = vehicle
        
        total_cost = 0
        
        # Process each package that needs to be moved
        for package, goal_loc in self.goal_locations.items():
            # If package is already at goal, skip
            if package in package_locations and package_locations[package] == goal_loc:
                continue
                
            # If package is in a vehicle, we need to drop it first
            if package in packages_in_vehicles:
                vehicle = packages_in_vehicles[package]
                current_loc = vehicle_locations[vehicle]
                total_cost += 1  # Drop action
                # Now treat as if package is at vehicle's location
                package_locations[package] = current_loc
            
            # Find current location (must exist since we're not at goal)
            current_loc = package_locations[package]
            
            # Find nearest vehicle that can carry this package
            min_vehicle_cost = float('inf')
            best_vehicle = None
            best_pickup_loc = None
            
            for vehicle, vehicle_loc in vehicle_locations.items():
                # Check if vehicle has capacity
                if vehicle not in vehicle_capacities:
                    continue  # Vehicle at max capacity?
                    
                # Current size must have a predecessor (can pick up)
                current_size = vehicle_capacities[vehicle]
                if current_size not in self.capacity_predecessors:
                    continue  # Can't pick up any more
                    
                # Calculate distances
                dist_vehicle_to_pkg = self._bfs_distance(vehicle_loc, current_loc)
                dist_pkg_to_goal = self._bfs_distance(current_loc, goal_loc)
                
                # Total cost for this vehicle:
                # Drive to package + pick up + drive to goal + drop
                cost = dist_vehicle_to_pkg + 1 + dist_pkg_to_goal + 1
                
                if cost < min_vehicle_cost:
                    min_vehicle_cost = cost
                    best_vehicle = vehicle
                    best_pickup_loc = current_loc
            
            if best_vehicle is None:
                # No vehicle can carry this package (shouldn't happen in valid states)
                return float('inf')
                
            total_cost += min_vehicle_cost
            
            # Update vehicle location (simulate it moving to goal)
            vehicle_locations[best_vehicle] = goal_loc
            
            # Reduce vehicle capacity (simulate picking up package)
            current_size = vehicle_capacities[best_vehicle]
            vehicle_capacities[best_vehicle] = self.capacity_predecessors[current_size]
        
        return total_cost
