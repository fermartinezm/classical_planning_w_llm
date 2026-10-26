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
    - The distance packages need to be moved (using a simple count of required road segments)
    - Whether packages need to be picked up or dropped
    - Vehicle capacity constraints

    # Assumptions:
    - The road network is bidirectional (if (road l1 l2) exists, we can travel both ways)
    - Vehicles can carry multiple packages if they have sufficient capacity
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) If the package is not in a vehicle:
            - Find the nearest vehicle that can carry it (considering capacity)
            - Estimate distance from package to vehicle and from vehicle to goal
            - Add pick-up and drop actions
        b) If the package is in a vehicle:
            - Estimate distance from vehicle's current location to goal
            - Add drop action
    2. For vehicles carrying packages:
        - Add drive actions needed to reach package goals
    3. Sum all estimated actions
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
        
        # Build road graph (bidirectional)
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
        
        # Extract capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # If we're already in a goal state, return 0
        if self.goals <= state:
            return 0
        
        # Track package and vehicle locations
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
            elif match(fact, "in", "*", "*"):
                pkg, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(pkg)
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
        
        total_cost = 0
        
        # For each package not at its goal
        for package, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(package, None)
            
            # If package is already at goal, skip
            if current_loc == goal_loc:
                continue
                
            # Find which vehicle is carrying this package (if any)
            carrying_vehicle = None
            for vehicle, packages in packages_in_vehicles.items():
                if package in packages:
                    carrying_vehicle = vehicle
                    break
            
            if carrying_vehicle:
                # Package is in a vehicle - need to drive to goal and drop
                vehicle_loc = vehicle_locations[carrying_vehicle]
                
                # Estimate driving distance (minimum 1 action)
                dist = self._estimate_distance(vehicle_loc, goal_loc)
                total_cost += dist
                
                # Add drop action
                total_cost += 1
            else:
                # Package is not in a vehicle - need to pick up first
                # Find nearest vehicle that can carry it
                min_dist = float('inf')
                best_vehicle = None
                
                for vehicle, vehicle_loc in vehicle_locations.items():
                    # Check if vehicle has capacity
                    capacity = vehicle_capacities.get(vehicle, None)
                    if capacity and capacity != "c0":  # Has some capacity
                        dist = self._estimate_distance(vehicle_loc, current_loc)
                        if dist < min_dist:
                            min_dist = dist
                            best_vehicle = vehicle
                
                if best_vehicle:
                    # Add drive distance to package
                    total_cost += min_dist
                    
                    # Add pick-up action
                    total_cost += 1
                    
                    # Now drive to goal location
                    goal_dist = self._estimate_distance(current_loc, goal_loc)
                    total_cost += goal_dist
                    
                    # Add drop action
                    total_cost += 1
        
        return total_cost
    
    def _estimate_distance(self, start, end):
        """Estimate the minimum number of drive actions between two locations."""
        if start == end:
            return 0
            
        # Simple BFS to find shortest path length
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
        
        # If no path found (shouldn't happen in valid problems), return large number
        return 1000
