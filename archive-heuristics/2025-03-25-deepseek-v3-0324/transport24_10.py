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
    - Whether vehicles need to pick up/drop off packages
    - Vehicle capacity constraints

    # Assumptions:
    - The road network is bidirectional (if (road l1 l2) exists, movement is possible both ways)
    - Packages can only be transported by vehicles
    - Each vehicle can carry multiple packages (depending on capacity)
    - The heuristic doesn't need to be admissible

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a. Find the shortest path distance from current location to goal location
        b. If package is not in a vehicle:
            - Add cost for picking it up (1 action)
            - Add cost for dropping it at goal (1 action)
        c. If package is in a vehicle:
            - Check if vehicle is at package's current location
            - Add cost for dropping it at goal (1 action)
        d. Add the movement cost (distance * 1 per step)
    2. For vehicles that need to transport packages:
        a. Add cost for moving to package locations
        b. Add cost for moving to goal locations
    3. Consider vehicle capacity constraints:
        - If a vehicle needs to make multiple trips due to capacity, add additional costs
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
        
        # Build road graph
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)  # roads are bidirectional
        
        # Extract capacity predecessor information
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def shortest_path_distance(self, start, goal):
        """Compute shortest path distance between two locations using BFS."""
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
            
            for neighbor in self.road_graph.get(current, set()):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        
        # If no path found (shouldn't happen in valid problems)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state
        
        # Check if goal is already reached
        if all(goal in state for goal in self.goals):
            return 0
            
        # Track package and vehicle locations
        package_locations = {}
        vehicle_locations = {}
        in_vehicle = {}
        vehicle_capacities = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):  # package
                    package_locations[obj] = loc
                else:  # vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
        
        total_cost = 0
        
        # Process each package
        for package, goal_loc in self.goal_locations.items():
            if package not in package_locations:
                continue  # package is in a vehicle
                
            current_loc = package_locations.get(package)
            if current_loc == goal_loc:
                continue  # package is already at goal
                
            # Find closest vehicle that can carry this package
            min_vehicle_dist = float('inf')
            best_vehicle = None
            
            for vehicle, vehicle_loc in vehicle_locations.items():
                # Check if vehicle has capacity
                if vehicle in vehicle_capacities and self.capacity_pred.get(vehicle_capacities[vehicle]):
                    dist = self.shortest_path_distance(vehicle_loc, current_loc)
                    if dist < min_vehicle_dist:
                        min_vehicle_dist = dist
                        best_vehicle = vehicle
            
            if best_vehicle is None:
                continue  # no vehicle can carry this package
                
            # Cost to move vehicle to package
            total_cost += min_vehicle_dist
            
            # Cost to pick up package
            total_cost += 1
            
            # Cost to move to goal location
            goal_dist = self.shortest_path_distance(current_loc, goal_loc)
            total_cost += goal_dist
            
            # Cost to drop package
            total_cost += 1
        
        # Add additional costs for packages already in vehicles
        for package, vehicle in in_vehicle.items():
            if package not in self.goal_locations:
                continue
                
            goal_loc = self.goal_locations[package]
            vehicle_loc = vehicle_locations.get(vehicle)
            
            if vehicle_loc is None:
                continue
                
            # Cost to move to goal location
            goal_dist = self.shortest_path_distance(vehicle_loc, goal_loc)
            total_cost += goal_dist
            
            # Cost to drop package
            total_cost += 1
        
        return total_cost
