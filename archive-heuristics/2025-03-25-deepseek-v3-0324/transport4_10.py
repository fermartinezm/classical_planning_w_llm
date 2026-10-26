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
    - The distance packages need to be moved (using road network distances)
    - Whether vehicles need to pick up/drop off packages
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can only be transported by vehicles (not moved directly)
    - Vehicles can carry multiple packages if they have sufficient capacity
    - The road network is bidirectional (if (road l1 l2) exists, movement is possible both ways)
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Store capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal:
        a) Find the nearest vehicle that can carry it (considering capacity)
        b) Calculate the distance from vehicle's current location to package
        c) Calculate the distance from package's current location to goal
        d) Add costs for pick-up and drop-off actions
    2. For vehicles:
        a) If already carrying packages, account for drop-off actions
        b) If capacity is limited, account for multiple trips
    3. Sum all estimated actions:
        - Drive actions (distance between locations)
        - Pick-up actions (1 per package)
        - Drop actions (1 per package)
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
        
        # Build road network graph
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
        
        # Extract capacity information
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
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        
        # If no path found, return a large number (but finite)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        
        # Check if goal is already reached
        if all(goal in state for goal in self.goals):
            return 0
            
        # Track current locations and capacities
        current_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}
        package_locations = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                if obj.startswith("p"):  # It's a package
                    package_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(package)
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                # Count how many packages the vehicle can still carry
                capacity = 0
                current_size = size
                while current_size in self.capacity_predecessors:
                    capacity += 1
                    current_size = self.capacity_predecessors[current_size]
                vehicle_capacities[vehicle] = capacity
        
        total_cost = 0
        
        # For each package not at its goal
        for package, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(package, None)
            
            # Package is already at goal
            if current_loc == goal_loc:
                continue
                
            # Package is in a vehicle
            if package in current_locations and current_locations[package].startswith("v"):
                vehicle = current_locations[package]
                vehicle_loc = current_locations[vehicle]
                # Cost to drop at goal
                total_cost += 1  # drop action
                # Distance from current vehicle location to goal
                dist = self._bfs_distance(vehicle_loc, goal_loc)
                total_cost += dist
            else:
                # Package is at a location, needs to be picked up
                package_loc = current_locations[package]
                
                # Find nearest vehicle with capacity
                min_dist = float('inf')
                best_vehicle = None
                
                for vehicle in vehicle_capacities:
                    if vehicle_capacities[vehicle] > 0:
                        vehicle_loc = current_locations[vehicle]
                        dist = self._bfs_distance(vehicle_loc, package_loc)
                        if dist < min_dist:
                            min_dist = dist
                            best_vehicle = vehicle
                
                if best_vehicle:
                    # Cost to pick up
                    total_cost += 1  # pick-up action
                    # Distance to drive to package
                    total_cost += min_dist
                    # Distance from package to goal
                    dist_to_goal = self._bfs_distance(package_loc, goal_loc)
                    total_cost += dist_to_goal
                    # Cost to drop
                    total_cost += 1  # drop action
                    # Reduce vehicle capacity
                    vehicle_capacities[best_vehicle] -= 1
                else:
                    # No available vehicle, use worst-case estimate
                    total_cost += 1000  # Large penalty
        
        return total_cost
