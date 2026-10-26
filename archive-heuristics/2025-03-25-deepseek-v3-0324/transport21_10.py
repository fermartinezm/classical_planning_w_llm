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
    - The capacity constraints of vehicles
    - The need to load/unload packages

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - Road connections are bidirectional (though this is handled via static facts)
    - Vehicles can carry multiple packages if they have sufficient capacity
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a road network graph from static facts
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) Find the shortest path from current location to goal location
        b) Estimate the driving actions needed (one per road segment)
        c) Add pick-up and drop-off actions (one each)
        d) If vehicle capacity is limited, account for multiple trips
    2. For vehicles:
        a) Estimate movement needed to reach packages
        b) Account for capacity constraints when multiple packages need transport
    3. Combine all estimates with appropriate weights
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
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
        
        # Extract capacity information
        self.capacity_predecessors = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_predecessors[s2] = s1

    def _shortest_path_length(self, start, goal):
        """BFS to find shortest path length between two locations."""
        if start == goal:
            return 0
            
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            if current == goal:
                return distance
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.road_graph.get(current, []):
                queue.append((neighbor, distance + 1))
        
        # If no path found (shouldn't happen in valid problems)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        
        # If all goals are satisfied, return 0
        if all(goal in state for goal in self.goals):
            return 0
            
        # Track package and vehicle locations
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        loaded_packages = {}  # vehicle: set of packages
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):  # package
                    package_locations[obj] = loc
                else:  # vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                pkg, vehicle = parts[1], parts[2]
                loaded_packages.setdefault(vehicle, set()).add(pkg)
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                # Convert size to numerical value by counting predecessors
                capacity = 0
                current = size
                while current in self.capacity_predecessors:
                    current = self.capacity_predecessors[current]
                    capacity += 1
                vehicle_capacities[vehicle] = capacity
        
        total_cost = 0
        
        # Process each package not at its goal
        for package, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(package)
            
            # If package is already at goal, skip
            if current_loc == goal_loc:
                continue
                
            # If package is in a vehicle, get vehicle's location
            in_vehicle = None
            for vehicle, packages in loaded_packages.items():
                if package in packages:
                    in_vehicle = vehicle
                    break
            
            if in_vehicle:
                current_loc = vehicle_locations[in_vehicle]
                # Cost to drop the package (1 action)
                total_cost += 1
                # If we're already at goal location, no more actions needed
                if current_loc == goal_loc:
                    continue
            
            # Find closest vehicle that can carry this package
            min_vehicle_cost = float('inf')
            for vehicle, vehicle_loc in vehicle_locations.items():
                # Skip vehicles that are full
                if len(loaded_packages.get(vehicle, set())) >= vehicle_capacities.get(vehicle, 1):
                    continue
                    
                # Cost to move vehicle to package and then to goal
                move_to_package = self._shortest_path_length(vehicle_loc, current_loc)
                move_to_goal = self._shortest_path_length(current_loc, goal_loc)
                vehicle_cost = move_to_package + move_to_goal + 2  # +2 for pick and drop
                
                if vehicle_cost < min_vehicle_cost:
                    min_vehicle_cost = vehicle_cost
            
            if min_vehicle_cost != float('inf'):
                total_cost += min_vehicle_cost
            else:
                # No available vehicle found - use pessimistic estimate
                total_cost += self._shortest_path_length(current_loc, goal_loc) + 2
        
        return total_cost
