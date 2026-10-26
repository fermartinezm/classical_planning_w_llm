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
    - Whether vehicles need to pick up/drop packages
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can only be transported by vehicles (not moved directly)
    - Vehicles can carry multiple packages if they have sufficient capacity
    - Road connections are bidirectional (though this isn't strictly required)
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package from task.goals
    - Build a road graph from static facts for pathfinding
    - Extract capacity information from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) Find the vehicle closest to the package that has capacity to carry it
        b) Calculate the distance from that vehicle to the package
        c) Calculate the distance from package's current location to goal location
        d) Add costs for pick-up and drop actions
    2. For vehicles:
        a) If a vehicle is carrying packages, add cost to drop them at goals
        b) If a vehicle is at a location with packages needing transport, add pick-up cost
    3. Sum all these costs to get the total heuristic estimate
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Build road graph
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
        
        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
        
        # Store capacity predecessor relationships
        self.capacity_pred = set()
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred.add((s1, s2))

    def _shortest_path(self, start, end):
        """BFS to find shortest path between two locations using road connections."""
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
        
        # If no path found, return large number (but finite)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # Check if goal is already reached
        if all(goal in state for goal in self.goals):
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
                if obj.startswith('p'):  # Package
                    package_locations[obj] = loc
                else:  # Vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(package)
        
        total_cost = 0
        
        # For each package not at its goal
        for package, current_loc in package_locations.items():
            goal_loc = self.goal_locations.get(package)
            if not goal_loc or current_loc == goal_loc:
                continue
            
            # Find closest vehicle that can carry this package
            min_dist = float('inf')
            best_vehicle = None
            
            for vehicle, vehicle_loc in vehicle_locations.items():
                # Check if vehicle has capacity
                if vehicle in vehicle_capacities:
                    # Find if there's a capacity that allows picking up
                    current_size = vehicle_capacities[vehicle]
                    has_capacity = any(s1 == current_size for s1, s2 in self.capacity_pred)
                    
                    if has_capacity:
                        dist = self._shortest_path(vehicle_loc, current_loc)
                        if dist < min_dist:
                            min_dist = dist
                            best_vehicle = vehicle
            
            if best_vehicle:
                # Cost to move vehicle to package
                total_cost += min_dist
                # Cost for pick-up action
                total_cost += 1
                
                # Cost to move from package location to goal
                transport_dist = self._shortest_path(current_loc, goal_loc)
                total_cost += transport_dist
                # Cost for drop action
                total_cost += 1
        
        # For packages already in vehicles
        for vehicle, packages in packages_in_vehicles.items():
            vehicle_loc = vehicle_locations.get(vehicle)
            if not vehicle_loc:
                continue
                
            for package in packages:
                goal_loc = self.goal_locations.get(package)
                if not goal_loc:
                    continue
                
                # Cost to move to goal location
                dist = self._shortest_path(vehicle_loc, goal_loc)
                total_cost += dist
                # Cost for drop action
                total_cost += 1
        
        return total_cost
