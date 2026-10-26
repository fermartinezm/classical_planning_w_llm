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
    - Whether packages need to be picked up or dropped

    # Assumptions:
    - Each package can be carried by any vehicle with sufficient capacity
    - Road connections are bidirectional (though this is handled by the road facts)
    - Vehicles may need to make multiple trips if their capacity is limited
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package from task.goals
    - Build a road graph from static facts for pathfinding
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a. Find its current location (either directly or via the vehicle carrying it)
        b. Find the closest vehicle that can carry it (considering capacity)
        c. Calculate the distance from vehicle to package (if not already in vehicle)
        d. Calculate the distance from package's location to goal location
        e. Add costs for pick-up and drop actions
    2. Sum all these costs to get the total heuristic estimate
    3. For packages already at their goal, add 0 to the total
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Build road graph from static facts
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
        
        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

    def _bfs_distance(self, start, goal):
        """Calculate the shortest path distance between two locations using BFS."""
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
        
        # If no path found, return a large number (but finite)
        return float('inf')

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        total_cost = 0
        
        # Track package locations and vehicle capacities
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        in_vehicle = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):  # Package
                    package_locations[obj] = loc
                else:  # Vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                # Convert size to numeric value (c0=0, c1=1, etc.)
                capacity = int(size[1:])
                vehicle_capacities[vehicle] = capacity
        
        for package, goal_loc in self.goal_locations.items():
            # Check if package is already at goal
            if package in package_locations and package_locations[package] == goal_loc:
                continue
                
            # Current location of package (either directly or via vehicle)
            if package in in_vehicle:
                # Package is in a vehicle - use vehicle's location
                vehicle = in_vehicle[package]
                current_loc = vehicle_locations[vehicle]
                # Cost to drop the package at goal
                dist = self._bfs_distance(current_loc, goal_loc)
                total_cost += dist + 1  # 1 for drop action
            else:
                # Package is at a location - need to pick it up first
                current_loc = package_locations[package]
                
                # Find closest vehicle that can carry it (capacity > 0)
                min_dist = float('inf')
                best_vehicle = None
                
                for vehicle, capacity in vehicle_capacities.items():
                    if capacity > 0:  # Vehicle has capacity
                        vehicle_loc = vehicle_locations[vehicle]
                        dist = self._bfs_distance(vehicle_loc, current_loc)
                        if dist < min_dist:
                            min_dist = dist
                            best_vehicle = vehicle
                
                if best_vehicle is None:
                    # No vehicle can carry it - large penalty
                    total_cost += 1000
                    continue
                    
                # Cost to pick up and transport to goal
                dist_to_goal = self._bfs_distance(current_loc, goal_loc)
                total_cost += min_dist + dist_to_goal + 2  # +2 for pick and drop
        
        return total_cost
