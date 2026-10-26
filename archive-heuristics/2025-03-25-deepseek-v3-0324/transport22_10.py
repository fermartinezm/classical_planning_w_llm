from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
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
    - Packages can only be transported by vehicles
    - Vehicles have limited capacity that decreases when carrying packages
    - Roads are bidirectional (if l1-l2 exists, l2-l1 exists)
    - The heuristic doesn't need to be admissible

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a road graph for distance calculations
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) If package is not in a vehicle:
            - Add cost for picking it up (1 action)
            - Find nearest vehicle that can carry it (using road distances)
            - Add driving cost for vehicle to reach package (distance)
        b) If package is in a vehicle:
            - Check if vehicle is at goal location
            - If not, add driving cost to goal (distance)
        c) Add cost for dropping package at goal (1 action)
    2. For packages already at goal locations:
        - If they're in a vehicle, add cost to drop them (1 action)
    3. Sum all these costs to get the total heuristic estimate
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Store goal locations for each package
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
        
        # If no path found (shouldn't happen in valid problems)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        total_cost = 0
        
        # Track package and vehicle locations
        package_locations = {}
        vehicle_locations = {}
        in_vehicle = {}
        vehicle_capacities = {}
        
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
                vehicle_capacities[vehicle] = size
        
        for package, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(package, None)
            
            # Package is already at goal location
            if current_loc == goal_loc and package not in in_vehicle:
                continue
                
            # Package is in a vehicle
            if package in in_vehicle:
                vehicle = in_vehicle[package]
                vehicle_loc = vehicle_locations[vehicle]
                
                # If vehicle is not at goal, add driving cost
                if vehicle_loc != goal_loc:
                    dist = self._bfs_distance(vehicle_loc, goal_loc)
                    total_cost += dist
                
                # Always need to drop the package
                total_cost += 1
            
            # Package is not in a vehicle
            else:
                # Find nearest vehicle that can carry this package
                min_dist = float('inf')
                best_vehicle = None
                
                for vehicle, loc in vehicle_locations.items():
                    # Check if vehicle has capacity to pick up package
                    if vehicle in vehicle_capacities and self.capacity_predecessors.get(vehicle_capacities[vehicle], None):
                        dist = self._bfs_distance(loc, current_loc)
                        if dist < min_dist:
                            min_dist = dist
                            best_vehicle = vehicle
                
                if best_vehicle is not None:
                    # Add cost for vehicle to reach package
                    total_cost += min_dist
                    # Add cost for pick-up action
                    total_cost += 1
                    # Add cost for driving to goal
                    dist_to_goal = self._bfs_distance(current_loc, goal_loc)
                    total_cost += dist_to_goal
                    # Add cost for drop action
                    total_cost += 1
                else:
                    # No available vehicle found (problem might be unsolvable)
                    return float('inf')
        
        return total_cost
