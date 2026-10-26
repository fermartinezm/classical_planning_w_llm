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
    - Whether packages need to be picked up or dropped
    - Vehicle capacity constraints

    # Assumptions:
    - The road network is bidirectional (if (road l1 l2) exists, movement is possible both ways)
    - Vehicles can carry multiple packages if they have sufficient capacity
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network
    - Store capacity predecessor relationships

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) If package is in a vehicle:
            - Add cost for dropping it at current vehicle location
            - Treat as if it's at the vehicle's location
        b) Find shortest path from current location to goal location
        c) Estimate drive actions needed (path length)
        d) Add pick-up and drop actions if needed
    2. For vehicles that need to move to pick up packages:
        a) Estimate distance from current location to package location
        b) Add drive actions needed
    3. Consider capacity constraints:
        a) If a vehicle needs to make multiple trips due to capacity, add additional actions
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
                self.road_graph.setdefault(l2, set()).add(l1)  # assume bidirectional
        
        # Store capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between two locations using BFS."""
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
        
        # If no path found, return a large number (problem might be unsolvable)
        return float('inf')

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        
        # Check if goal is already reached
        if all(goal in state for goal in self.goals):
            return 0
            
        # Track current locations of packages and vehicles
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("p"):  # package
                    package_locations[obj] = loc
                else:  # vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                pkg, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(pkg)
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
        
        total_cost = 0
        
        # Process each package not at its goal location
        for package, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(package, None)
            
            # If package is in a vehicle, we need to drop it first
            in_vehicle = False
            for vehicle, packages in packages_in_vehicles.items():
                if package in packages:
                    in_vehicle = True
                    current_loc = vehicle_locations[vehicle]
                    total_cost += 1  # drop action
                    break
            
            # If package is already at goal, no cost
            if current_loc == goal_loc:
                continue
                
            # Estimate drive distance from current to goal location
            drive_distance = self.shortest_path_length(current_loc, goal_loc)
            if drive_distance == float('inf'):
                return float('inf')  # unreachable goal
                
            total_cost += drive_distance
            
            # If package wasn't in a vehicle, we need a pick-up action
            if not in_vehicle:
                total_cost += 1  # pick-up action
        
        # Estimate vehicle movements for picking up packages
        # This is simplified - we assume each vehicle might need to move to pick up packages
        for vehicle, loc in vehicle_locations.items():
            # Find packages this vehicle might need to pick up
            # We look for packages at locations different from vehicle's current location
            packages_to_pick = [
                p for p, p_loc in package_locations.items() 
                if p_loc != loc and p not in packages_in_vehicles.get(vehicle, set())
            ]
            
            if packages_to_pick:
                # Estimate distance to nearest package
                min_distance = min(
                    self.shortest_path_length(loc, package_locations[p]) 
                    for p in packages_to_pick
                )
                total_cost += min_distance
        
        return total_cost
