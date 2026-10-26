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
    - Whether packages need to be picked up or dropped
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - The road network is bidirectional (though this is encoded in the static facts)
    - Vehicles can carry multiple packages if they have sufficient capacity
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract road network information from static facts
    - Extract goal locations for each package
    - Extract capacity predecessor relationships

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) If package is not in a vehicle:
            - Find closest vehicle that can carry it (considering capacity)
            - Estimate distance from vehicle to package
            - Add pick-up action
        b) If package is in a vehicle:
            - Check if vehicle has capacity to carry it (may need to drop others)
        c) Estimate distance from current location to goal
        d) Add drop action if needed
    2. Sum all estimated actions:
        - Drive actions (based on road distances)
        - Pick-up and drop actions
        - Capacity adjustment actions if needed
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract road network
        self.roads = set()
        # Extract capacity predecessor relationships
        self.capacity_pred = {}
        
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "road":
                self.roads.add((parts[1], parts[2]))
            elif parts[0] == "capacity-predecessor":
                self.capacity_pred[parts[1]] = parts[2]
        
        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # If we're already in a goal state, return 0
        if self.goals <= state:
            return 0
            
        # Track package locations and vehicle info
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj.startswith("p"):  # Package
                    package_locations[obj] = loc
                else:  # Vehicle
                    vehicle_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(package)
            elif parts[0] == "capacity":
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size
        
        total_cost = 0
        
        # For each package not at its goal
        for package, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(package)
            
            # Skip if already at goal
            if current_loc == goal_loc:
                continue
                
            # Check if package is in a vehicle
            in_vehicle = False
            containing_vehicle = None
            for vehicle, packages in packages_in_vehicles.items():
                if package in packages:
                    in_vehicle = True
                    containing_vehicle = vehicle
                    break
            
            if in_vehicle:
                # Package is in a vehicle - need to drive to goal and drop
                vehicle_loc = vehicle_locations[containing_vehicle]
                
                # Estimate distance from vehicle's current location to goal
                distance = self.estimate_distance(vehicle_loc, goal_loc)
                total_cost += distance
                
                # Add drop action
                total_cost += 1
            else:
                # Package is not in a vehicle - need to pick up and transport
                # Find closest vehicle that can carry it
                min_distance = float('inf')
                best_vehicle = None
                
                for vehicle, vehicle_loc in vehicle_locations.items():
                    # Check if vehicle has capacity
                    if vehicle in vehicle_capacities:
                        # Simple check - if capacity isn't c0, it can carry something
                        if vehicle_capacities[vehicle] != "c0":
                            distance = self.estimate_distance(vehicle_loc, current_loc)
                            if distance < min_distance:
                                min_distance = distance
                                best_vehicle = vehicle
                
                if best_vehicle:
                    total_cost += min_distance  # Drive to package
                    total_cost += 1  # Pick-up action
                    
                    # Now estimate distance from package location to goal
                    distance_to_goal = self.estimate_distance(current_loc, goal_loc)
                    total_cost += distance_to_goal
                    
                    total_cost += 1  # Drop action
        
        return total_cost
    
    def estimate_distance(self, loc1, loc2):
        """
        Estimate the minimal number of drive actions needed to get from loc1 to loc2.
        Uses a simple BFS to find the shortest path in the road network.
        """
        if loc1 == loc2:
            return 0
            
        visited = set()
        queue = [(loc1, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            if current == loc2:
                return distance
                
            visited.add(current)
            
            # Check all connected locations
            for (l1, l2) in self.roads:
                if l1 == current and l2 not in visited:
                    queue.append((l2, distance + 1))
                elif l2 == current and l1 not in visited:
                    queue.append((l1, distance + 1))
        
        # If no path found (shouldn't happen in valid problems)
        return float('inf')
